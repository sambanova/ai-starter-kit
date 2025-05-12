import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

sys.path.append('../')
sys.path.append('../src')
sys.path.append('../prompts')
sys.path.append('../src/llmperf')

# Get the absolute path of my_project
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

sys.path.insert(0, project_root)

import logging

from dotenv import load_dotenv

from benchmarking.src.performance_evaluation import RealWorkLoadPerformanceEvaluator, SyntheticPerformanceEvaluator
from benchmarking.utils import read_perf_eval_json_files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv(os.path.join(project_root, '.env'), override=True)


def get_grouping_and_batching_info(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Generate grouping and batching info from DataFrame."""
    df = df.sort_values('end_time').reset_index(drop=True)
    df['group'] = (df['server_ttft_s'] != df['server_ttft_s'].shift()).cumsum()

    consecutive_counts = df.groupby(['group', 'server_ttft_s']).size().reset_index(name='consecutive_count')
    requests_grouping = consecutive_counts['consecutive_count'].tolist()
    requests_batching = [1 << (x - 1).bit_length() for x in requests_grouping]

    return requests_grouping, requests_batching


def extract_file_info(file_name: str) -> Tuple[str, int, int, Optional[int], Optional[float]]:
    """Extract model, input, output, and concurrency from file name."""

    if 'multimodal' in file_name:
        _, _, model, _, _, in_tok, out_tok, con_type, *_ = file_name.split('_')
    else:
        _, _, model, in_tok, out_tok, con_type, *_ = file_name.split('_')

    if 'synthetic' in file_name:
        con = int(con_type)
        qps = None
    elif 'realworkload' in file_name:
        con = None
        qps = float(con_type.replace('-', '.'))

    return model, int(in_tok), int(out_tok), con, qps


# Read config file
CONFIG_FILE_PATH = os.path.join(current_dir, 'config.yaml')
with open(CONFIG_FILE_PATH) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
config['output_files_dir'] = os.path.expanduser(config['output_files_dir'])
config['model_configs_path'] = os.path.expanduser(config['model_configs_path'])
model_configs_df = pd.read_csv(config['model_configs_path'])

# Additional parameters:
run_time = datetime.now().strftime('%Y%m%d-%H%M%S.%f')
output_files_dir = os.path.join(config['output_files_dir'], run_time)
sampling_params: Dict[str, Any] = {}
user_metadata: Dict[str, Any] = {}

model_configs_df[["input_tokens", "output_tokens", "num_requests"]] = \
    model_configs_df[["input_tokens", "output_tokens", "num_requests"]].astype('Int64')
# Loop over models and configs to run performance evaluation
for idx, row in model_configs_df.iterrows():
    model_name = row['model_name']
    input_tokens = row['input_tokens']
    output_tokens = row['output_tokens']
    num_requests = row['num_requests']
    concurrent_requests = int(row['concurrent_requests']) if pd.notna(row['concurrent_requests']) else 0
    qps = float(row['qps']) if pd.notna(row['qps']) else 0.0
    qps_distribution = row['qps_distribution']
    multimodal_img_size = row['multimodal_img_size']
    multimodal_img_size = multimodal_img_size if pd.notna(multimodal_img_size) else 'na'

    cr_set = pd.notna(concurrent_requests) and concurrent_requests != 0
    qps_set = pd.notna(qps) and qps != 0

    if cr_set and qps_set:
        print(f"Row {idx}: {model_name}-{multimodal_img_size}-{input_tokens}-{output_tokens}-\
            {concurrent_requests}-{qps} is invalid: both 'concurrent_requests' and 'qps' are set.\
            Set only one of them, skipping run.")
        continue

    evaluator: SyntheticPerformanceEvaluator | RealWorkLoadPerformanceEvaluator

    if cr_set:
        logging.info(
            f'Running model_name {model_name}, input_tokens {input_tokens}, output_tokens {output_tokens}, '
            f'concurrent_requests {concurrent_requests}, num_requests {num_requests}, '
            f'multimodal_img_size {multimodal_img_size}'
        )

        # Instantiate evaluator
        user_metadata['model_idx'] = 0  # no need to change
        evaluator = SyntheticPerformanceEvaluator(
            multimodal_image_size=multimodal_img_size,
            model_name=model_name,
            results_dir=os.path.expanduser(output_files_dir),
            num_concurrent_requests=concurrent_requests,
            timeout=config['timeout'],
            user_metadata=user_metadata,
            llm_api=config['llm_api'],
        )

    elif qps_set:
        logging.info(
            f'Running model_name {model_name}, input_tokens {input_tokens}, output_tokens {output_tokens},'
            f'qps {qps}, qps_distribution {qps_distribution}, multimodal_img_size {multimodal_img_size}'
        )

        # Instantiate evaluator
        user_metadata['model_idx'] = 0  # no need to change
        evaluator = RealWorkLoadPerformanceEvaluator(
            multimodal_image_size=multimodal_img_size,
            model_name=model_name,
            results_dir=os.path.expanduser(output_files_dir),
            qps=qps,
            qps_distribution=qps_distribution,
            timeout=config['timeout'],
            user_metadata=user_metadata,
            llm_api=config['llm_api'],
        )

    else:
        logging.error(
            f'Set either concurrent_requests or qps for model_name {model_name}, \
            input_tokens {input_tokens}, output_tokens {output_tokens}. Skipping run.'
        )
        continue

    try:
        # Run performance evaluation
        model_results_summary, model_results_per_request = evaluator.run_benchmark(
            num_input_tokens=input_tokens,
            num_output_tokens=output_tokens,
            num_requests=num_requests,
            sampling_params=sampling_params,
        )
    except Exception as e:
        logging.error(f'Error while running model_name {model_name}, \
            input_tokens {input_tokens}, \
            output_tokens {output_tokens}, \
            num_requests {num_requests}, \
            concurrent_requests {concurrent_requests}, \
            qps {qps}, \
            qps_distribution {qps_distribution} \
            multimodal_img_size {multimodal_img_size}')
        logging.error(e)

    logging.info(f"Time delay: {config['time_delay']} seconds")
    time.sleep(config['time_delay'])

# Consolidate results
if config['consolidated_results_dir']:
    logging.info(f"Writting consolidated results to {config['consolidated_results_dir']}")
    try:
        # Read summary files
        df_summary = read_perf_eval_json_files(output_files_dir, type='summary')

        # Fill missing values
        df_summary['num_concurrent_requests'] = (
            None if 'num_concurrent_requests' not in df_summary.columns else df_summary['num_concurrent_requests']
        )
        df_summary['qps'] = None if 'qps' not in df_summary.columns else df_summary['qps']
        df_summary['qps'] = None if 'qps' not in df_summary.columns else df_summary['qps']
        df_summary['qps_distribution'] = (
            None if 'qps_distribution' not in df_summary.columns else df_summary['qps_distribution']
        )
        df_summary['multimodal_img_size'] = df_summary['name'].str.extract(
            r'multimodal_(small|medium|large)', expand=False
        )

        df_summary = df_summary[
            [
                'model',
                'num_input_tokens',
                'num_output_tokens',
                'num_concurrent_requests',
                'qps',
                'qps_distribution',
                'multimodal_img_size',
                'server_ttft_s_min',
                'server_ttft_s_p50',
                'server_ttft_s_max',
                'server_end_to_end_latency_s_min',
                'server_end_to_end_latency_s_p50',
                'server_end_to_end_latency_s_max',
                'server_output_token_per_s_min',
                'server_output_token_per_s_p50',
                'server_output_token_per_s_max',
                'acceptance_rate_min',
                'acceptance_rate_p50',
                'acceptance_rate_max',
                'server_number_input_tokens_p50',
                'server_number_output_tokens_p50',
                'client_ttft_s_min',
                'client_ttft_s_p50',
                'client_ttft_s_max',
                'client_end_to_end_latency_s_min',
                'client_end_to_end_latency_s_p50',
                'client_end_to_end_latency_s_max',
                'client_output_token_per_s_min',
                'client_output_token_per_s_p50',
                'client_output_token_per_s_max',
                'num_requests_started',
                'num_completed_requests',
                'number_errors',
                'error_code_frequency',
            ]
        ].copy()

        df_summary['model'] = df_summary['model'].str.replace('.', '-')
        df_summary['requests_grouping'] = pd.Series(None, index=df_summary.index, dtype=object)
        df_summary['requests_batching'] = pd.Series(None, index=df_summary.index, dtype=object)
        df_summary = df_summary.set_index(
            ['model', 'num_input_tokens', 'num_output_tokens', 'num_concurrent_requests', 'qps']
        )

        # Read individual responses
        df = read_perf_eval_json_files(output_files_dir, type='individual_responses')

        # Process individual files and add requests batching approximation
        for filename in os.listdir(output_files_dir):
            if 'individual_responses' in filename:
                model_finame: str
                in_tok_finame: int
                out_tok_finame: int
                concurrency_finame: Optional[int]
                qps_finame: Optional[float]

                model_finame, in_tok_finame, out_tok_finame, concurrency_finame, qps_finame = extract_file_info(
                    filename
                )
                df_file = df[df['filename'] == filename].copy()
                df_file = df_file[df_file['error_code'].isnull()]

                requests_grouping, requests_batching = get_grouping_and_batching_info(df_file)

                key = (model_finame, in_tok_finame, out_tok_finame, concurrency_finame, qps_finame)

                if key in df_summary.index:
                    df_summary.at[key, 'requests_grouping'] = requests_grouping
                    df_summary.at[key, 'requests_batching'] = requests_batching
                else:
                    raise KeyError(f'Key {key} not found in dictionary. File: {file}')

        consolidated_results_dir = os.path.expanduser(config['consolidated_results_dir'])
        if not os.path.exists(consolidated_results_dir):
            os.makedirs(consolidated_results_dir)
        df_summary.sort_values(
            by=['model', 'num_input_tokens', 'num_output_tokens', 'num_concurrent_requests', 'qps'], inplace=True
        )
        df_summary.to_excel(os.path.join(consolidated_results_dir, f'consolidated_results_{run_time}.xlsx'))

    except Exception as e:
        logging.error(f"Error while writing consolidated results to {config['consolidated_results_dir']}")
        logging.error(e)
