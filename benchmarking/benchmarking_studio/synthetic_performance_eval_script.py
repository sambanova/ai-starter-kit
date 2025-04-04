import os
import sys
import yaml
import time
from typing import Any, Dict
import pandas as pd

sys.path.append('../')
sys.path.append('../src')
sys.path.append('../prompts')
sys.path.append('../src/llmperf')

# Get the absolute path of my_project
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

sys.path.insert(0, project_root)

from benchmarking.src.performance_evaluation import SyntheticPerformanceEvaluator
from benchmarking.utils import read_synthetic_json_files

from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv(os.path.join(project_root,'.env'), override=True)

# Read config file
CONFIG_FILE_PATH = os.path.join(current_dir, 'config.yaml')
with open(CONFIG_FILE_PATH) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
config = config['synthetic']
config['output_files_dir'] = os.path.expanduser(config['output_files_dir'])
config['model_configs_path'] = os.path.expanduser(config['model_configs_path'])
model_configs_df = pd.read_csv(config['model_configs_path'])

# Additional parameters:
run_time = time.strftime("%Y%m%d-%H%M%S")
output_files_dir = os.path.join(config['output_files_dir'], run_time)
sampling_params: Dict[str, Any] = {}
user_metadata: Dict[str, Any] = {}

# Loop over models and configs to run performance evaluation
for model_name, input_tokens, output_tokens, concurrent_requests  in zip(model_configs_df['model_names'], model_configs_df['input_tokens'], model_configs_df['output_tokens'], model_configs_df['concurrent_requests']):
    num_requests = concurrent_requests * config['ratio']
    
    logging.info(
        f'Running model_name {model_name}, input_tokens {input_tokens}, output_tokens {output_tokens},'
        f'concurrent_requests {concurrent_requests}, num_requests {num_requests}'
    )
    
    try:
        # Instantiate evaluator
        user_metadata['model_idx'] = 0 # no need to change
        evaluator = SyntheticPerformanceEvaluator(
            model_name=model_name,
            results_dir=os.path.expanduser(output_files_dir),
            num_concurrent_requests=concurrent_requests,
            timeout=config['timeout'],
            user_metadata=user_metadata,
            llm_api=config['llm_api'],
        )

        # Run performance evaluation
        model_results_summary, model_results_per_request = evaluator.run_benchmark(
            num_input_tokens=input_tokens,
            num_output_tokens=output_tokens,
            num_requests=num_requests,
            sampling_params=sampling_params,
        )
    except Exception as e:
        logging.error(f"Error while running model_name {model_name}, input_tokens {input_tokens}, output_tokens {output_tokens}, concurrent_requests {concurrent_requests}, num_requests {num_requests}")
        logging.error(e)
        
    logging.info(f"Time delay: {config['time_delay']} seconds")
    time.sleep(config['time_delay'])

# Consolidate results
if config['consolidated_results_dir']:
    logging.info(f"Writting consolidated results to {config['consolidated_results_dir']}")
    try:
        df = read_synthetic_json_files(output_files_dir, type='summary')
        df = df[[
            'model', 'num_input_tokens', 'num_output_tokens', 'num_concurrent_requests',
            'server_ttft_s_min', 'server_ttft_s_p50', 'server_ttft_s_max',
            'server_end_to_end_latency_s_min', 'server_end_to_end_latency_s_p50', 'server_end_to_end_latency_s_max',
            'server_output_token_per_s_min', 'server_output_token_per_s_p50', 'server_output_token_per_s_max',
            'acceptance_rate_min', 'acceptance_rate_p50', 'acceptance_rate_max',
            'server_number_input_tokens_p50', 'server_number_output_tokens_p50',
            'client_ttft_s_min', 'client_ttft_s_p50', 'client_ttft_s_max',
            'client_end_to_end_latency_s_min', 'client_end_to_end_latency_s_p50', 'client_end_to_end_latency_s_max',
            'client_output_token_per_s_min', 'client_output_token_per_s_p50', 'client_output_token_per_s_max',
            'num_requests_started', 'num_completed_requests', 'number_errors', 'error_code_frequency'
        ]].sort_values(['model', 'num_input_tokens', 'num_output_tokens', 'num_concurrent_requests']).copy()

        consolidated_results_dir = os.path.expanduser(config['consolidated_results_dir'])
        consolidated_results_dir = os.path.join(consolidated_results_dir, run_time)

        os.makedirs(consolidated_results_dir, exist_ok=True)
        df.to_excel(os.path.join(consolidated_results_dir, 'consolidated_results.xlsx'), index=False)
    except Exception as e:
        logging.error(f"Error while writing consolidated results to {config['consolidated_results_dir']}")
        logging.error(e)