# =========================================================
#                   refactored_benchmark_final.py
# =========================================================
import logging
import os
import re
import sys
import time
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import pandas as pd
import yaml
from dotenv import load_dotenv

# =========================================================
#                   LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =========================================================
#                   DATA CLASSES
# =========================================================

@dataclass
class ModelConfigRow:
    model_name: str
    input_tokens: int
    output_tokens: int
    num_requests: int
    concurrent_requests: Optional[int] = None
    qps: Optional[float] = None
    qps_distribution: str = 'constant'
    multimodal_img_size: str = 'na'


# =========================================================
#                   CONFIG LOADER
# =========================================================

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path

    def load(self) -> Dict[str, Any]:
        with open(self.config_path) as fh:
            cfg = yaml.load(fh, Loader=yaml.FullLoader)
        cfg['output_files_dir'] = os.path.expanduser(cfg.get('output_files_dir', '..'))
        cfg['model_configs_path'] = os.path.expanduser(cfg.get('model_configs_path', ''))
        return cfg


# =========================================================
#                   FILENAME PARSER
# =========================================================

class FileNameParser:
    UUID_RE = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')

    def find_uuid(self, file_name: str) -> str:
        match = self.UUID_RE.search(file_name)
        if not match:
            raise ValueError(f'UUID not found in filename {file_name}')
        return match.group(0)

    def extract_file_info(self, file_name: str) -> Tuple[str, int, int, Optional[int], Optional[float]]:
        parts = file_name.split('_')
        try:
            if 'multimodal' in file_name:
                model = parts[2]
                in_tok = int(parts[5])
                out_tok = int(parts[6])
                con_type = parts[7]
            else:
                model = parts[2]
                in_tok = int(parts[3])
                out_tok = int(parts[4])
                con_type = parts[5]
        except Exception:
            raise ValueError(f'Unexpected filename format: {file_name}')

        if 'synthetic' in file_name:
            con = int(con_type)
            qps = None
        elif 'realworkload' in file_name:
            con = None
            qps = float(con_type.replace('-', '.'))
        else:
            con, qps = None, None

        return model, in_tok, out_tok, con, qps


# =========================================================
#                   BATCH ANALYZER
# =========================================================

class BatchAnalyzer:
    @staticmethod
    def get_grouping_and_batching_info(df: pd.DataFrame) -> Tuple[List[int], List[int], pd.DataFrame]:
        if df.empty:
            return [], [], df

        df = df.sort_values('end_time').reset_index(drop=True)
        df['group'] = (df['server_ttft_s'] != df['server_ttft_s'].shift()).cumsum()

        group_counts = df.groupby(['group', 'server_ttft_s']).size().reset_index(name='consecutive_count')
        requests_grouping = group_counts['consecutive_count'].tolist()
        requests_batching = [1 << (x - 1).bit_length() for x in requests_grouping]

        group_to_count = group_counts.set_index('group')['consecutive_count']
        group_to_batching = {g: 1 << (cnt - 1).bit_length() for g, cnt in group_to_count.items()}

        df['requests_grouping_per_request'] = df['group'].map(group_to_count)
        df['requests_batching_per_request'] = df['group'].map(group_to_batching)

        return requests_grouping, requests_batching, df.drop(columns=['group'])


# =========================================================
#                   REPRESENTATIVE FINDER
# =========================================================

class RepresentativeFinder:
    @staticmethod
    def find_median_in_batches(lst: Sequence[int]) -> Optional[int]:
        if not lst:
            return None
        total_sum = sum(lst)
        counter = Counter(lst)
        for value, count in counter.items():
            if (value * count) / total_sum > 0.5:
                return value
        return None


# =========================================================
#                   SWITCHING TIME CALCULATOR
# =========================================================

class SwitchingTimeCalculator:
    """Compute switching time per UUID run."""

    @staticmethod
    def calculate_switching_time(df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for uuid, group in df.groupby('uuid'):
            group = group.sort_values('start_time')
            max_batching = group['requests_batching_per_request'].max()
            max_batch_rows = group[group['requests_batching_per_request'] == max_batching]

            highest_ttft = max_batch_rows['server_ttft_s'].max()
            lowest_ttft = max_batch_rows['server_ttft_s'].min()
            switching_time = highest_ttft - lowest_ttft

            results.append({
                'uuid': uuid,
                'switching_time': switching_time
            })
        return pd.DataFrame(results).set_index('uuid')


# =========================================================
#                   RESULTS CONSOLIDATOR
# =========================================================

class ResultsConsolidator:
    def __init__(self, read_perf_eval_json_files_fn, file_parser, batch_analyzer, rep_finder):
        self.read_perf_eval_json_files = read_perf_eval_json_files_fn
        self.file_parser = file_parser
        self.batch_analyzer = batch_analyzer
        self.rep_finder = rep_finder

    def consolidate(self, output_files_dir: str, consolidated_results_dir: str, run_name: str) -> None:
        out_dir = os.path.expanduser(output_files_dir)
        consolidated_dir = os.path.expanduser(consolidated_results_dir)

        df_summary = self.read_perf_eval_json_files(out_dir, type='summary')
        df_individual = self.read_perf_eval_json_files(out_dir, type='individual_responses')

        df_summary['uuid'] = df_summary['name'].apply(self.file_parser.find_uuid)

        # Add batch + switching time data
        dfs_with_batching = []
        for filename in os.listdir(out_dir):
            if 'individual_responses' not in filename:
                continue

            try:
                df_file = df_individual[df_individual['filename'] == filename].copy()
                _, _, _, _, _ = self.file_parser.extract_file_info(filename)
                grouping, batching, df_with_batching = self.batch_analyzer.get_grouping_and_batching_info(df_file)
                dfs_with_batching.append(df_with_batching)
            except Exception as e:
                logger.warning(f'Error processing {filename}: {e}')
                continue

        if not dfs_with_batching:
            logger.warning("No valid batching data found.")
            return

        df_all = pd.concat(dfs_with_batching)
        
        df_all['uuid'] = df_all['filename'].apply(self.file_parser.find_uuid)
        
        df_switching = SwitchingTimeCalculator.calculate_switching_time(df_all)

        # Merge switching time into summary
        df_summary = df_summary.merge(df_switching, on='uuid', how='left')
        df_summary['representative_batch_size'] = df_summary['uuid'].map(
            lambda u: self.rep_finder.find_median_in_batches(
                df_all[df_all['uuid'] == u]['requests_batching_per_request'].tolist()
            )
        )

        # get batching frequencies
        def get_batching_frequencies(uuid: str) -> Dict[int, int]:
            df_uuid = df_all[df_all['uuid'] == uuid]
            freq = dict(Counter(df_uuid['requests_batching_per_request']))
            return freq
        df_summary['request_batching_frequencies'] = df_summary['uuid'].map(get_batching_frequencies)
        
        os.makedirs(consolidated_dir, exist_ok=True)
        out_path = os.path.join(consolidated_dir, f'{run_name}.xlsx')
        df_summary.sort_values('timestamp', inplace=True)     

       # --- Dynamically determine which columns to include before export ---
        missing_columns = []

        if 'num_concurrent_requests' not in df_summary.columns:
            missing_columns.append('num_concurrent_requests')

        if 'qps' not in df_summary.columns:
            missing_columns.append('qps')

        if 'qps_distribution' not in df_summary.columns:
            missing_columns.append('qps_distribution')

        df_summary['multimodal_img_size'] = df_summary['name'].str.extract(
            r'multimodal_(small|medium|large)', expand=False
        )

        if df_summary['multimodal_img_size'].isnull().all():
            missing_columns.append('multimodal_img_size')

        # Set fields to report
        selected_columns = [
            'name',
            'model',
            'num_input_tokens',
            'num_output_tokens',
            'num_concurrent_requests',
            'qps',
            'qps_distribution',
            'multimodal_img_size',
            'server_ttft_s_min',
            'server_ttft_s_mean',
            'server_ttft_s_p50',
            'server_ttft_s_max',
            'server_end_to_end_latency_s_min',
            'server_end_to_end_latency_s_mean',
            'server_end_to_end_latency_s_p50',
            'server_end_to_end_latency_s_max',                
            'server_output_token_per_s_min',
            'server_output_token_per_s_mean', 
            'server_output_token_per_s_p50',
            'server_output_token_per_s_max',                               
            'acceptance_rate_min',
            'acceptance_rate_p50',
            'acceptance_rate_max',
            'server_number_input_tokens_p50',
            'server_number_output_tokens_p50',
            'client_ttft_s_min',
            'client_ttft_s_mean',
            'client_ttft_s_p50',
            'client_ttft_s_max',
            'client_end_to_end_latency_s_min',
            'client_end_to_end_latency_s_mean',
            'client_end_to_end_latency_s_p50',
            'client_end_to_end_latency_s_max',                
            'client_output_token_per_s_min',
            'client_output_token_per_s_mean',                
            'client_output_token_per_s_p50',
            'client_output_token_per_s_max',                
            'client_total_output_throughput',
            'num_requests_started',
            'num_completed_requests',
            'num_completed_requests_per_min',
            'number_errors',
            'error_code_frequency',
            'switching_time',
            'request_batching_frequencies',
            'representative_batch_size',
        ]

        # Remove missing columns safely
        selected_columns = [c for c in selected_columns if c not in missing_columns and c in df_summary.columns]
        # Keep only selected columns for export
        df_summary[selected_columns].to_excel(out_path)
        logger.info(f'✅ Wrote consolidated results with switching time to {out_path}')


# =========================================================
#                   BENCHMARK RUNNER
# =========================================================

class BenchmarkRunner:
    def __init__(self, config, evaluator_factories, read_perf_eval_json_files_fn, file_parser, batch_analyzer, rep_finder):
        self.config = config
        self.evaluator_factories = evaluator_factories
        self.read_perf_eval_json_files = read_perf_eval_json_files_fn
        self.file_parser = file_parser
        self.batch_analyzer = batch_analyzer
        self.rep_finder = rep_finder

    def run(self, run_name: Optional[str] = None) -> None:
        from benchmarking.src.performance_evaluation import RealWorkLoadPerformanceEvaluator, SyntheticPerformanceEvaluator

        model_configs_df = pd.read_csv(self.config['model_configs_path'])
        model_configs_df = model_configs_df.astype({'input_tokens': 'Int64', 'output_tokens': 'Int64', 'num_requests': 'Int64'})

        run_time = datetime.now().strftime('%Y%m%d-%H%M%S.%f')
        if not run_name:
            run_name = run_time
        output_files_dir = os.path.join(self.config['output_files_dir'], run_name)

        for _, row in model_configs_df.iterrows():
            model_name = row['model_name']
            num_requests = int(row['num_requests'])
            input_tokens = int(row['input_tokens'])
            output_tokens = int(row['output_tokens'])
            concurrent_requests = int(row.get('concurrent_requests', 0) or 0)
            qps = float(row.get('qps', 0.0) or 0.0)
            multimodal_img_size = row.get('multimodal_img_size') if pd.notna(row.get('multimodal_img_size')) else 'na'
            
            evaluator = None
            try:
                if concurrent_requests:
                    evaluator = SyntheticPerformanceEvaluator(
                        multimodal_image_size=multimodal_img_size,
                        model_name=model_name,
                        results_dir=os.path.expanduser(output_files_dir),
                        num_concurrent_requests=concurrent_requests,
                        timeout=self.config['timeout'],
                        user_metadata={'model_idx': 0},
                        llm_api=self.config['llm_api'],
                        use_multiple_prompts=self.config['use_multiple_prompts'],
                    )
                elif qps:
                    evaluator = RealWorkLoadPerformanceEvaluator(
                        multimodal_image_size=multimodal_img_size,
                        model_name=model_name,
                        results_dir=os.path.expanduser(output_files_dir),
                        qps=qps,
                        qps_distribution=row.get('qps_distribution', 'constant'),
                        timeout=self.config['timeout'],
                        user_metadata={'model_idx': 0},
                        llm_api=self.config['llm_api'],
                    )
                else:
                    logger.warning(f'Skipping {model_name}: missing concurrency or QPS.')
                    continue

                evaluator.run_benchmark(
                    num_input_tokens=input_tokens,
                    num_output_tokens=output_tokens,
                    num_requests=num_requests,
                    sampling_params={},
                )

            except Exception as e:
                logger.exception(f"Error running evaluator for model {model_name}: {e}")

            time.sleep(self.config.get('time_delay', 0))

        # Consolidation phase
        # run_name = '20251031-180601.530151'
        # output_files_dir = os.path.join(self.config['output_files_dir'], run_name)
        consolidator = ResultsConsolidator(
            self.read_perf_eval_json_files,
            self.file_parser,
            self.batch_analyzer,
            self.rep_finder,
        )
        consolidator.consolidate(output_files_dir, self.config['consolidated_results_dir'], run_name)


# =========================================================
#                   ENTRY POINT
# =========================================================

def read_perf_eval_json_files_wrapper(path: str, type: str) -> pd.DataFrame:
    from benchmarking.utils import read_perf_eval_json_files as _read_fn
    return _read_fn(path, type=type)


def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../'))
    sys.path.insert(0, project_root)

    load_dotenv(os.path.join(project_root, '.env'), override=True)

    config_path = os.path.join(current_dir, 'config.yaml')
    config = ConfigLoader(config_path).load()

    runner = BenchmarkRunner(
        config=config,
        evaluator_factories={},
        read_perf_eval_json_files_fn=read_perf_eval_json_files_wrapper,
        file_parser=FileNameParser(),
        batch_analyzer=BatchAnalyzer(),
        rep_finder=RepresentativeFinder(),
    )

    runner.run()


if __name__ == '__main__':
    main()
