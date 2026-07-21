import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

# Default batch sizes used to infer batching (powers of two up to 128).
# Can be overridden via the `batch_sizes` key in config.yaml.
DEFAULT_BATCH_SIZES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128]


# =========================================================
#                   DATA CLASSES
# =========================================================


@dataclass
class ModelConfigRow:
    model_name: str
    input_tokens: int
    output_tokens: int
    num_requests: int
    num_warmup_requests: int = 0
    concurrent_requests: Optional[int] = None
    qps: Optional[float] = None
    qps_distribution: str = 'constant'
    multimodal_img_size: str = 'na'


# =========================================================
#                   CONFIG LOADER
# =========================================================


class ConfigLoader:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    def load(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        with open(self.config_path) as fh:
            cfg = yaml.load(fh, Loader=yaml.FullLoader)
        cfg['output_files_dir'] = os.path.expanduser(cfg.get('output_files_dir', '..'))
        cfg['model_configs_path'] = os.path.expanduser(cfg.get('model_configs_path', ''))
        cfg['batch_sizes'] = cfg.get('batch_sizes') or DEFAULT_BATCH_SIZES
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
    def __init__(self, batch_sizes: Optional[Sequence[int]] = None) -> None:
        sizes = batch_sizes if batch_sizes else DEFAULT_BATCH_SIZES
        self.batch_sizes = sorted({int(s) for s in sizes})

    def _snap_batch_size(self, count: int) -> int:
        """Snap an observed group count UP to the nearest allowed batch size.

        Requests processed in the same server batch share an identical
        server_ttft_s, so the count of consecutive identical-TTFT requests is
        the observed group size. That count is mapped to the smallest configured
        batch size that is >= it (falling back to the largest configured size if
        the count exceeds every allowed value).
        """
        for size in self.batch_sizes:
            if size >= count:
                return size
        return self.batch_sizes[-1]

    def get_grouping_and_batching_info(self, df: pd.DataFrame) -> Tuple[List[int], List[int], pd.DataFrame]:
        if df.empty:
            return [], [], df

        df = df.sort_values('end_time').reset_index(drop=True)
        df['group'] = (df['server_ttft_s'] != df['server_ttft_s'].shift()).cumsum()

        group_counts = df.groupby(['group', 'server_ttft_s']).size().reset_index(name='consecutive_count')
        requests_grouping = group_counts['consecutive_count'].tolist()
        requests_batching = [self._snap_batch_size(x) for x in requests_grouping]

        group_to_count = group_counts.set_index('group')['consecutive_count']
        group_to_batching = {g: self._snap_batch_size(cnt) for g, cnt in group_to_count.items()}

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
        if total_sum > 0:
            for value, count in counter.items():
                if (value * count) / total_sum > 0.5:
                    return value
        # No single batch size dominates (e.g. a failed request skewed the
        # distribution) - fall back to the most frequently observed batch size.
        return counter.most_common(1)[0][0]


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

            results.append({'uuid': uuid, 'switching_time': switching_time})
        return pd.DataFrame(results).set_index('uuid')


# =========================================================
#                   BUNDLE SUMMARY CALCULATOR
# =========================================================


class BundleSummaryCalculator:
    """Derive bundle-level (multi-row) throughput metrics from per-row summary data.

    All inputs come from columns already present in the per-row summary
    (`timestamp`, `num_completed_requests`, `num_completed_requests_per_min`,
    `number_errors`, `model`) - no changes to the core evaluator/runner are
    required or made. Per-row `start_time`/`end_time` are never persisted, so
    each row's wall-clock window is approximated as
    `[timestamp - duration_row_s, timestamp]`, where `duration_row_s` is back
    out of the row's already-computed RPM and completed-request count.

    Output rows expose both the numerator and denominator of `bundle_rps`/
    `bundle_rpm` so the calculation can be reproduced by hand:
        total_effective_duration_s = total_duration_s - delay_time_s
        bundle_rps = total_completed_requests / total_effective_duration_s
        bundle_rpm = bundle_rps * 60
    """

    def __init__(self, family_lookup_fn: Any) -> None:
        self.family_lookup_fn = family_lookup_fn

    @staticmethod
    def _row_duration_s(row: pd.Series) -> float:
        rpm = row.get('num_completed_requests_per_min', 0) or 0
        completed = row.get('num_completed_requests', 0) or 0
        if rpm <= 0 or completed <= 0:
            return 0.0
        return completed / (rpm / 60.0)

    def _annotate_spans(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['duration_row_s'] = df.apply(self._row_duration_s, axis=1)
        df['approx_end_wallclock'] = df['timestamp']
        df['approx_start_wallclock'] = df['timestamp'] - df['duration_row_s']
        return df

    def _summarize_group(
        self,
        df_group: pd.DataFrame,
        time_delay: float,
        concurrency_enabled: bool,
        label: str,
    ) -> Dict[str, Any]:
        # Rows with no usable duration (failed/errored runs) can't contribute
        # a meaningful start/end point, so they're excluded from the span
        # min/max, but their requests/errors still count toward the totals.
        valid = df_group[df_group['duration_row_s'] > 0]

        num_model_configs = len(df_group)
        if 'num_requests_started' in df_group.columns:
            total_num_requests_started = int(df_group['num_requests_started'].fillna(0).sum())
        else:
            total_num_requests_started = 0
        total_completed = int(df_group['num_completed_requests'].fillna(0).sum())
        total_errors = int(df_group['number_errors'].fillna(0).sum())

        if valid.empty:
            total_duration_s = 0.0
            delay_time_s = 0.0
            total_effective_duration_s = 0.0
            bundle_rps = 0.0
        else:
            start = valid['approx_start_wallclock'].min()
            end = valid['approx_end_wallclock'].max()
            total_duration_s = max(end - start, 0.0)

            if concurrency_enabled or num_model_configs <= 1:
                delay_time_s = 0.0
            else:
                delay_time_s = time_delay * max(num_model_configs - 1, 0)

            total_effective_duration_s = max(total_duration_s - delay_time_s, 0.0)
            bundle_rps = (total_completed / total_effective_duration_s) if total_effective_duration_s > 0 else 0.0

        bundle_rpm = bundle_rps * 60.0

        return {
            'family': label,
            'num_model_configs': num_model_configs,
            'total_num_requests_started': total_num_requests_started,
            'total_errors': total_errors,
            # numerator of bundle_rps/bundle_rpm
            'total_completed_requests': total_completed,
            # raw wall-clock window covering all rows in this group
            'total_duration_s': round(total_duration_s, 4),
            'delay_time_s': round(delay_time_s, 4),
            # denominator of bundle_rps/bundle_rpm: total_duration_s - delay_time_s
            'total_effective_duration_s': round(total_effective_duration_s, 4),
            # bundle_rps = total_completed_requests / total_effective_duration_s
            'bundle_rps': round(bundle_rps, 4),
            # bundle_rpm = bundle_rps * 60
            'bundle_rpm': round(bundle_rpm, 4),
            'concurrency_enabled': bool(concurrency_enabled),
        }

    def build_summary(
        self,
        df_summary: pd.DataFrame,
        time_delay: float,
        concurrency_enabled: bool,
    ) -> pd.DataFrame:
        df = self._annotate_spans(df_summary)

        rows = [self._summarize_group(df, time_delay, concurrency_enabled, label='ALL')]

        # NOTE: unrecognized model names fall back to the 'llama2' family
        # (see find_family_model_type) - such rows will surface under the
        # 'llama2' row rather than a distinct 'unknown' bucket. Per-family
        # spans are only a clean, non-overlapping decomposition when a
        # family's rows are contiguous in execution order; interleaved
        # families can produce overlapping family-level spans, which is
        # expected, not a bug.
        df['model_family'] = df['model'].apply(self.family_lookup_fn)
        for family, df_family in df.groupby('model_family'):
            rows.append(self._summarize_group(df_family, time_delay, concurrency_enabled, label=family))

        return pd.DataFrame(rows)


# =========================================================
#                   RESULTS CONSOLIDATOR
# =========================================================


class ResultsConsolidator:
    def __init__(
        self,
        read_perf_eval_json_files_fn: Any,
        file_parser: FileNameParser,
        batch_analyzer: BatchAnalyzer,
        rep_finder: RepresentativeFinder,
        bundle_summary_calculator: BundleSummaryCalculator,
    ) -> None:
        self.read_perf_eval_json_files = read_perf_eval_json_files_fn
        self.file_parser = file_parser
        self.batch_analyzer = batch_analyzer
        self.rep_finder = rep_finder
        self.bundle_summary_calculator = bundle_summary_calculator

    def consolidate(
        self,
        output_files_dir: str,
        consolidated_results_dir: str,
        run_name: str,
        time_delay: float = 0.0,
        concurrency_enabled: bool = False,
    ) -> None:
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
            logger.warning('No valid batching data found.')
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

        # Bundle-level summary is computed from the full df_summary (before
        # column pruning) so it stays independent of the per-model sheet's
        # column set.
        df_bundle = self.bundle_summary_calculator.build_summary(
            df_summary, time_delay=time_delay, concurrency_enabled=concurrency_enabled
        )

        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df_summary[selected_columns].to_excel(writer, sheet_name='per_model')
            df_bundle.to_excel(writer, sheet_name='bundle_summary', index=False)
        logger.info(f'✅ Wrote consolidated results with switching time and bundle summary to {out_path}')


# =========================================================
#                   BENCHMARK RUNNER
# =========================================================


class BenchmarkRunner:
    def __init__(
        self,
        config: Dict[str, Any],
        evaluator_factories: Dict[str, Any],
        read_perf_eval_json_files_fn: Any,
        file_parser: FileNameParser,
        batch_analyzer: BatchAnalyzer,
        rep_finder: RepresentativeFinder,
    ) -> None:
        self.config = config
        self.evaluator_factories = evaluator_factories
        self.read_perf_eval_json_files = read_perf_eval_json_files_fn
        self.file_parser = file_parser
        self.batch_analyzer = batch_analyzer
        self.rep_finder = rep_finder

    def _run_single_row(self, row: pd.Series, output_files_dir: str) -> None:
        from benchmarking.src.performance_evaluation import (
            RealWorkLoadPerformanceEvaluator,
            SyntheticPerformanceEvaluator,
        )

        model_name = row['model_name']
        num_requests = int(row['num_requests'])
        num_warmup_requests = int(row.get('num_warmup_requests', 0) or 0)
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
                    num_warmup_requests=num_warmup_requests,
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
                    num_warmup_requests=num_warmup_requests,
                )
            else:
                logger.warning(f'Skipping {model_name}: missing concurrency or QPS.')
                return

            evaluator.run_benchmark(
                num_input_tokens=input_tokens,
                num_output_tokens=output_tokens,
                num_requests=num_requests,
                sampling_params={},
            )

        except Exception as e:
            logger.exception(f'Error running evaluator for model {model_name}: {e}')

        time.sleep(self.config.get('time_delay', 0))

    def run(self, run_name: Optional[str] = None) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        model_configs_df = pd.read_csv(self.config['model_configs_path'])
        int_columns = {'input_tokens': 'Int64', 'output_tokens': 'Int64', 'num_requests': 'Int64'}
        # `num_warmup_requests` is an optional per-row column; only cast it when present so older
        # CSVs without the column still load.
        if 'num_warmup_requests' in model_configs_df.columns:
            int_columns['num_warmup_requests'] = 'Int64'
        model_configs_df = model_configs_df.astype(int_columns)

        run_time = datetime.now().strftime('%Y%m%d-%H%M%S.%f')
        if not run_name:
            run_name = run_time
        output_files_dir = os.path.join(self.config['output_files_dir'], run_name)

        if self.config['concurrency_enabled']:
            logger.info(f'🚀 Running benchmarks with row-level concurrency (max_workers={self.config["max_workers"]})')
            with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                futures = [
                    executor.submit(self._run_single_row, row, output_files_dir)
                    for _, row in model_configs_df.iterrows()
                ]

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.exception(f'Unhandled exception in concurrent run: {e}')
        else:
            logger.info('🐢 Running benchmarks sequentially')
            for _, row in model_configs_df.iterrows():
                self._run_single_row(row, output_files_dir)

        # Consolidation phase
        # For debugging, you can set a specific run_name here
        # run_name = '20260629-163957.994322'
        # output_files_dir = os.path.join(self.config['output_files_dir'], run_name)
        consolidator = ResultsConsolidator(
            self.read_perf_eval_json_files,
            self.file_parser,
            self.batch_analyzer,
            self.rep_finder,
            BundleSummaryCalculator(family_lookup_fn=find_family_model_type_wrapper),
        )
        consolidator.consolidate(
            output_files_dir,
            self.config['consolidated_results_dir'],
            run_name,
            time_delay=self.config.get('time_delay', 0),
            concurrency_enabled=self.config.get('concurrency_enabled', False),
        )


# =========================================================
#                   ENTRY POINT
# =========================================================


def read_perf_eval_json_files_wrapper(path: str, type: str) -> pd.DataFrame:
    from benchmarking.utils import read_perf_eval_json_files as _read_fn

    return _read_fn(path, type=type)


def find_family_model_type_wrapper(model_name: str) -> str:
    from benchmarking.benchmarking_utils import find_family_model_type as _find_family_fn

    return _find_family_fn(model_name)


# ---------------------------------------------------------
# Load per-request dataframe with batching + switching time
# ---------------------------------------------------------


def load_requests_with_switching(
    output_files_dir: str,
    read_perf_eval_json_files_fn: Any,
    file_parser: Optional[FileNameParser] = None,
    batch_analyzer: Optional[BatchAnalyzer] = None,
) -> pd.DataFrame:
    """
    Returns a per-request dataframe enriched with:
      - batching info
      - uuid
      - switching_time (same value repeated per request in a run)
    """

    file_parser = file_parser or FileNameParser()
    batch_analyzer = batch_analyzer or BatchAnalyzer()

    df_individual = read_perf_eval_json_files_fn(output_files_dir, type='individual_responses')

    dfs = []

    for filename in os.listdir(output_files_dir):
        if 'individual_responses' not in filename:
            continue

        try:
            df_file = df_individual[df_individual['filename'] == filename].copy()
            _, _, _, _, _ = file_parser.extract_file_info(filename)

            _, _, df_with_batching = batch_analyzer.get_grouping_and_batching_info(df_file)

            df_with_batching['uuid'] = filename
            dfs.append(df_with_batching)

        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    df_all['uuid'] = df_all['filename'].apply(file_parser.find_uuid)

    # Compute switching time per UUID
    df_switching = SwitchingTimeCalculator.calculate_switching_time(df_all)

    # Broadcast switching time to every request
    df_all = df_all.merge(df_switching, left_on='uuid', right_index=True, how='left')

    return df_all


def main() -> None:
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
        batch_analyzer=BatchAnalyzer(config.get('batch_sizes')),
        rep_finder=RepresentativeFinder(),
    )

    runner.run()


if __name__ == '__main__':
    main()
