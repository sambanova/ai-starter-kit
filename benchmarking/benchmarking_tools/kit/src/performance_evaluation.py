import abc
import base64
import json
import os
import random
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

file_location = Path(__file__).parent.resolve()
kit_location = os.path.join(file_location, '../../../')  # benchmarking/ dir (kit/src/ lives 3 levels down)

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import transformers
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import add_script_run_ctx
from tqdm import tqdm

import benchmarking.benchmarking_tools.kit.src.llmperf.llmperf_utils as llmperf_utils
from benchmarking.benchmarking_utils import get_tokenizer
from benchmarking.benchmarking_tools.kit.src.llmperf import common_metrics
from benchmarking.benchmarking_tools.kit.src.llmperf.llmperf_utils import flatten
from benchmarking.benchmarking_tools.kit.src.llmperf.models import LLMResponse, RequestConfig
from benchmarking.benchmarking_tools.kit.src.llmperf.sambanova_client import llm_request
from benchmarking.benchmarking_tools.kit.src.executor_base import BaseExecutorMixin
from benchmarking.benchmarking_tools.kit.src.schemas import BenchmarkSummary, QuantileStats, RequestMetric
from benchmarking.utils import CONFIG_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
# disable annoying streamlit logging
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').disabled = True
transformers.logging.set_verbosity_error()  # type: ignore[no-untyped-call]
# File-relative (not cwd-relative): this module is imported from several different working
# directories (kit/'s own quickstart.sh, evaluator.py invoked from benchmarking/, Streamlit, the
# bundles runner), so a cwd-relative '../.env' would only resolve correctly for one of them.
load_dotenv(os.path.join(file_location, '../../../../.env'), override=True)

SYSTEM_PROMPT_PATH = os.path.join(file_location, '../../../prompts/system-prompt_template.yaml')
USER_PROMPT_TEXT_INSTRUCT_PATH = os.path.join(
    file_location, '../../../prompts/user-prompt_template-text_instruct.yaml'
)
USER_PROMPT_VISION_INSTRUCT_PATH = os.path.join(
    file_location, '../../../prompts/user-prompt_template-vision_instruct.yaml'
)


class BasePerformanceEvaluator(BaseExecutorMixin, abc.ABC):
    def __init__(
        self,
        model_name: str,
        results_dir: str,
        multimodal_image_size: str = 'na',
        user_metadata: Dict[str, Any] = {},
        llm_api: str = 'sncloud',
        use_debugging_mode: bool = False,
        api_variables: Dict[str, str] = {},
        is_stream_mode: bool = True,
        timeout: int = 600,
        num_warmup_requests: int = 0,
        config: Dict[str, Any] = {},
    ) -> None:
        # Set kit's config file
        if not config:
            with open(CONFIG_PATH, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = config
        self.show_results_in_terminal = self.config['show_results_in_terminal']
        self.multimodal_image_size = multimodal_image_size
        self.model_name = model_name
        self.results_dir = results_dir
        self.user_metadata = user_metadata
        self.num_concurrent_requests: Optional[int] = None
        self.llm_api = llm_api
        self.use_debugging_mode = use_debugging_mode
        self.api_variables = api_variables
        self.is_stream_mode = is_stream_mode
        self.timeout = timeout
        self.num_warmup_requests = num_warmup_requests
        # Absolute monotonic deadline shared across warm-up + measured run. Set at the start of
        # each run so a timeout stops everything, in whichever phase it is reached. None => no deadline.
        self.deadline: Optional[float] = None
        self.tokenizer = get_tokenizer(self.model_name)
        self.stop_event = threading.Event()
        self.ui_progress_bar = None
        self.cli_progress_bar = None
        # Label shown by the UI progress bar; switched to 'Warming up' during the warm-up phase.
        self.progress_phase_label = 'Running requests'
        self.run_uuid = uuid.uuid4()
        # Overridden by each concrete preset (Custom/Synthetic/RealWorkLoad) to the workload
        # shape it implements; used by save_results to tag the canonical BenchmarkSummary.
        self.workload_mode = 'unknown'

        # To be set upon saving of results
        self.summary_file_path: Optional[str] = None
        self.individual_responses_file_path: Optional[str] = None

    def get_token_length(self, input_text: str) -> int:
        return len(self.tokenizer.encode(input_text))

    @staticmethod
    def sanitize_file_prefix(prefix: str) -> str:
        """Utility for sanitizing the output file prefix.

        Args:
            prefix (str): Output file prefix

        Returns:
            Sanitized outfile prefix
        """
        outfile_prefix = re.sub(r'[^\w\d-]+', '-', prefix)
        outfile_prefix = re.sub(r'-{2,}', '-', outfile_prefix)
        return outfile_prefix

    @abc.abstractmethod
    def create_output_filename(self, *args: Any, **kwargs: Any) -> str:
        pass

    @abc.abstractmethod
    def run_benchmark(
        self, sampling_params: Dict[str, Any] = {}, *args: Any, **kwargs: Any
    ) -> Tuple[Dict[str, Any], List[LLMResponse]]:
        pass

    @abc.abstractmethod
    def get_token_throughput_latencies(
        self, *args: Any, **kwargs: Any
    ) -> (
        Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], str, RequestConfig]]]
        | Tuple[dict[str, object], List[LLMResponse]]
    ):
        pass

    @abc.abstractmethod
    def build_request_configs(self, *args: Any, **kwargs: Any) -> List[RequestConfig]:
        pass

    @abc.abstractmethod
    def build_prompt(self, *args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], int]:
        pass

    def adjust_to_exact_tokens(self, text: str, target_token_count: int) -> str:
        """Modifies original text to desired number of output tokens based on corresponding tokenizer.
        For smaller outputs, process trims original text.
        For larger outputs, process pads original text with multiple pad tokens.

        Args:
            text (str): text to adjust
            target_token_count (int): number of desired tokens

        Returns:
            str: adjusted text
        """
        # if not Path(tokenized_text_filepath).exists():
        tokens = self.tokenizer.tokenize(text)
        token_count = len(tokens)

        if token_count > target_token_count:
            # Trim the text
            tokens = tokens[: target_token_count - 1]
        elif token_count < target_token_count:
            # Pad the text
            pad_token = self.tokenizer.pad_token if self.tokenizer.pad_token else '<pad>'
            tokens += [pad_token] * (target_token_count - token_count - 1)

        # Convert tokens back to text
        adjusted_text = str(self.tokenizer.convert_tokens_to_string(tokens))

        # Validate token count
        assert len(self.tokenizer.tokenize(adjusted_text)) == (target_token_count - 1), 'Token count mismatch!'

        return adjusted_text

    def send_requests(
        self,
        request_config_batch: List[Any],
        completed_requests: List[Any],
        progress: List[Any],
        start_time: float,
        num_requests: int,
    ) -> None:
        """Sends multiple requests to LLM and collects results

        Args:
            request_config_batch (list): list of request configs for LLM calls
            completed_requests (list): list of completed outputs from requests
            progress (int): progress value
            start_time (float): start time of the process
            num_requests (int): number of total requests
        """
        # The timeout is a single budget shared across warm-up and the measured run: once the
        # shared deadline passes, requests stop regardless of which phase is running. Fall back to
        # the legacy per-call bound if no deadline was set.
        deadline = self.deadline if self.deadline is not None else (start_time + self.timeout)
        for request_config in request_config_batch:
            if self.stop_event.is_set():
                logger.info('Stopping request processing in thread due to stop signal.')
                break
            if time.monotonic() >= deadline:
                break
            req_metrics, response_text, request_config = llm_request(request_config, self.tokenizer)

            # Create response object containing metrics, generated text, and corresponding request config
            response_object = LLMResponse(
                metrics=req_metrics,
                response_text=response_text,
                request_config=request_config,
            )
            completed_requests.extend([response_object])
            update_unit = 1
            progress.append(update_unit)

            if self.cli_progress_bar:
                self.cli_progress_bar.update(update_unit)
            if self.ui_progress_bar:
                self.ui_progress_bar(len(progress), num_requests, self.progress_phase_label)

    def build_warmup_configs(self, request_configs: List[RequestConfig]) -> List[RequestConfig]:
        """Selects `num_warmup_requests` configs to use for the warm-up phase.

        Warm-up may request more requests than the measured run builds (e.g. warm up 30 requests for a
        10-request test), so the available configs are cycled to reach the requested warm-up count. Each
        config is deep-copied before being sent in `run_warmup`, so reusing the same objects is safe.

        Args:
            request_configs (List[RequestConfig]): The configs built for the measured run.

        Returns:
            List[RequestConfig]: Exactly `num_warmup_requests` configs (empty if warm-up is disabled).
        """
        if not self.num_warmup_requests or not request_configs:
            return []
        return [request_configs[i % len(request_configs)] for i in range(self.num_warmup_requests)]

    def run_warmup(self, warmup_request_configs: List[RequestConfig]) -> None:
        """Sends throwaway requests to warm the server before the measured run.

        Warm-up is executed BEFORE the measurement clock (`start_time`) is captured and its
        responses are discarded, so it never enters the metrics summary. Its purpose is to absorb
        one-time costs that would otherwise skew the reported latencies/throughput and the inferred
        batch size: server-side cold start (weight/KV-cache allocation, graph compilation, autoscaler
        spin-up), connection/TLS setup, and the server ramping up to the target batch size.

        Requests are run at the test's concurrency (`num_concurrent_requests`) so the server reaches
        the same batching regime as the measured phase. When concurrency is unset (e.g. real workload),
        the warm-up requests are simply fanned out together.

        Args:
            warmup_request_configs (List[RequestConfig]): Request configs to send as warm-up.
        """
        if not warmup_request_configs:
            return

        # Deep-copy the configs: sending a request mutates its sampling_params in place
        # (e.g. `max_tokens_to_generate` is popped in the client), so warming up on the
        # original objects would corrupt the configs the measured run reuses.
        warmup_request_configs = [rc.model_copy(deep=True) for rc in warmup_request_configs]

        max_workers = self.num_concurrent_requests or len(warmup_request_configs)
        logger.info(
            f'Warming up with {len(warmup_request_configs)} request(s) at concurrency {max_workers} '
            '(results discarded)...'
        )

        # Throwaway sinks - these are intentionally never returned or summarized
        throwaway_responses: List[LLMResponse] = []
        throwaway_progress: List[Any] = []

        # Show warm-up progress on its own indicators so the user knows the wait is warm-up,
        # not a stall. The CLI gets a dedicated tqdm bar; the Streamlit (UI) callback is kept
        # live but re-labelled 'Warming up' and re-scaled to the warm-up request count. All of
        # these are restored to their measured-run state afterwards.
        saved_cli_progress_bar = self.cli_progress_bar
        saved_ui_progress_bar = self.ui_progress_bar
        saved_progress_phase_label = self.progress_phase_label
        self.cli_progress_bar = tqdm(total=len(warmup_request_configs), desc='Warming Up')
        self.progress_phase_label = 'Warming up'

        # Show the warm-up label immediately so the (often slow) first cold-start request doesn't
        # leave the UI looking stalled before the first completion updates the bar.
        if self.ui_progress_bar:
            self.ui_progress_bar(0, len(warmup_request_configs), self.progress_phase_label)

        warmup_start = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for request_config in warmup_request_configs:
                    if self.stop_event.is_set():
                        logger.info('Stopping warm-up due to stop signal.')
                        break
                    if self.deadline is not None and time.monotonic() >= self.deadline:
                        logger.warning('Timeout reached during warm-up; stopping warm-up early.')
                        break
                    future = executor.submit(
                        self.send_requests,
                        [request_config],
                        throwaway_responses,
                        throwaway_progress,
                        warmup_start,
                        len(warmup_request_configs),
                    )
                    futures.append(future)
                    for t in executor._threads:
                        add_script_run_ctx(t)

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        # A failed warm-up request must not abort the benchmark
                        logger.warning(f'Warm-up request failed (ignored): {e}')
        finally:
            if self.cli_progress_bar is not None:
                self.cli_progress_bar.close()
            self.cli_progress_bar = saved_cli_progress_bar
            self.ui_progress_bar = saved_ui_progress_bar
            self.progress_phase_label = saved_progress_phase_label

        completed = len(throwaway_responses)
        logger.info(f'Warm-up complete ({completed}/{len(warmup_request_configs)} sent).')

    def build_metrics_summary(
        self,
        metrics: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
    ) -> Dict[str, Any]:
        """Builds a summary of metrics from a list of dictionaries.

        This function takes a list of dictionaries, each representing a metric, and a start and end time.
        It filters out any metrics that resulted in an error, calculates descriptive statistics for a
        number of metrics, and records various other metrics such as the number of requests started,
        the error rate and count, the overall throughput, and the number of completed requests.

        Parameters:
        metrics (List[Dict[str, Any]]): A list of dictionaries, each representing a metric.
        start_time (time): The start time of the metrics collection.
        end_time (time): The end time of the metrics collection.

        Returns:
        Dict[str, Any]: A dictionary containing the summary metrics.
        """
        # Create empty metrics summary to be filled and returned
        metrics_summary: Dict[str, Any] = {}

        # Create base df from metrics returned from request responses
        raw_df = pd.DataFrame(metrics)

        # Remove errored requests
        metrics_df = raw_df[raw_df[common_metrics.ERROR_CODE].isna()]

        # Record descriptive statistics for the metrics in the following list
        for metric in [
            common_metrics.TTFT,
            common_metrics.E2E_LAT,
            common_metrics.REQ_OUTPUT_THROUGHPUT,
            common_metrics.NUM_INPUT_TOKENS,
            common_metrics.NUM_OUTPUT_TOKENS,
            common_metrics.MEAN_INTER_TOKEN_LATENCY,
            common_metrics.NETWORK_LATENCY_TTFT,
            common_metrics.NETWORK_LATENCY_E2E,
        ]:
            if self.show_results_in_terminal:
                logger.info(f'Building Client Metrics Summary for metric: {metric}')
            metrics_summary[metric] = {}

            # Skip metric if column is absent (for backward compatibility with old result files)
            if metric not in metrics_df.columns:
                if self.show_results_in_terminal:
                    logger.info(f'    Column {metric} not present, skipping')
                continue

            # Get flattened list from metric column in metrics df
            series = pd.Series(list(flatten(metrics_df[metric]))).dropna()

            # Skip metric if no valid data (for backward compatibility with old result files)
            if len(series) == 0:
                if self.show_results_in_terminal:
                    logger.info(f'    No valid data for {metric}, skipping')
                continue

            # Generate statistics for specific metric
            raw_quantiles = series.quantile([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            quantiles_reformatted_keys = {}
            for quantile, value in raw_quantiles.items():
                reformatted_key = f'p{int(quantile * 100)}'
                if self.show_results_in_terminal:
                    logger.info(f'    {reformatted_key} = {value:.6g}')
                quantiles_reformatted_keys[reformatted_key] = round(value, 4)
            metrics_summary[metric]['quantiles'] = quantiles_reformatted_keys
            series_mean = round(series.mean(), 4)
            metrics_summary[metric]['mean'] = series_mean
            series_min = round(series.min(), 4)
            metrics_summary[metric]['min'] = series_min
            series_max = round(series.max(), 4)
            metrics_summary[metric]['max'] = series_max
            series_std = round(series.std(), 4)
            metrics_summary[metric]['stddev'] = series_std

            if self.show_results_in_terminal:
                logger.info(f'    mean = {series.mean():.6g}')
                logger.info(f'    min = {series.min():.6g}')
                logger.info(f'    max = {series.max():.6g}')
                logger.info(f'    stddev = {series.std():.6g}')

        # Record descriptive statistics for the metrics in the following list
        for metric in [
            common_metrics.TTFT_SERVER,
            common_metrics.E2E_LAT_SERVER,
            common_metrics.REQ_OUTPUT_THROUGHPUT_SERVER,
            common_metrics.REQ_OUTPUT_THROUGHPUT_SERVER_FIRST_TEN,
            common_metrics.REQ_OUTPUT_THROUGHPUT_SERVER_FIRST_TEN,
            common_metrics.NUM_INPUT_TOKENS_SERVER,
            common_metrics.NUM_OUTPUT_TOKENS_SERVER,
            common_metrics.NUM_REASONING_TOKENS_SERVER,
            common_metrics.NUM_CACHED_TOKENS_SERVER,
            common_metrics.ACCEPTANCE_RATE,
        ]:
            if self.show_results_in_terminal:
                logger.info(f'Building Server Metrics Summary for metric: {metric}')
            metrics_summary[metric] = {}

            # Skip metric if column is absent (for backward compatibility with old result files)
            if metric not in metrics_df.columns:
                if self.show_results_in_terminal:
                    logger.info(f'    Column {metric} not present, skipping')
                continue

            # Get flattened list from metric column in metrics df
            series = pd.Series(list(flatten(metrics_df[metric]))).dropna()

            # Skip metric if no valid data (for backward compatibility with old result files)
            if len(series) == 0:
                if self.show_results_in_terminal:
                    logger.info(f'    No valid data for {metric}, skipping')
                continue

            # Generate statistics for specific metric
            raw_quantiles = series.quantile([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            quantiles_reformatted_keys = {}
            for quantile, value in raw_quantiles.items():
                reformatted_key = f'p{int(quantile * 100)}'
                if self.show_results_in_terminal:
                    logger.info(f'    {reformatted_key} = {value:.6g}')
                quantiles_reformatted_keys[reformatted_key] = round(value, 4)
            metrics_summary[metric]['quantiles'] = quantiles_reformatted_keys
            series_mean = round(series.mean(), 4)
            metrics_summary[metric]['mean'] = series_mean
            series_min = round(series.min(), 4)
            metrics_summary[metric]['min'] = series_min
            series_max = round(series.max(), 4)
            metrics_summary[metric]['max'] = series_max
            series_std = round(series.std(), 4)
            metrics_summary[metric]['stddev'] = series_std

            if self.show_results_in_terminal:
                logger.info(f'    mean = {series.mean():.6g}')
                logger.info(f'    min = {series.min():.6g}')
                logger.info(f'    max = {series.max():.6g}')
                logger.info(f'    stddev = {series.std():.6g}')

        # Record number of requests started
        metrics_summary[common_metrics.NUM_REQ_STARTED] = len(metrics)

        # Record error count and rate
        error_codes = raw_df[common_metrics.ERROR_CODE].dropna()
        num_errors = len(error_codes)
        metrics_summary[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
        metrics_summary[common_metrics.NUM_ERRORS] = num_errors
        if self.show_results_in_terminal:
            logger.info(f'Number Of Errored Requests: {num_errors}')

        # Record specific error code frequencies
        error_code_frequency = dict(error_codes.value_counts())
        if num_errors:
            if self.show_results_in_terminal:
                logger.error('Error Code Frequency')
                logger.error(error_code_frequency)
        metrics_summary[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)

        # Record overall throughput
        overall_output_throughput = round(
            metrics_df[common_metrics.NUM_OUTPUT_TOKENS].sum() / (end_time - start_time),
            4,
        )
        metrics_summary[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

        # Calculate mean output throughput (average of per-request throughputs)
        mean_output_throughput = round(
            metrics_df[common_metrics.REQ_OUTPUT_THROUGHPUT].mean(),
            4,
        )
        metrics_summary[common_metrics.MEAN_OUTPUT_THROUGHPUT] = mean_output_throughput

        # Record number of requests completed
        num_completed_requests = len(metrics_df)
        num_completed_requests_per_min = round(num_completed_requests / (end_time - start_time) * 60, 4)

        if self.show_results_in_terminal:
            logger.info(f'Overall Output Throughput: {overall_output_throughput}')
            logger.info(f'Number Of Completed Requests: {num_completed_requests}')
        if self.num_concurrent_requests:
            if self.show_results_in_terminal:
                logger.info(f'Number Of Concurrent Requests: {self.num_concurrent_requests}')
        if self.show_results_in_terminal:
            logger.info(f'Completed Requests Per Minute: {num_completed_requests_per_min}')

        metrics_summary[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
        metrics_summary[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

        return metrics_summary

    def save_results(
        self,
        filename: str,
        summary: Dict[str, Any],
        individual_responses: (
            List[LLMResponse]
            | List[Tuple[Dict[str, Any], str, RequestConfig]]
            | Tuple[Dict[str, object], List[LLMResponse]]
        ),
    ) -> None:
        """Save the performance evaluation results to a file.

        Constructs a canonical `BenchmarkSummary` (benchmarking.benchmarking_tools.kit.src.schemas) from
        `summary` before serializing, replacing the ad hoc `LLMPerfResults`/`flatten_dict` path.
        `BenchmarkSummary.to_legacy_flat_dict()` emits the exact same flattened key shape
        `LLMPerfResults.to_dict()` did, plus new additive fields (`tool`, `workload_mode`,
        `run_uuid`) — existing readers (`benchmarking/utils.py::read_perf_eval_json_files`,
        the bundles module's `ResultsConsolidator`, notebooks, Streamlit pages) keep working
        unmodified. This is deliberately the last piece of the Kit-native evaluator migrated to
        the shared schema (see the multi-tool benchmarking architecture plan) — vLLM and aiperf
        landed first, so any shape gap in the schema had already surfaced against two
        independent implementations before this, the highest-blast-radius call site, changed.

        Args:
            filename (str): The base name of the file to save the results to.
            summary (dict): A dictionary containing the summary of the performance evaluation.
            individual_responses (list): A list of individual responses from the performance evaluation.
            save_response_texts (bool): Whether to save the llm output text to an output file.

        Returns:
            None

        Raises:
            ValueError: If the results directory does not exist or is not a directory.
        """
        summary_filename = f'{filename}_summary'
        individual_responses_filename = f'{filename}_individual_responses'

        # Update to metadata. (Side-effecting on the caller's `summary` dict, matching the
        # pre-existing behavior — no current caller depends on the mutation, but preserved for
        # zero behavior change.)
        summary.update(self.user_metadata)

        benchmark_summary = self._build_benchmark_summary(summary_filename, summary)
        results_dir = Path(self.results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f'{results_dir} is not a directory')

        # Save summary results
        try:
            self.summary_file_path = f'{results_dir}/{summary_filename}.json'
            with open(self.summary_file_path, 'w') as f:
                json.dump(benchmark_summary.to_legacy_flat_dict(), f, indent=4, default=str)
        except Exception as e:
            logger.error(benchmark_summary.to_legacy_flat_dict())
            raise e

        # Save individual response results. Validated against RequestMetric where possible —
        # Kit's own metrics dict is always built from common_metrics.py's fixed vocabulary, so
        # this should always succeed; the fallback to the raw dict (with a logged warning)
        # exists only as a safety net so an unexpected key can never turn a completed benchmark
        # run into a lost result file.
        try:
            self.individual_responses_file_path = f'{results_dir}/{individual_responses_filename}.json'

            response_metrics = []
            for response in individual_responses:
                if not isinstance(response, LLMResponse):
                    continue
                try:
                    response_metrics.append(RequestMetric(**response.metrics).model_dump())
                except Exception as validation_error:
                    logger.warning(
                        f'RequestMetric validation failed for a response, writing raw dict instead: '
                        f'{validation_error}'
                    )
                    response_metrics.append(response.metrics)

            with open(self.individual_responses_file_path, 'w') as f:
                json.dump(response_metrics, f, indent=4)
        except Exception as e:
            logger.error(individual_responses)
            raise e

    def _build_benchmark_summary(self, name: str, summary: Dict[str, Any]) -> BenchmarkSummary:
        """Converts `get_token_throughput_latencies`'s mode-specific `summary` dict (already
        merged with `self.user_metadata`) into a canonical `BenchmarkSummary`.

        `summary['results']` mixes per-metric quantile dicts (built by `build_metrics_summary`,
        keyed by `common_metrics.py` field name) with run-level scalars
        (`num_requests_started`, `error_rate`, ...) in one flat dict — split apart here by
        whether each value is itself a dict.
        """
        results_raw = summary.get('results', {})
        results: Dict[str, QuantileStats] = {}
        scalar_fields: Dict[str, Any] = {}
        for key, value in results_raw.items():
            if isinstance(value, dict):
                results[key] = QuantileStats(**value)
            else:
                scalar_fields[key] = value

        # Custom mode's prompt-source metadata uses the key `sampling_params`; synthetic/
        # real_workload's uses `additional_sampling_params`. Both are preserved under their
        # own on-disk key name (matching today's per-mode file shape) — custom's goes through
        # `extra` (which flattens recursively too), synthetic/real_workload's through the
        # canonical `additional_sampling_params` field.
        additional_sampling_params = summary.get('additional_sampling_params', {}) or {}
        extra: Dict[str, Any] = {}
        if 'sampling_params' in summary:
            extra['sampling_params'] = summary['sampling_params']

        # Everything already consumed above (model, results, num_input_tokens, etc.) is dropped;
        # any leftover key (e.g. from a future preset) is preserved verbatim in `extra` rather
        # than silently discarded.
        known_top_level_keys = {
            'model', 'results', 'num_concurrent_requests', 'qps', 'qps_distribution',
            'num_input_tokens', 'num_output_tokens', 'request_count', 'sampling_params',
            'additional_sampling_params', *self.user_metadata.keys(),
        }
        for key, value in summary.items():
            if key not in known_top_level_keys:
                extra[key] = value
        extra.update(self.user_metadata)

        return BenchmarkSummary(
            tool='kit',
            workload_mode=self.workload_mode,
            name=name,
            timestamp=int(time.time()),
            model=summary.get('model', self.model_name),
            run_uuid=str(self.run_uuid),
            num_concurrent_requests=summary.get('num_concurrent_requests'),
            qps=summary.get('qps'),
            qps_distribution=summary.get('qps_distribution'),
            num_input_tokens=summary.get('num_input_tokens'),
            num_output_tokens=summary.get('num_output_tokens'),
            num_requests=summary.get('request_count'),
            results=results,
            num_requests_started=scalar_fields.get('num_requests_started'),
            num_completed_requests=scalar_fields.get('num_completed_requests'),
            num_completed_requests_per_min=scalar_fields.get('num_completed_requests_per_min'),
            error_rate=scalar_fields.get('error_rate'),
            number_errors=scalar_fields.get('number_errors'),
            error_code_frequency=scalar_fields.get('error_code_frequency'),
            client_total_output_throughput=scalar_fields.get('client_total_output_throughput'),
            mean_output_throughput_token_per_s=scalar_fields.get('mean_output_throughput_token_per_s'),
            additional_sampling_params=additional_sampling_params,
            extra=extra,
        )

    def stop_benchmark(self) -> None:
        """Stops the benchmarking process by setting the stop event."""
        self.stop_event.set()
        logger.info('Benchmarking process has been stopped.')

    def get_image(self, image_location: str = '') -> str:
        """Utility function for encoding an image to base64.

        Args:
            image_location (str, optional): Image location path. Defaults to ''.

        Returns:
            str: Encoded image in base64 format
        """
        if len(image_location) == 0:
            image_location = llmperf_utils.LVLM_IMAGE_PATHS[self.multimodal_image_size]
            image_location = os.path.join(kit_location, image_location)
        with open(image_location, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_image



class PromptSource(abc.ABC):
    """Strategy for building prompts and their `RequestConfig`s. Varies orthogonally from
    `LoadPattern` (how requests get scheduled) — see the multi-tool benchmarking architecture
    plan (section 2: "Unifying custom/synthetic/real_workload into one Kit-native engine").
    """

    # True for prompt sources that generate text to hit an exact input/output token count
    # (synthetic-shaped); False for sources that read pre-existing prompts (custom-shaped).
    # `PerformanceEvaluator.run_benchmark`/`get_token_throughput_latencies` branch on this to
    # decide whether num_input_tokens/num_output_tokens/num_requests apply.
    uses_fixed_token_counts: bool = False

    @abc.abstractmethod
    def build_prompt(
        self, evaluator: 'PerformanceEvaluator', *args: Any, **kwargs: Any
    ) -> Tuple[Dict[str, Any], int]:
        pass

    @abc.abstractmethod
    def build_request_configs(
        self,
        evaluator: 'PerformanceEvaluator',
        num_requests: Optional[int],
        sampling_params: Dict[str, Any],
        **kwargs: Any,
    ) -> List[RequestConfig]:
        pass

    def run_metadata(self, **kwargs: Any) -> Dict[str, Any]:
        """Prompt-source-specific fields contributed to the run's summary metadata dict."""
        return {}


class CustomPromptSource(PromptSource):
    """Reads user-provided prompts (and optional images) from a `.jsonl` dataset file."""

    uses_fixed_token_counts = False

    def __init__(self, input_file_path: str) -> None:
        self.file_name = os.path.basename(input_file_path)
        self.dataset = self.read_dataset(input_file_path)
        self.prompt_key = list(self.dataset[0].keys())[0]
        self.img_path_key = None
        if len(list(self.dataset[0].keys())) == 2:
            self.img_path_key = list(self.dataset[0].keys())[1]

    @staticmethod
    def read_dataset(input_file_path: str) -> List[Dict[str, Any]]:
        """Utility function for reading in the `.jsonl` file provided by the user for custom dataset evaluation.

        Args:
            input_file_path (str): The absolute file path of the input file provided by the user

        Returns:
            List[Dict]: A list of json objects (python dictionaries) containing the individual prompts the user wants
            to evaluate on
        """
        with open(input_file_path, 'r') as file:
            data = [json.loads(line) for line in file]

        # check if dataframe headers contain 'prompt'
        if not all([list(d.keys())[0] == 'prompt' for d in data]):
            raise ValueError(
                'All rows in input file must contain the same first column name "prompt" \
                and its respective text value'
            )

        # check if dataframe headers contain 'img_path' if there are two columns
        if all([len(list(d.keys())) == 2 for d in data]):
            if not all([list(d.keys())[1] == 'image_path' for d in data]):
                raise ValueError(
                    'If input file has two columns, all rows in input file must contain \
                    the same second column name "image_path" and its respective text value'
                )
            if any([d['image_path'].startswith('http') for d in data]):
                raise ValueError('Urls are not supported for image_path. Please provide local image paths.')
        # check if there are more than two columns
        elif any([len(list(d.keys())) > 2 for d in data]):
            raise ValueError('Input file can not contain more then two columns.')

        return data

    def build_prompt(
        self, evaluator: 'PerformanceEvaluator', raw_prompt: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], int]:
        """Builds an input prompt from the given raw prompt. Custom prompts are used verbatim, so this
        just measures the token length.

        Args:
        - raw_prompt (Dict[str, Any]): The raw input prompt dictionary to be used in building a processed input
          prompt.

        Returns:
        - A tuple containing the raw prompt dictionary and the token length of the prompt.
        """
        return (raw_prompt, evaluator.get_token_length(raw_prompt['template']))

    def build_request_configs(
        self,
        evaluator: 'PerformanceEvaluator',
        num_requests: Optional[int],
        sampling_params: Dict[str, Any],
        **kwargs: Any,
    ) -> List[RequestConfig]:
        """Builds a list of request configs for the LLM API. This method iterates through the provided dataset and
        builds a RequestConfig object for each data point. The RequestConfig object contains the necessary
        information to send a request to the LLM API, including the model name, prompt, sampling parameters, LLM API
        endpoint, generation mode, and number of concurrent requests. The method returns a list of these RequestConfig
        objects.

        Args:
            num_requests: unused — the custom dataset determines the request count (`len(self.dataset)`).
            sampling_params (Dict[str, Any]): A dictionary of sampling parameters to be passed into the RequestConfig
            constructor.

        Returns:
            List[RequestConfig]: A list of RequestConfig objects, each representing a request to the LLM API.
        """
        request_configs = []

        for request_idx, data_point in enumerate(self.dataset):
            raw_prompt = {'name': 'custom_prompt', 'template': data_point[self.prompt_key]}
            prompt_tuple = self.build_prompt(evaluator, raw_prompt)

            image = None
            if self.img_path_key:
                image = evaluator.get_image(data_point[self.img_path_key])

            request_config = RequestConfig(
                request_idx=request_idx,
                model=evaluator.model_name,
                prompt_tuple=prompt_tuple,
                image=image,
                sampling_params=sampling_params,
                llm_api=evaluator.llm_api,
                use_debugging_mode=evaluator.use_debugging_mode,
                api_variables=evaluator.api_variables,
                is_stream_mode=evaluator.is_stream_mode,
                num_concurrent_requests=evaluator.num_concurrent_requests,
            )

            request_configs.append(request_config)

        return request_configs

    def run_metadata(self, **kwargs: Any) -> Dict[str, Any]:
        return {'request_count': len(self.dataset), 'sampling_params': kwargs.get('sampling_params', {})}


class SyntheticPromptSource(PromptSource):
    """Generates a prompt by repeating a template and trimming/padding it to hit an exact
    input token count.
    """

    uses_fixed_token_counts = True

    def __init__(self, use_multiple_prompts: bool = False, supports_multiple_prompts: bool = True) -> None:
        self.use_multiple_prompts = use_multiple_prompts
        # Escape hatch for presets that can't support prompt cycling at all (e.g. a future
        # preset with a single fixed prompt source); every preset today can, so this stays True.
        self.supports_multiple_prompts = supports_multiple_prompts

    def load_prompts(self, prompts_file_path: str) -> Any:
        """Loads prompts from yaml file.

        Raises:
            ValueError: Validates if entries have the right structure.
            ValueError: Validates performance level entry values

        Returns:
            Dict[Dict[str, Any]]: List of dictionaries containing the name, performance level
            and text template of each prompt.
        """
        with open(prompts_file_path, 'r') as file:
            data = yaml.safe_load(file)

        valid_prompt_structure = ['name', 'template']

        # Validate the structure of the default prompt
        for prompt in data.get('default_prompt', []):
            if not all(key in prompt for key in valid_prompt_structure):
                raise ValueError(
                    f'Invalid prompt structure: {prompt}.\
                    It must include the fields {valid_prompt_structure}.'
                )

        # Validate the structure of the multiple prompts
        for prompt in data.get('multiple_prompts', []):
            if not all(key in prompt for key in valid_prompt_structure):
                raise ValueError(
                    f'Invalid prompt structure: {prompt}.\
                    It must include the fields {valid_prompt_structure}.'
                )

        return data

    def select_raw_prompts(self, raw_prompts: List[Dict[str, Any]], num_requests: int) -> List[Dict[str, Any]]:
        """Selects prompts randomly

        Args:
            num_requests (int): Number of requests to be generated

        Returns:
            List[Dict[str,Any]]: List of randomly selected prompts
        """
        random_selected_prompts = random.choices(raw_prompts, k=num_requests)
        assert len(random_selected_prompts) == num_requests, (
            'Number of selected prompts \
            does not match the requested count'
        )
        return random_selected_prompts

    def build_prompt(
        self, evaluator: 'PerformanceEvaluator', prompt_dict: Dict[str, Any], num_input_tokens: int
    ) -> Tuple[Dict[str, Any], int]:
        """Synthesizes an input prompt for the LLM to be queried. This prompt is created by repeating a prompt_template
        multiple times to reach a user set input_token_count.

        Args:
            prompt_dict (Dict[str, Any]): The raw input prompt dictionary to be used in building a processed input
            prompt.
            num_input_tokens (int): The user specified length of the input prompt.

        Returns:
            Tuple[str, int]: A tuple containing the generated prompt and its length in tokens.
        """
        max_words = num_input_tokens  # User-defined word limit

        # Calculate the maximum number of repetitions
        num_repeats = max(1, max_words // len(prompt_dict['template'].split()) + 1)

        # Repeat the prompt
        repeated_prompt_text = (prompt_dict['template'] + ' ') * num_repeats

        # Adjust prompt according to desired input tokens
        full_input_prompt_text = evaluator.adjust_to_exact_tokens(repeated_prompt_text, num_input_tokens)

        # Output prompt
        adjusted_prompt = prompt_dict
        adjusted_prompt['template'] = full_input_prompt_text

        return (adjusted_prompt, evaluator.get_token_length(full_input_prompt_text))

    def build_request_configs(
        self,
        evaluator: 'PerformanceEvaluator',
        num_requests: Optional[int],
        sampling_params: Dict[str, Any],
        **kwargs: Any,
    ) -> List[RequestConfig]:
        """Builds a list of request configuration objects used to send requests to the LLM. It iterates through the
        specified number of requests, builds an input prompt for each request, updates the sampling parameters with
        the maximum number of tokens to generate, and then creates the request configuration object. The request
        configurations are then returned as a list.

        Args:
            num_requests (int): The number of request configurations to build.
            sampling_params (dict): A dictionary of sampling parameters for the LLM.
            num_input_tokens (int): The number of input tokens to use when building the prompt.
            num_output_tokens (int): The number of output tokens each request should return.

        Returns:
            List[RequestConfig]: A list of request configurations, each containing the model name, prompt, sampling
            parameters, LLM API, generation mode, and number of concurrent requests.
        """
        input_token_count = kwargs['num_input_tokens']
        output_token_count = kwargs['num_output_tokens']
        assert num_requests is not None

        request_configs = []
        image = None

        # If not using multiple prompts (or this preset doesn't support them — real_workload)
        if not self.use_multiple_prompts or not self.supports_multiple_prompts:
            # Load prompts based on the model type
            if evaluator.multimodal_image_size == 'na':
                prompts_data = self.load_prompts(USER_PROMPT_TEXT_INSTRUCT_PATH)
                raw_prompt = prompts_data['default_prompt'][0]
            else:
                prompts_data = self.load_prompts(USER_PROMPT_VISION_INSTRUCT_PATH)
                raw_prompt = prompts_data['default_prompt'][0]
                image = evaluator.get_image()

            # Build input text prompt to be sent in LLM request
            prompt_tuple = self.build_prompt(evaluator, raw_prompt, input_token_count)

            # Iterate through data points and build a request config for each
            for request_idx in range(num_requests):
                updated_sampling_params = {'max_tokens_to_generate': output_token_count}
                updated_sampling_params.update(sampling_params)

                request_config = RequestConfig(
                    request_idx=request_idx,
                    model=evaluator.model_name,
                    prompt_tuple=prompt_tuple,
                    image=image,
                    sampling_params=updated_sampling_params,
                    llm_api=evaluator.llm_api,
                    use_debugging_mode=evaluator.use_debugging_mode,
                    api_variables=evaluator.api_variables,
                    is_stream_mode=evaluator.is_stream_mode,
                    num_concurrent_requests=evaluator.num_concurrent_requests,
                )

                request_configs.append(request_config)

        # If using multiple prompts
        else:
            if evaluator.multimodal_image_size != 'na':
                raise ValueError(
                    'Multiple prompts are not supported for multimodal models. '
                    'Please set use_multiple_prompts to False.'
                )

            # Load text-instruct prompts
            with open(USER_PROMPT_TEXT_INSTRUCT_PATH, 'r') as file:
                prompts_data = yaml.safe_load(file)
            raw_prompts = prompts_data['multiple_prompts']

            # Select text prompts randomly equal to the number of requests
            selected_raw_prompts = self.select_raw_prompts(raw_prompts, num_requests)

            # Build input prompt to be sent in LLM request
            for request_idx, raw_prompt in enumerate(selected_raw_prompts):
                prompt_tuple = self.build_prompt(evaluator, raw_prompt, input_token_count)

                updated_sampling_params = {'max_tokens_to_generate': output_token_count}
                updated_sampling_params.update(sampling_params)

                request_config = RequestConfig(
                    request_idx=request_idx,
                    model=evaluator.model_name,
                    prompt_tuple=prompt_tuple,
                    image=image,
                    sampling_params=updated_sampling_params,
                    llm_api=evaluator.llm_api,
                    api_variables=evaluator.api_variables,
                    is_stream_mode=evaluator.is_stream_mode,
                    num_concurrent_requests=evaluator.num_concurrent_requests,
                )

                request_configs.append(request_config)

        return request_configs

    def run_metadata(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            'num_input_tokens': kwargs.get('num_input_tokens'),
            'num_output_tokens': kwargs.get('num_output_tokens'),
            'additional_sampling_params': kwargs.get('sampling_params', {}),
        }


class LoadPattern(abc.ABC):
    """Strategy for scheduling/dispatching requests. Varies orthogonally from `PromptSource`."""

    @abc.abstractmethod
    def run(
        self, evaluator: 'PerformanceEvaluator', request_configs: List[RequestConfig], num_requests: int
    ) -> Tuple[List[LLMResponse], float, float]:
        """Dispatches `request_configs` and returns `(llm_responses, start_time, end_time)`."""
        pass

    def run_metadata(self) -> Dict[str, Any]:
        """Load-pattern-specific fields contributed to the run's summary metadata dict."""
        return {}


class ConcurrencyLoadPattern(LoadPattern):
    """Closed-loop concurrency: splits requests into `num_concurrent_requests` batches on a
    `ThreadPoolExecutor`, matching today's Custom/Synthetic `get_token_throughput_latencies`.
    """

    def __init__(self, num_concurrent_requests: int) -> None:
        self.num_concurrent_requests = num_concurrent_requests

    def run(
        self, evaluator: 'PerformanceEvaluator', request_configs: List[RequestConfig], num_requests: int
    ) -> Tuple[List[LLMResponse], float, float]:
        total_request_count = len(request_configs)
        request_config_batches: List[List[RequestConfig]] = []

        if self.num_concurrent_requests:
            requests_per_thread = total_request_count // self.num_concurrent_requests
            remainder = total_request_count % self.num_concurrent_requests

            idx = 0
            for concurrent_requests in range(self.num_concurrent_requests):
                num_requests_for_thread = requests_per_thread + (1 if concurrent_requests < remainder else 0)
                request_config_batch = request_configs[idx : idx + num_requests_for_thread].copy()
                idx += num_requests_for_thread
                request_config_batches.append(request_config_batch)

        llm_responses: List[LLMResponse] = []
        progress: List[Any] = []

        start_time = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.num_concurrent_requests) as executor:
            futures = []

            for request_config_batch in request_config_batches:
                if evaluator.stop_event.is_set():
                    logger.info('Stopping task submission due to stop signal.')
                    break

                future = executor.submit(
                    evaluator.send_requests,
                    request_config_batch,
                    llm_responses,
                    progress,
                    start_time,
                    num_requests,
                )
                futures.append(future)
                for t in executor._threads:
                    add_script_run_ctx(t)

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f'Error occurred in a thread: {e}')

        end_time = time.monotonic()
        return llm_responses, start_time, end_time

    def run_metadata(self) -> Dict[str, Any]:
        return {'num_concurrent_requests': self.num_concurrent_requests}


class QPSLoadPattern(LoadPattern):
    """Open-loop QPS-driven scheduler: submits one request at a time, sleeping between
    submissions according to a configurable distribution, matching today's
    `RealWorkLoadPerformanceEvaluator.get_token_throughput_latencies`.
    """

    def __init__(self, qps: float, qps_distribution: str = 'constant') -> None:
        self.qps = qps
        self.qps_distribution = qps_distribution

    def _get_wait_time(self) -> float:
        mean_wait = 1 / self.qps
        if self.qps_distribution == 'exponential':
            wait = random.expovariate(1 / mean_wait)
        elif self.qps_distribution == 'uniform':
            wait = random.uniform(0, 2 * mean_wait)
        elif self.qps_distribution == 'constant':
            wait = mean_wait
        else:
            raise ValueError(
                f'Unknown distribution {self.qps_distribution}. \
                Possible values: constant, uniform, exponential.'
            )
        return wait

    def run(
        self, evaluator: 'PerformanceEvaluator', request_configs: List[RequestConfig], num_requests: int
    ) -> Tuple[List[LLMResponse], float, float]:
        llm_responses: List[LLMResponse] = []
        progress: List[Any] = []

        start_time = time.monotonic()
        with ThreadPoolExecutor(max_workers=10000) as executor:
            futures = []

            for request_config in request_configs:
                if evaluator.stop_event.is_set():
                    logger.info('Stopping task submission due to stop signal.')
                    break

                future = executor.submit(
                    evaluator.send_requests,
                    [request_config],
                    llm_responses,
                    progress,
                    start_time,
                    num_requests,
                )
                futures.append(future)
                for t in executor._threads:
                    add_script_run_ctx(t)

                wait_time = self._get_wait_time()
                time.sleep(wait_time)

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f'Error occurred in a thread: {e}')

        end_time = time.monotonic()
        return llm_responses, start_time, end_time

    def run_metadata(self) -> Dict[str, Any]:
        return {'qps': self.qps, 'qps_distribution': self.qps_distribution}


class PerformanceEvaluator(BasePerformanceEvaluator):
    """Kit-native evaluation engine, parameterized by an orthogonal `PromptSource` (what to
    send) and `LoadPattern` (how to schedule sending it). `custom`/`synthetic`/`real_workload`
    are no longer three independent class hierarchies — they are three named
    `(PromptSource, LoadPattern)` presets, implemented below as thin, backward-compatible
    facade subclasses (`CustomPerformanceEvaluator`, `SyntheticPerformanceEvaluator`,
    `RealWorkLoadPerformanceEvaluator`) that keep their exact original constructor signatures
    and public behavior so every existing call site keeps working unchanged.
    """

    def __init__(
        self,
        prompt_source: PromptSource,
        load_pattern: LoadPattern,
        save_response_texts: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prompt_source = prompt_source
        self.load_pattern = load_pattern
        self.save_response_texts = save_response_texts

        # Mirror strategy-owned params onto self for backward-compat attribute access — e.g.
        # Streamlit's custom_performance_eval_st.py reads `performance_evaluator.num_concurrent_requests`
        # directly.
        self.num_concurrent_requests = getattr(load_pattern, 'num_concurrent_requests', None)
        self.qps = getattr(load_pattern, 'qps', None)
        self.qps_distribution = getattr(load_pattern, 'qps_distribution', None)
        self.use_multiple_prompts = getattr(prompt_source, 'use_multiple_prompts', False)

    def create_output_filename(self, *args: Any, **kwargs: Any) -> str:
        # Filename conventions are preset-specific (custom_/synthetic_/realworkload_ prefixes,
        # depending on both axes at once) — each facade subclass below overrides this.
        raise NotImplementedError('create_output_filename must be provided by a named preset subclass')

    def build_prompt(self, *args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], int]:
        return self.prompt_source.build_prompt(self, *args, **kwargs)

    def build_request_configs(self, *args: Any, **kwargs: Any) -> List[RequestConfig]:
        return self.prompt_source.build_request_configs(self, *args, **kwargs)

    def save_results(
        self,
        filename: str,
        summary: Dict[str, Any],
        individual_responses: (
            List[LLMResponse]
            | List[Tuple[Dict[str, Any], str, RequestConfig]]
            | Tuple[Dict[str, object], List[LLMResponse]]
        ),
    ) -> None:
        """Save the performance evaluation results to a file, and completion texts if save_response_text condition is
        setup as True

        Args:
            filename (str): The base name of the file to save the results to.
            summary (Dict[str, Any]): A dictionary containing the summary of the performance evaluation.
            individual_responses (List[LLMResponse]): A list of individual responses from the performance evaluation.

        Raises:
            e: if an error happens when creating the output file related to prompts and completions, an error will be
            raised
        """
        super().save_results(filename, summary, individual_responses)

        # If specified, save the llm responses to output file
        if self.save_response_texts:
            response_texts_file_name = f'{filename}_response_texts'
            results_dir = Path(self.results_dir)

            try:
                self.response_texts_file_path = f'{results_dir}/{response_texts_file_name}.jsonl'
                with open(self.response_texts_file_path, 'w') as f:
                    for response in individual_responses:
                        if isinstance(response, LLMResponse):
                            output_json = {
                                'prompt': response.request_config.prompt_tuple[0],
                                'completion': str(response.response_text),
                            }
                            f.write(json.dumps(output_json))
                            f.write('\n')
            except Exception as e:
                logger.error('ERROR SAVING LLM OUTPUTS')
                raise e

    def run_benchmark(
        self, sampling_params: Dict[str, Any] = {}, *args: Any, **kwargs: Any
    ) -> Tuple[Dict[str, Any], List[LLMResponse]]:
        """Run a benchmark test for the specified LLM.

        Args:
            sampling_params (Dict[str, Any]): The sampling parameters in JSON format.

        Returns:
            summary (dict): structure with performance metrics and stats for the run
            individual_responses (tuple): list of performance metrics per request
        """
        self.ui_progress_bar = kwargs.get('progress_bar', None)

        if self.prompt_source.uses_fixed_token_counts:
            num_input_tokens = kwargs.get('num_input_tokens', 1000)
            num_output_tokens = kwargs.get('num_output_tokens', 10)
            num_requests = kwargs.get('num_requests', 1)

            self.cli_progress_bar = tqdm(total=num_requests, desc='Running Requests')

            if num_input_tokens < 40:
                raise ValueError(
                    'The minimum number of input tokens that will be sent is 40 because of the prompting logic '
                    'right now'
                )

            summary, individual_responses = self.get_token_throughput_latencies(
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                num_requests=num_requests,
                sampling_params=sampling_params,
            )
            filename_args: Tuple[Any, ...] = (num_input_tokens, num_output_tokens)
        else:
            dataset = getattr(self.prompt_source, 'dataset', [])
            self.cli_progress_bar = tqdm(total=len(dataset), desc='Running Requests')

            summary, individual_responses = self.get_token_throughput_latencies(sampling_params=sampling_params)
            filename_args = ()

        if self.results_dir:
            filename = self.create_output_filename(*filename_args)
            self.save_results(filename, summary, individual_responses)

        return summary, individual_responses

    def get_token_throughput_latencies(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[Dict[str, Any], List[LLMResponse]]:
        """This function runs a token benchmark for the given model and API, measuring the throughput and
        latencies for the configured prompt source and load pattern.

        Returns:
            metadata (dict): A dictionary containing the results of the benchmark.
            completed_requests (list): A list of completed requests.

        Raises:
            Exception: If an unexpected error occurs during the execution of requests.
        """
        sampling_params = kwargs.get('sampling_params', {})

        if self.prompt_source.uses_fixed_token_counts:
            num_input_tokens = kwargs['num_input_tokens']
            num_output_tokens = kwargs['num_output_tokens']
            num_requests = kwargs['num_requests']
            request_configs = self.build_request_configs(
                num_requests,
                sampling_params,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
            )
        else:
            num_input_tokens = None
            num_output_tokens = None
            request_configs = self.build_request_configs(None, sampling_params)
            # Custom mode: the dataset (not a caller-supplied value) determines the request count.
            num_requests = len(request_configs)

        # Single timeout budget spanning warm-up + measured run: a timeout stops whichever phase
        # is active. Set before warm-up so warm-up time counts against the same budget.
        self.deadline = time.monotonic() + self.timeout

        # Warm-up phase (discarded, runs before the measured clock starts)
        if self.num_warmup_requests:
            self.run_warmup(self.build_warmup_configs(request_configs))

        # If the shared timeout was exhausted during warm-up, stop before measuring.
        if time.monotonic() >= self.deadline:
            logger.warning('Timeout reached during warm-up; skipping measured run (no results collected).')
            return {}, []

        llm_responses, start_time, end_time = self.load_pattern.run(self, request_configs, num_requests)

        if self.stop_event.is_set():
            logger.info('Benchmarking process terminated early due to stop signal.')
            return {}, []

        # Error handling
        error_codes = [llm_response.metrics['error_code'] for llm_response in llm_responses]

        if not any([pd.isnull(error_code) for error_code in error_codes]):
            unique_error_codes = list(
                set(
                    [
                        llm_response.metrics['error_code']
                        for llm_response in llm_responses
                        if not pd.isnull(llm_response.metrics['error_code'])
                    ]
                )
            )
            unique_error_msgs = list(
                set(
                    [
                        llm_response.metrics['error_msg']
                        for llm_response in llm_responses
                        if not pd.isnull(llm_response.metrics['error_code'])
                    ]
                )
            )
            nl = '\n'
            raise Exception(
                f"""Unexpected error happened when executing requests:\
                {nl}{f'{nl}'.join([f'- {error_code}' for error_code in unique_error_codes])}\
                {nl}Additional messages:{nl}{f'{nl}'.join([f'- {error_msg}' for error_msg in unique_error_msgs])}"""
            )

        logger.info('Tasks Executed!')
        logger.info(f'Benchmarking results obtained for model {self.model_name} queried with the {self.llm_api} API.')

        results = self.build_metrics_summary(
            metrics=[response.metrics for response in llm_responses],
            start_time=start_time,
            end_time=end_time,
        )

        metadata: Dict[str, Any] = {'model': self.model_name, **self.load_pattern.run_metadata(), 'results': results}
        metadata.update(
            self.prompt_source.run_metadata(
                sampling_params=sampling_params,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
            )
        )

        return metadata, llm_responses


class CustomPerformanceEvaluator(PerformanceEvaluator):
    """Preset: custom prompt source (user-provided `.jsonl` dataset) + closed-loop concurrency."""

    def __init__(
        self,
        num_concurrent_requests: int,
        input_file_path: str,
        save_response_texts: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        prompt_source = CustomPromptSource(input_file_path)
        super().__init__(
            prompt_source,
            ConcurrencyLoadPattern(num_concurrent_requests),
            save_response_texts,
            *args,
            **kwargs,
        )
        self.file_name = prompt_source.file_name
        self.workload_mode = 'custom'

    def create_output_filename(self) -> str:
        """Utility for creating a unique filename for a custom benchmarking experiment with a dataset.

        Returns:
            str: Filename for the custom benchmark run.
        """
        generation_mode = ''
        if self.is_stream_mode:
            generation_mode = 'stream'

        model_name = self.model_name.replace('_', '-')
        output_file_name = f'custom_{model_name}_{self.file_name}_\
            {self.num_concurrent_requests}_{generation_mode}_{self.run_uuid}'
        return self.sanitize_file_prefix(output_file_name)


class SyntheticPerformanceEvaluator(PerformanceEvaluator):
    """Preset: synthetic prompt source (repeat-and-trim to exact token count) + closed-loop concurrency."""

    def __init__(
        self,
        num_concurrent_requests: int,
        use_multiple_prompts: bool = False,
        save_response_texts: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            SyntheticPromptSource(use_multiple_prompts),
            ConcurrencyLoadPattern(num_concurrent_requests),
            save_response_texts,
            *args,
            **kwargs,
        )
        self.workload_mode = 'synthetic'

    def create_output_filename(self, num_input_tokens: int, num_output_tokens: int) -> str:
        """Utility for creating a unique filename for a synthetic benchmarking experiment given user specified params.

        Returns:
            str: Filename for the synthetic benchmark run.
        """
        generation_mode = ''
        if self.is_stream_mode:
            generation_mode = 'stream'

        multimodal_suffix = ''
        if self.multimodal_image_size != 'na':
            multimodal_suffix = f'_multimodal_{self.multimodal_image_size}'

        model_name = self.model_name.replace('_', '-')
        output_file_name = (
            f'synthetic_{self.user_metadata["model_idx"]}_{model_name}{multimodal_suffix}_{num_input_tokens}'
            f'_{num_output_tokens}_{self.num_concurrent_requests}_{generation_mode}_{self.run_uuid}'
        )

        return self.sanitize_file_prefix(output_file_name)


class RealWorkLoadPerformanceEvaluator(PerformanceEvaluator):
    """Preset: synthetic prompt source (single template by default, or cycled at random via
    `use_multiple_prompts`) + open-loop QPS-driven load pattern.
    """

    def __init__(
        self,
        qps: float,
        qps_distribution: str = 'constant',
        num_concurrent_requests: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        use_multiple_prompts = kwargs.pop('use_multiple_prompts', False)
        save_response_texts = kwargs.pop('save_response_texts', False)
        super().__init__(
            SyntheticPromptSource(use_multiple_prompts),
            QPSLoadPattern(qps, qps_distribution),
            save_response_texts,
            *args,
            **kwargs,
        )
        # QPSLoadPattern has no num_concurrent_requests of its own; restore the constructor
        # value here to match today's RealWorkLoadPerformanceEvaluator attribute exactly.
        self.num_concurrent_requests = num_concurrent_requests
        self.workload_mode = 'real_workload'

    def create_output_filename(self, num_input_tokens: int, num_output_tokens: int) -> str:
        """Utility for creating a unique filename for a synthetic benchmarking experiment given user specified params.

        Returns:
            str: Filename for the synthetic benchmark run.
        """
        generation_mode = ''
        if self.is_stream_mode:
            generation_mode = 'stream'

        multimodal_suffix = ''
        if self.multimodal_image_size != 'na':
            multimodal_suffix = f'_multimodal_{self.multimodal_image_size}'

        model_name = self.model_name.replace('_', '-')
        output_file_name = (
            f'realworkload_{self.user_metadata["model_idx"]}_{model_name}{multimodal_suffix}_{num_input_tokens}'
            f'_{num_output_tokens}_{self.qps}_{self.qps_distribution}_{generation_mode}_{self.run_uuid}'
        )

        return self.sanitize_file_prefix(output_file_name)
