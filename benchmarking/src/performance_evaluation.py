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
kit_location = os.path.join(file_location, '../')

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import transformers
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import add_script_run_ctx
from tqdm import tqdm

import benchmarking.src.llmperf.llmperf_utils as llmperf_utils
from benchmarking.src.llmperf import common_metrics
from benchmarking.src.llmperf.llmperf_utils import (
    LLMPerfResults,
    flatten,
    get_tokenizer,
)
from benchmarking.src.llmperf.models import LLMResponse, RequestConfig
from benchmarking.src.llmperf.sambanova_client import llm_request
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
load_dotenv('../.env', override=True)

SYSTEM_PROMPT_PATH = os.path.join(file_location, '../prompts/system-prompt_template.yaml')
USER_PROMPT_TEXT_INSTRUCT_PATH = os.path.join(file_location, '../prompts/user-prompt_template-text_instruct.yaml')
USER_PROMPT_VISION_INSTRUCT_PATH = os.path.join(file_location, '../prompts/user-prompt_template-vision_instruct.yaml')


class BasePerformanceEvaluator(abc.ABC):
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
        self.tokenizer = get_tokenizer(self.model_name)
        self.stop_event = threading.Event()
        self.ui_progress_bar = None
        self.cli_progress_bar = None
        self.run_uuid = uuid.uuid4()

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
        for request_config in request_config_batch:
            if self.stop_event.is_set():
                logger.info('Stopping request processing in thread due to stop signal.')
                break
            if time.monotonic() - start_time >= self.timeout:
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
                self.ui_progress_bar(len(progress), num_requests)

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
        ]:
            if self.show_results_in_terminal:
                logger.info(f'Building Client Metrics Summary for metric: {metric}')
            metrics_summary[metric] = {}

            # Get flattened list from metric column in metrics df
            series = pd.Series(list(flatten(metrics_df[metric]))).dropna()

            # Generate statistics for specific metric
            quantiles = series.quantile([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(4).to_dict()
            quantiles_reformatted_keys = {}
            for quantile, value in quantiles.items():
                reformatted_key = f'p{int(quantile * 100)}'
                if self.show_results_in_terminal:
                    logger.info(f'    {reformatted_key} = {value}')
                quantiles_reformatted_keys[reformatted_key] = value
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
                logger.info(f'    mean = {series_mean}')
                logger.info(f'    min = {series_min}')
                logger.info(f'    max = {series_max}')
                logger.info(f'    stddev = {series_std}')

        # Record descriptive statistics for the metrics in the following list
        for metric in [
            common_metrics.TTFT_SERVER,
            common_metrics.E2E_LAT_SERVER,
            common_metrics.REQ_OUTPUT_THROUGHPUT_SERVER,
            common_metrics.REQ_OUTPUT_THROUGHPUT_SERVER_FIRST_TEN,
            common_metrics.REQ_OUTPUT_THROUGHPUT_SERVER_FIRST_TEN,
            common_metrics.NUM_INPUT_TOKENS_SERVER,
            common_metrics.NUM_OUTPUT_TOKENS_SERVER,
            common_metrics.ACCEPTANCE_RATE,
        ]:
            if self.show_results_in_terminal:
                logger.info(f'Building Server Metrics Summary for metric: {metric}')
            metrics_summary[metric] = {}

            # Get flattened list from metric column in metrics df
            series = pd.Series(list(flatten(metrics_df[metric]))).dropna()

            # Generate statistics for specific metric
            quantiles = series.quantile([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(4).to_dict()
            quantiles_reformatted_keys = {}
            for quantile, value in quantiles.items():
                reformatted_key = f'p{int(quantile * 100)}'
                if self.show_results_in_terminal:
                    logger.info(f'    {reformatted_key} = {value}')
                quantiles_reformatted_keys[reformatted_key] = value
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
                logger.info(f'    mean = {series_mean}')
                logger.info(f'    min = {series_min}')
                logger.info(f'    max = {series_max}')
                logger.info(f'    stddev = {series_std}')

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

        # Update to metadata.
        summary.update(self.user_metadata)

        results = LLMPerfResults(name=summary_filename, metadata=summary)
        results_dir = Path(self.results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f'{results_dir} is not a directory')

        # Save summary results
        try:
            self.summary_file_path = f'{results_dir}/{summary_filename}.json'
            with open(self.summary_file_path, 'w') as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            logger.error(results.to_dict())
            raise e

        # Save individual response results
        try:
            self.individual_responses_file_path = f'{results_dir}/{individual_responses_filename}.json'

            response_metrics = [
                response.metrics for response in individual_responses if isinstance(response, LLMResponse)
            ]
            with open(self.individual_responses_file_path, 'w') as f:
                json.dump(response_metrics, f, indent=4)
        except Exception as e:
            logger.error(individual_responses)
            raise e

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


class CustomPerformanceEvaluator(BasePerformanceEvaluator):
    def __init__(
        self,
        num_concurrent_requests: int,
        input_file_path: str,
        save_response_texts: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_concurrent_requests = num_concurrent_requests
        self.file_name = os.path.basename(input_file_path)
        self.dataset = self.read_dataset(input_file_path)
        self.prompt_key = list(self.dataset[0].keys())[0]
        self.img_path_key = None
        if len(list(self.dataset[0].keys())) == 2:
            self.img_path_key = list(self.dataset[0].keys())[1]
        self.save_response_texts = save_response_texts

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
            # Create response texts file name
            response_texts_file_name = f'{filename}_response_texts'
            results_dir = Path(self.results_dir)

            # Save response texts
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
        """Run a benchmark test for the specified LLM using a custom dataset provided by the user.

        Args:
            sampling_params (Dict[str, Any]): The sampling parameters in JSON format.

        Returns:
            None
        """
        self.cli_progress_bar = tqdm(total=len(self.dataset), desc='Running Requests')
        self.ui_progress_bar = kwargs.get('progress_bar', None)

        # Calculate performance metrics individually and summary
        summary, individual_responses = self.get_token_throughput_latencies(
            sampling_params=sampling_params,
        )

        # Save benchmarking results to the specified results directory, it it exists
        if self.results_dir:
            filename = self.create_output_filename()
            self.save_results(
                filename,
                summary,
                individual_responses,
            )
        return summary, individual_responses

    def get_token_throughput_latencies(
        self, sampling_params: Dict[str, Any]
    ) -> Tuple[dict[str, Any], List[LLMResponse]]:
        """This function is used to measure the token throughput and latencies.

        Args:
            sampling_params (Dict[str, Any]): A dictionary containing the parameters for sampling.

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, Any]]]: A tuple containing metadata and completed requests.

        Raises:
            Exception: If an unexpected error happens when executing requests.

        Note:
            This function uses threading to send requests concurrently. It splits the total request count evenly among
            the threads. If there is a remainder, it assigns one extra request to the first threads.
        """
        # random.seed(11111)

        request_configs = self.build_request_configs(
            sampling_params,
        )

        # Get batch size details
        total_request_count = len(request_configs)
        request_config_batches: List[List[RequestConfig]] = []

        if self.num_concurrent_requests:
            requests_per_thread = total_request_count // self.num_concurrent_requests
            remainder = total_request_count % self.num_concurrent_requests

            idx = 0
            # Create batches of requests for each concurrent request
            for concurrent_requests in range(self.num_concurrent_requests):
                num_requests_for_thread = requests_per_thread + (1 if concurrent_requests < remainder else 0)
                request_config_batch = request_configs[idx : idx + num_requests_for_thread].copy()
                idx = idx + num_requests_for_thread
                request_config_batches.append(request_config_batch)

        # Execute requests concurrently
        llm_responses: List[LLMResponse] = []
        progress: List[Any] = []

        start_time = time.monotonic()
        # Use ThreadPoolExecutor to handle threads
        with ThreadPoolExecutor(max_workers=self.num_concurrent_requests) as executor:
            # Store futures for the tasks
            futures = []

            for request_config_batch in request_config_batches:
                if self.stop_event.is_set():
                    logger.info('Stopping task submission due to stop signal.')
                    break

                # Submit the task to the executor
                future = executor.submit(
                    self.send_requests,
                    request_config_batch,
                    llm_responses,
                    progress,
                    start_time,
                    total_request_count,
                )
                futures.append(future)
                for t in executor._threads:
                    add_script_run_ctx(t)

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    # Retrieve result if needed
                    future.result()
                except Exception as e:
                    logger.error(f'Error occurred in a thread: {e}')

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

        end_time = time.monotonic()
        logger.info('Tasks Executed!')
        logger.info(f'Benchmarking results obtained for model {self.model_name} queried with the {self.llm_api} API.')
        results = self.build_metrics_summary(
            metrics=[response.metrics for response in llm_responses],
            start_time=start_time,
            end_time=end_time,
        )

        metadata = {
            'model': self.model_name,
            'num_concurrent_requests': self.num_concurrent_requests,
            'results': results,
            'request_count': len(self.dataset),
            'sampling_params': sampling_params,
        }

        return metadata, llm_responses

    def build_request_configs(self, sampling_params: Dict[str, Any]) -> List[RequestConfig]:
        """Builds a list of request configs for the LLM API. This method iterates through the provided dataset and
        builds a RequestConfig object for each data point. The RequestConfig object contains the necessary information
        to send a request to the LLM API, including the model name, prompt, sampling parameters, LLM API endpoint,
        generation mode, and number of concurrent requests. The method returns a list of these RequestConfig objects.

        Args:
            sampling_params (Dict[str, Any]): A dictionary of sampling parameters to be passed into the RequestConfig
            constructor.

        Returns:
            List[RequestConfig]: A list of RequestConfig objects, each representing a request to the LLM API.
        """
        # Empty list to be filled with valid request configs and then returned
        request_configs = []

        # Iterate through data points and build a request config for each
        for request_idx, data_point in enumerate(self.dataset):
            # Make raw prompt dictionary
            raw_prompt = {'name': 'custom_prompt', 'template': data_point[self.prompt_key]}

            # Apply prompt templating to get final prompt to send to LLM API along with tokenized prompt length
            prompt_tuple = self.build_prompt(raw_prompt=raw_prompt)

            # Image to be sent in LLM request if exists
            image = None
            if self.img_path_key:
                image = self.get_image(data_point[self.img_path_key])

            request_config = RequestConfig(
                request_idx=request_idx,
                model=self.model_name,
                prompt_tuple=prompt_tuple,
                image=image,
                sampling_params=sampling_params,
                llm_api=self.llm_api,
                use_debugging_mode=self.use_debugging_mode,
                api_variables=self.api_variables,
                is_stream_mode=self.is_stream_mode,
                num_concurrent_requests=self.num_concurrent_requests,
            )

            request_configs.append(request_config)

        return request_configs

    def build_prompt(self, raw_prompt: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Builds an input prompt from the given raw prompt by applying prompt templating based on the model type.

        Args:
        - raw_prompt (Dict[str, Any]): The raw input prompt dictionary to be used in building a processed input prompt.

        Returns:
        - A tuple containing the raw prompt dictionary and the token length of the prompt.

        Description:
        This method builds a prompt for the given raw prompt based on the model type.
        The method returns a tuple containing the raw prompt and the token length of the prompt.
        """

        return (raw_prompt, self.get_token_length(raw_prompt['template']))


class SyntheticPerformanceEvaluator(BasePerformanceEvaluator):
    def __init__(
        self,
        num_concurrent_requests: int,
        use_multiple_prompts: bool = False,
        save_response_texts: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_concurrent_requests = num_concurrent_requests
        self.use_multiple_prompts = use_multiple_prompts
        self.save_response_texts = save_response_texts

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
            # Create response texts file name
            response_texts_file_name = f'{filename}_response_texts'
            results_dir = Path(self.results_dir)

            # Save response texts
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
        """Run a benchmark test for the specified LLM using synthetically generated data.

        Args:
            num_input_tokens (int): The number of input tokens to be sent.
            num_output_tokens (int): The number of output tokens to be received.
            num_requests (int): The number of requests to be made.
            sampling_params (str): The sampling parameters in JSON format.

        Raises:
            ValueError: If the number of input tokens is less than 40.

        Returns:
            summary (dict): structure with performance metrics and stats for the run
            individual_responses (tuple): list of performance metrics per request
        """
        num_input_tokens = kwargs.get('num_input_tokens', 1000)
        num_output_tokens = kwargs.get('num_output_tokens', 10)
        num_requests = kwargs.get('num_requests', 1)

        self.cli_progress_bar = tqdm(total=num_requests, desc='Running Requests')
        self.ui_progress_bar = kwargs.get('progress_bar', None)

        if num_input_tokens < 40:
            raise ValueError(
                'The minimum number of input tokens that will be sent is 40 because of the prompting logic right now'
            )

        # Calculate performance metrics individually and summary
        summary, individual_responses = self.get_token_throughput_latencies(
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            num_requests=num_requests,
            sampling_params=sampling_params,
        )

        if self.results_dir:
            filename = self.create_output_filename(num_input_tokens, num_output_tokens)
            self.save_results(filename, summary, individual_responses)

        return summary, individual_responses

    def add_metric_after_key(
        self,
        metrics_dict: Dict[str, Any],
        new_key: str,
        new_value: float,
        after_key: str,
    ) -> Dict[str, Any]:
        """Adds a new metric (dict key and value) to a dict after an specific key

        Args:
            metrics_dict (dict): dictionary to add new metric
            new_key (str): new key
            new_value (float): new value for key
            after_key (str): key for reference to add new key after

        Returns:
            dict: dictionary with new key and value added
        """

        # Create a new dictionary
        new_metrics_dict = {}

        for key, value in metrics_dict.items():
            # Copy the key-value pair to the new dictionary
            new_metrics_dict[key] = value

            # Check if this is the key after which to insert the new key-value pair
            if key == after_key:
                new_metrics_dict[new_key] = new_value

        return new_metrics_dict

    def calculate_switching_time(self, llm_responses: list[LLMResponse]) -> list[LLMResponse]:
        """Logic to calculate switching time. Based on the first request TTFT,
        if this value is significantly larger (more than 3 standard deviations) than the average TTFT
        of the rest requests, then switching time will be the difference between first TTFT
        and average of the coming TTFTs.

        Args:
            llm_responses (list[LLMResponse]): list of LLMResponse objects

        Returns:
            list[LLMResponse]: list of LLMResponse objects including switching time
        """
        # collect necessary information for switching time calculation
        responses_ttfts = []

        for llm_response in llm_responses:
            if pd.isnull(llm_response.metrics['error_code']):
                request_idx = llm_response.request_config.request_idx
                start_time = llm_response.metrics['start_time']
                server_ttft_s = llm_response.metrics['server_ttft_s']
                responses_ttfts.append(
                    {
                        'request_idx': request_idx,
                        'start_time': start_time,
                        'server_ttft_s': server_ttft_s,
                    }
                )

        df_valid_responses = pd.DataFrame(responses_ttfts)

        # transforming str to date time for sorting
        df_valid_responses['start_time'] = pd.to_datetime(df_valid_responses['start_time'])
        df_valid_responses = df_valid_responses.sort_values(by=['start_time'])

        # initialize a column for the switching time
        df_valid_responses['server_switching_time'] = None

        # check server ttft in case metric is not coming in response
        if df_valid_responses['server_ttft_s'].notna().all():
            # calculate switching time
            first_ttft = df_valid_responses['server_ttft_s'].iloc[0]
            mean_ttft = df_valid_responses['server_ttft_s'].iloc[1:].mean()
            std_ttft = df_valid_responses['server_ttft_s'].iloc[1:].std()
            std_ttft = 1e-16 if np.isnan(std_ttft) else std_ttft

            switching_time = first_ttft - mean_ttft
            outlier_switching_time = None

            if switching_time > (mean_ttft + 3 * std_ttft):
                outlier_switching_time = switching_time
                df_valid_responses['server_switching_time'].iloc[0] = outlier_switching_time

        # assign switching time back to request object
        for llm_response in llm_responses:
            metrics = llm_response.metrics

            if llm_response.request_config.request_idx == df_valid_responses.head(1)['request_idx'].values[0]:
                server_switching_time = df_valid_responses.head(1)['server_switching_time'].values[0]
            else:
                server_switching_time = None

            llm_response.metrics = self.add_metric_after_key(
                metrics,
                new_key='server_switching_time',
                new_value=server_switching_time,
                after_key=common_metrics.TTFT_SERVER,
            )

        return llm_responses

    def get_token_throughput_latencies(
        self,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        sampling_params: Dict[str, Any],
    ) -> Tuple[dict[str, Any], List[LLMResponse]]:
        """This function runs a token benchmark for the given model and API,
        measuring the throughput and latencies for the specified number of input and output tokens,
        and the specified number of requests.

        Args:
            num_input_tokens (int): The user specified number of input tokens.
            num_output_tokens (int): The user specified number of output tokens.
            num_requests (int): The user specified number of requests to run.
            sampling_params (dict): User specified sampling parameters for generation.

        Returns:
            metadata (dict): A dictionary containing the results of the benchmark,
                            including the model name, number of concurrent requests,
                            results, number of input tokens, number of output tokens,
                            and additional sampling parameters.
            completed_requests (list): A list of completed requests.

        Raises:
            Exception: If an unexpected error occurs during the execution of requests.
        """
        # random.seed(11111)

        # Build the request config objects that are to be sent to the LLM API endpoint
        request_configs = self.build_request_configs(num_requests, num_input_tokens, num_output_tokens, sampling_params)

        # Get the request counts in order to place them into threads to be executed in batches
        total_request_count = len(request_configs)
        request_config_batches: List[List[RequestConfig]] = []

        if self.num_concurrent_requests:
            requests_per_thread = (total_request_count) // self.num_concurrent_requests
            remainder = (total_request_count) % self.num_concurrent_requests

            idx = 0
            # Create batches of requests for each concurrent request
            for concurrent_requests in range(self.num_concurrent_requests):
                num_requests_for_thread = requests_per_thread + (1 if concurrent_requests < remainder else 0)
                request_config_batch = request_configs[idx : idx + num_requests_for_thread].copy()
                idx += num_requests_for_thread
                request_config_batches.append(request_config_batch)

        # Execute requests concurrently
        llm_responses: List[LLMResponse] = []
        progress: List[Any] = []

        start_time = time.monotonic()
        # Use ThreadPoolExecutor to handle threads
        with ThreadPoolExecutor(max_workers=self.num_concurrent_requests) as executor:
            # Store futures for the tasks
            futures = []

            for request_config_batch in request_config_batches:
                if self.stop_event.is_set():
                    logger.info('Stopping task submission due to stop signal.')
                    break
                # Submit the task to the executor
                future = executor.submit(
                    self.send_requests,
                    request_config_batch,
                    llm_responses,
                    progress,
                    start_time,
                    num_requests,
                )
                futures.append(future)
                for t in executor._threads:
                    add_script_run_ctx(t)

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    # Retrieve result if needed
                    future.result()
                except Exception as e:
                    logger.error(f'Error occurred in a thread: {e}')

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

        # Capture end time and notify user
        end_time = time.monotonic()
        logger.info('Tasks Executed!')
        logger.info(f'Benchmarking results obtained for model {self.model_name} queried with the {self.llm_api} API.')

        # Calculate switching time
        llm_responses = self.calculate_switching_time(llm_responses)

        # Build a metrics summary for the results of the benchmarking run
        results = self.build_metrics_summary(
            metrics=[response.metrics for response in llm_responses],
            start_time=start_time,
            end_time=end_time,
        )

        # Construct metadata payload to be returned
        metadata = {
            'model': self.model_name,
            'num_concurrent_requests': self.num_concurrent_requests,
            'results': results,
            'num_input_tokens': num_input_tokens,
            'num_output_tokens': num_output_tokens,
            'additional_sampling_params': sampling_params,
        }

        return metadata, llm_responses

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

    def build_request_configs(
        self,
        num_requests: int,
        input_token_count: int,
        output_token_count: int,
        sampling_params: Dict[str, Any],
    ) -> List[RequestConfig]:
        """Builds a list of request configuration objects used to send requests to the LLM. It iterates through the
        specified number of requests, builds an input prompt for each request, updates the sampling parameters with
        the maximum number of tokens to generate, and then creates the request configuration object. The request
        configurations are then returned as a list.

        Args:
            num_requests (int): The number of request configurations to build.
            input_token_count (int): The number of input tokens to use when building the prompt.
            output_token_count (int): The number of output tokens each request should return.
            sampling_params (dict): A dictionary of sampling parameters for the LLM.

        Returns:
            List[RequestConfig]: A list of request configurations, each containing the model name, prompt, sampling
            parameters, LLM API, generation mode, and number of concurrent requests.
        """
        # Empty list to be filled with valid request configs and then returned
        request_configs = []
        # Instantiate image variable
        image = None

        # If not using multiple prompts
        if not self.use_multiple_prompts:
            # Load prompts based on the model type
            if self.multimodal_image_size == 'na':
                # Read prompt for text-instruct model
                prompts_data = self.load_prompts(USER_PROMPT_TEXT_INSTRUCT_PATH)
                raw_prompt = prompts_data['default_prompt'][0]

            else:
                # Read prompt for vision-instruct model
                prompts_data = self.load_prompts(USER_PROMPT_VISION_INSTRUCT_PATH)
                raw_prompt = prompts_data['default_prompt'][0]
                image = self.get_image()

            # Build input text prompt to be sent in LLM request
            prompt_tuple = self.build_prompt(raw_prompt, input_token_count)

            # Iterate through data points and build a request config for each
            for request_idx in range(num_requests):
                # Add generic max tokens parameter to `sampling_params` dictionary
                updated_sampling_params = {
                    'max_tokens_to_generate': output_token_count,
                }
                updated_sampling_params.update(sampling_params)

                # Create request config object
                request_config = RequestConfig(
                    request_idx=request_idx,
                    model=self.model_name,
                    prompt_tuple=prompt_tuple,
                    image=image,
                    sampling_params=updated_sampling_params,
                    llm_api=self.llm_api,
                    use_debugging_mode=self.use_debugging_mode,
                    api_variables=self.api_variables,
                    is_stream_mode=self.is_stream_mode,
                    num_concurrent_requests=self.num_concurrent_requests,
                )

                request_configs.append(request_config)

        # If using multiple prompts
        else:
            if self.multimodal_image_size != 'na':
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
            # Iterate through data points and build a request config for each
            for request_idx, raw_prompt in enumerate(selected_raw_prompts):
                prompt_tuple = self.build_prompt(raw_prompt, input_token_count)

                updated_sampling_params = {'max_tokens_to_generate': output_token_count}
                updated_sampling_params.update(sampling_params)

                request_config = RequestConfig(
                    request_idx=request_idx,
                    model=self.model_name,
                    prompt_tuple=prompt_tuple,
                    image=image,
                    sampling_params=updated_sampling_params,
                    llm_api=self.llm_api,
                    api_variables=self.api_variables,
                    is_stream_mode=self.is_stream_mode,
                    num_concurrent_requests=self.num_concurrent_requests,
                )

                request_configs.append(request_config)

        return request_configs

    def build_prompt(self, prompt_dict: Dict[str, Any], num_input_tokens: int) -> Tuple[Dict[str, Any], int]:
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
        full_input_prompt_text = self.adjust_to_exact_tokens(repeated_prompt_text, num_input_tokens)

        # Output prompt
        adjusted_prompt = prompt_dict
        adjusted_prompt['template'] = full_input_prompt_text

        return (adjusted_prompt, self.get_token_length(full_input_prompt_text))


class RealWorkLoadPerformanceEvaluator(SyntheticPerformanceEvaluator):
    def __init__(
        self,
        qps: float,
        qps_distribution: str = 'constant',
        num_concurrent_requests: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_concurrent_requests, *args, **kwargs)
        self.qps = qps
        self.qps_distribution = qps_distribution

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

    def get_token_throughput_latencies(
        self,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        sampling_params: Dict[str, Any],
    ) -> Tuple[dict[str, Any], List[LLMResponse]]:
        """This function runs a token benchmark for the given model and API,
        measuring the throughput and latencies for the specified number of input and output tokens,
        and the specified number of requests.

        Args:
            qps (float): Queries per second to be sent to the LLM API.
            qps_distribution (str): Distribution name of queries per second.
            num_input_tokens (int): The user specified number of input tokens.
            num_output_tokens (int): The user specified number of output tokens.
            num_requests (int): The user specified number of requests to run.
            sampling_params (dict): User specified sampling parameters for generation.

        Returns:
            metadata (dict): A dictionary containing the results of the benchmark,
                            including the model name, number of concurrent requests,
                            results, number of input tokens, number of output tokens,
                            and additional sampling parameters.
            completed_requests (list): A list of completed requests.

        Raises:
            Exception: If an unexpected error occurs during the execution of requests.
        """
        # random.seed(11111)

        # Build the request config objects that are to be sent to the LLM API endpoint
        request_configs = self.build_request_configs(num_requests, num_input_tokens, num_output_tokens, sampling_params)

        # Execute requests concurrently
        llm_responses: List[LLMResponse] = []
        progress: List[Any] = []

        start_time = time.monotonic()
        # Use ThreadPoolExecutor to handle threads
        with ThreadPoolExecutor(max_workers=10000) as executor:
            # Store futures for the tasks
            futures = []

            for request_config in request_configs:
                if self.stop_event.is_set():
                    logger.info('Stopping task submission due to stop signal.')
                    break

                # Submit the task to the executor
                future = executor.submit(
                    self.send_requests,
                    [request_config],
                    llm_responses,
                    progress,
                    start_time,
                    num_requests,
                )
                futures.append(future)
                for t in executor._threads:
                    add_script_run_ctx(t)

                # Get wait time based on the distribution
                wait_time = self._get_wait_time()
                time.sleep(wait_time)

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    # Retrieve result if needed
                    future.result()
                except Exception as e:
                    logger.error(f'Error occurred in a thread: {e}')

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

        # Capture end time and notify user
        end_time = time.monotonic()
        logger.info('Tasks Executed!')
        logger.info(f'Benchmarking results obtained for model {self.model_name} queried with the {self.llm_api} API.')

        # Build a metrics summary for the results of the benchmarking run
        results = self.build_metrics_summary(
            metrics=[response.metrics for response in llm_responses],
            start_time=start_time,
            end_time=end_time,
        )

        # Construct metadata payload to be returned
        metadata = {
            'model': self.model_name,
            'qps': self.qps,
            'qps_distribution': self.qps_distribution,
            'results': results,
            'num_input_tokens': num_input_tokens,
            'num_output_tokens': num_output_tokens,
            'additional_sampling_params': sampling_params,
        }

        return metadata, llm_responses

    def build_request_configs(
        self,
        num_requests: int,
        input_token_count: int,
        output_token_count: int,
        sampling_params: Dict[str, Any],
    ) -> List[RequestConfig]:
        """Builds a list of request configuration objects used to send requests to the LLM. It iterates through the
        specified number of requests, builds an input prompt for each request, updates the sampling parameters with
        the maximum number of tokens to generate, and then creates the request configuration object. The request
        configurations are then returned as a list.

        Args:
            num_requests (int): The number of request configurations to build.
            input_token_count (int): The number of input tokens to use when building the prompt.
            output_token_count (int): The number of output tokens each request should return.
            sampling_params (dict): A dictionary of sampling parameters for the LLM.

        Returns:
            List[RequestConfig]: A list of request configurations, each containing the model name, prompt, sampling
            parameters, LLM API, generation mode, and number of concurrent requests.
        """
        # Empty list to be filled with valid request configs and then returned
        request_configs = []
        # Instantiate image variable
        image = None

        # Load prompts based on the model type
        if self.multimodal_image_size == 'na':
            # Read prompt for text-instruct model
            prompts_data = self.load_prompts(USER_PROMPT_TEXT_INSTRUCT_PATH)
            raw_prompt = prompts_data['default_prompt'][0]

        else:
            # Read prompt for vision-instruct model
            prompts_data = self.load_prompts(USER_PROMPT_VISION_INSTRUCT_PATH)
            raw_prompt = prompts_data['default_prompt'][0]
            image = self.get_image()

        # Build input prompt to be sent in LLM request
        prompt_tuple = self.build_prompt(raw_prompt, input_token_count)

        # Iterate through data points and build a request config for each
        for request_idx in range(num_requests):
            # Add generic max tokens parameter to `sampling_params` dictionary
            updated_sampling_params = {
                'max_tokens_to_generate': output_token_count,
            }
            updated_sampling_params.update(sampling_params)

            # Create request config object
            request_config = RequestConfig(
                request_idx=request_idx,
                model=self.model_name,
                prompt_tuple=prompt_tuple,
                image=image,
                sampling_params=updated_sampling_params,
                llm_api=self.llm_api,
                use_debugging_mode=self.use_debugging_mode,
                api_variables=self.api_variables,
                is_stream_mode=self.is_stream_mode,
            )

            request_configs.append(request_config)

        return request_configs
