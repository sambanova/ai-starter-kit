import abc
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

file_location = Path(__file__).parent.resolve()

import logging

import llmperf.utils as utils
import pandas as pd
import transformers
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from llmperf import common_metrics
from llmperf.models import LLMResponse, RequestConfig
from llmperf.sambanova_client import llm_request
from llmperf.utils import LLMPerfResults, flatten, get_tokenizer
from stqdm import stqdm
from streamlit.runtime.scriptrunner import add_script_run_ctx
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()
load_dotenv('../.env', override=True)

SYSTEM_PROMPT_PATH = os.path.join(file_location, '../prompts/system-prompt_template.yaml')
USER_PROMPT_PATH = os.path.join(file_location, '../prompts/user-prompt_template.yaml')


class BasePerformanceEvaluator(abc.ABC):
    def __init__(
        self,
        model_name: str,
        results_dir: str,
        num_workers: int,
        user_metadata: Dict[str, Any] = {},
        llm_api: str = 'sambastudio',
        is_stream_mode: bool = True,
        timeout: int = 600,
    ) -> None:
        self.model_name = model_name
        self.results_dir = results_dir
        self.num_workers = num_workers
        self.user_metadata = user_metadata
        self.llm_api = llm_api
        self.is_stream_mode = is_stream_mode
        self.timeout = timeout
        self.tokenizer = get_tokenizer(self.model_name)

        # To be set upon saving of results
        self.summary_file_path = None
        self.individual_responses_file_path = None

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
    def run_benchmark(self, sampling_params: Dict = {}, *args: Any, **kwargs: Any) -> None:
        pass

    @abc.abstractmethod
    def get_token_throughput_latencies(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], str, RequestConfig]]]:
        pass

    @abc.abstractmethod
    def build_request_configs(self, *args: Any, **kwargs: Any) -> List[RequestConfig]:
        pass

    @abc.abstractmethod
    def build_prompt(self, *args: Any, **kwargs: Any) -> Tuple[str, int]:
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
        adjusted_text = self.tokenizer.convert_tokens_to_string(tokens)

        # Validate token count
        assert len(self.tokenizer.tokenize(adjusted_text)) == (target_token_count - 1), 'Token count mismatch!'

        return adjusted_text

    def send_requests(
        self,
        request_config_batch: list,
        completed_requests: list,
        progress_bar: tqdm,
        start_time: float,
    ) -> None:
        """Sends multiple requests to LLM and collects results

        Args:
            request_config_batch (list): list of request configs for LLM calls
            completed_requests (list): list of completed outputs from requests
            progress_bar (tqdm): progress bar
            start_time (float): start time of the process
        """
        for request_config in request_config_batch:
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
            progress_bar.update(1)

    def build_metrics_summary(
        self,
        metrics: List[Dict[str, Any]],
        start_time: time,
        end_time: time,
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
        metrics_summary = {}

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
            logger.info(f'Building Metrics Summary for metric: {metric}')
            metrics_summary[metric] = {}

            # Get flattened list from metric column in metrics df
            series = pd.Series(list(flatten(metrics_df[metric]))).dropna()

            # Generate statistics for specific metric
            quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(4).to_dict()
            quantiles_reformatted_keys = {}
            for quantile, value in quantiles.items():
                reformatted_key = f'p{int(quantile * 100)}'
                logger.info(f'    {reformatted_key} = {value}')
                quantiles_reformatted_keys[reformatted_key] = value
            metrics_summary[metric]['quantiles'] = quantiles_reformatted_keys

            series_mean = round(series.mean(), 4)
            logger.info(f'    mean = {series_mean}')
            metrics_summary[metric]['mean'] = series_mean

            series_min = round(series.min(), 4)
            logger.info(f'    min = {series_min}')
            metrics_summary[metric]['min'] = series_min

            series_max = round(series.max(), 4)
            logger.info(f'    max = {series_max}')
            metrics_summary[metric]['max'] = series_max

            series_std = round(series.std(), 4)
            logger.info(f'    stddev = {series_std}')
            metrics_summary[metric]['stddev'] = series_std

        # Record number of requests started
        metrics_summary[common_metrics.NUM_REQ_STARTED] = len(metrics)

        # Record error count and rate
        error_codes = raw_df[common_metrics.ERROR_CODE].dropna()
        num_errors = len(error_codes)
        metrics_summary[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
        metrics_summary[common_metrics.NUM_ERRORS] = num_errors
        logger.info(f'Number Of Errored Requests: {num_errors}')

        # Record specific error code frequencies
        error_code_frequency = dict(error_codes.value_counts())
        if num_errors:
            error_code_frequency = dict(error_codes.value_counts())
            logger.error('Error Code Frequency')
            logger.error(error_code_frequency)
        metrics_summary[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)

        # Record overall throughput
        overall_output_throughput = round(
            metrics_df[common_metrics.NUM_OUTPUT_TOKENS].sum() / (end_time - start_time),
            4,
        )
        logger.info(f'Overall Output Throughput: {overall_output_throughput}')
        metrics_summary[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

        # Record number of requests completed
        num_completed_requests = len(metrics_df)
        num_completed_requests_per_min = round(num_completed_requests / (end_time - start_time) * 60, 4)
        logger.info(f'Number Of Completed Requests: {num_completed_requests}')
        logger.info(f'Number Of Concurrent Workers: {self.num_workers}')
        logger.info(f'Completed Requests Per Minute: {num_completed_requests_per_min}')

        metrics_summary[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
        metrics_summary[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

        return metrics_summary

    def save_results(
        self,
        filename: str,
        summary: Dict[str, Any],
        individual_responses: List[LLMResponse],
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

            response_metrics = [response.metrics for response in individual_responses]
            with open(self.individual_responses_file_path, 'w') as f:
                json.dump(response_metrics, f, indent=4)
        except Exception as e:
            logger.error(individual_responses)
            raise e


class CustomPerformanceEvaluator(BasePerformanceEvaluator):
    def __init__(self, input_file_path: str, save_response_texts: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.file_name = os.path.basename(input_file_path)
        self.dataset = self.read_dataset(input_file_path)
        self.prompt_key = list(self.dataset[0].keys())[0]
        self.save_response_texts = save_response_texts

    @staticmethod
    def read_dataset(input_file_path: str) -> List[Dict]:
        """Utility function for reading in the `.jsonl` file provided by the user for custom dataset evaluation.

        Args:
            input_file_path (str): The absolute file path of the input file provided by the user

        Returns:
            List[Dict]: A list of json objects (python dictionaries) containing the individual prompts the user wants
            to evaluate on
        """
        with open(input_file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return data

    def create_output_filename(self) -> str:
        """Utility for creating a unique filename for a custom benchmarking experiment with a dataset.

        Returns:
            str: Filename for the custom benchmark run.
        """
        generation_mode = ''
        if self.is_stream_mode:
            generation_mode = 'stream'

        output_file_name = f'{self.model_name}_{self.file_name}_{self.num_workers}_{generation_mode}'
        return self.sanitize_file_prefix(output_file_name)

    def save_results(
        self,
        filename: str,
        summary: Dict[str, Any],
        individual_responses: List[LLMResponse],
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
                        output_json = {
                            'prompt': response.request_config.prompt_tuple[0],
                            'completion': str(response.response_text),
                        }
                        f.write(json.dumps(output_json))
                        f.write('\n')
            except Exception as e:
                logger.error('ERROR SAVING LLM OUTPUTS')
                raise e

    def run_benchmark(self, sampling_params: Dict[str, Any]) -> None:
        """Run a benchmark test for the specified LLM using a custom dataset provided by the user.

        Args:
            sampling_params (Dict[str, Any]): The sampling parameters in JSON format.

        Returns:
            None
        """
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

    def get_token_throughput_latencies(
        self, sampling_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], str, RequestConfig]]]:
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
        random.seed(11111)
        start_time = time.monotonic()

        request_configs = self.build_request_configs(
            sampling_params,
        )

        # Get batch size details
        total_request_count = len(request_configs)
        requests_per_thread = total_request_count // self.num_workers
        remainder = total_request_count % self.num_workers

        request_config_batches = []
        idx = 0
        for worker in range(self.num_workers):
            num_requests_for_thread = requests_per_thread + (1 if worker < remainder else 0)
            request_config_batch = request_configs[idx : idx + num_requests_for_thread].copy()
            idx = idx + num_requests_for_thread
            request_config_batches.append(request_config_batch)

        threads = []
        llm_responses: List[LLMResponse] = []
        progress_bar = stqdm(total=total_request_count, desc='Running Requests', mininterval=1)
        # progress_bar = tqdm(total=total_request_count, desc="Running Requests")

        for request_config_batch in request_config_batches:
            thread = threading.Thread(
                target=self.send_requests,
                args=(
                    request_config_batch,
                    llm_responses,
                    progress_bar,
                    start_time,
                ),
            )
            threads.append(thread)
            add_script_run_ctx(thread)  # Give Streamlit context to thread
            thread.start()

        for thread in threads:
            add_script_run_ctx(thread)
            thread.join()

        if llm_responses[0].metrics[common_metrics.ERROR_CODE]:
            raise Exception(
                f"""Unexpected error happened when executing requests: {llm_responses[0].metrics['error_code']}.
                  Additional message: {llm_responses[0].metrics['error_msg']}"""
            )

        end_time = time.monotonic()
        logger.info('Tasks Executed!')
        logger.info(f'Results for token benchmark for {self.model_name} queried with the {self.llm_api} api.')
        results = self.build_metrics_summary(
            metrics=[response.metrics for response in llm_responses],
            start_time=start_time,
            end_time=end_time,
        )

        metadata = {
            'model': self.model_name,
            'num_concurrent_workers': self.num_workers,
            'results': results,
            'request_count': len(self.dataset),
            'sampling_params': sampling_params,
        }

        return metadata, llm_responses

    def build_request_configs(self, sampling_params: Dict[str, Any]) -> List[RequestConfig]:
        """Builds a list of request configs for the LLM API. This method iterates through the provided dataset and
        builds a RequestConfig object for each data point. The RequestConfig object contains the necessary information
        to send a request to the LLM API, including the model name, prompt, sampling parameters, LLM API endpoint,
        generation mode, and number of concurrent workers. The method returns a list of these RequestConfig objects.

        Args:
            sampling_params (Dict[str, Any]): A dictionary of sampling parameters to be passed into the RequestConfig
            constructor.

        Returns:
            List[RequestConfig]: A list of RequestConfig objects, each representing a request to the LLM API.
        """
        # Empty list to be filled with valid request configs and then returned
        request_configs = []

        # Iterate through data points and build a request config for each
        for data_point in self.dataset:
            # Apply prompt templating to get final prompt to send to LLM API along with tokenized prompt length
            prompt_tuple = self.build_prompt(raw_prompt=data_point[self.prompt_key])

            request_config = RequestConfig(
                model=self.model_name,
                prompt_tuple=prompt_tuple,
                sampling_params=sampling_params,
                llm_api=self.llm_api,
                is_stream_mode=self.is_stream_mode,
                num_concurrent_workers=self.num_workers,
            )

            request_configs.append(request_config)

        return request_configs

    def build_prompt(self, raw_prompt: str) -> Tuple[str, int]:
        """Builds an input prompt from the given raw prompt by applying prompt templating based on the model type.

        Args:
        - raw_prompt (str): The raw input prompt to be used in building a processed input prompt.

        Returns:
        - A tuple containing the processed prompt and the token length of the prompt.

        Description:
        This method builds a prompt for the given raw prompt based on the model type.
        It checks if the model type is'mistral' or 'llama3' and applies specific templating for those models.
        For other models, it applies a default templating.
        The method returns a tuple containing the processed prompt and the token length of the prompt.
        """

        sys_prompt_template = (
            'You are a helpful assistant that provides concise and helpful assistance on a variety of subjects'
        )

        # Specific prompt templating for mistral models
        if utils.MODEL_TYPE_IDENTIFIER['mistral'] in self.model_name.lower().replace('-', ''):
            prompt = '[INST]' + raw_prompt + '[/INST]'

        # Specific prompt templating for Llama-3 models
        elif utils.MODEL_TYPE_IDENTIFIER['llama3'] in self.model_name.lower().replace('-', ''):
            system_prompt = (
                f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt_template}<|eot_id|>'
            )

            prompt = (
                system_prompt
                + '<|start_header_id|>user<|end_header_id|>'
                + raw_prompt
                + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>Answer:'
            )

        # Specific prompt templating for Llama-2 models
        elif utils.MODEL_TYPE_IDENTIFIER['llama2'] in self.model_name.lower().replace('-', ''):
            system_prompt = f'[INST]<<SYS>>{sys_prompt_template}<</SYS>>'
            prompt = system_prompt + raw_prompt + '[/INST]'

        # Prompt templating for other models (Deepseek, Solar, Eeve)
        else:
            system_prompt = f'{sys_prompt_template}'
            prompt = system_prompt + raw_prompt

        return (prompt, self.get_token_length(prompt))


class SyntheticPerformanceEvaluator(BasePerformanceEvaluator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def create_output_filename(self, num_input_tokens: int, num_output_tokens: int) -> str:
        """Utility for creating a unique filename for a synthetic benchmarking experiment given user specified params.

        Returns:
            str: Filename for the synthetic benchmark run.
        """
        generation_mode = ''
        if self.is_stream_mode:
            generation_mode = 'stream'

        output_file_name = (
            f'{self.model_name}_{num_input_tokens}_{num_output_tokens}_{self.num_workers}_{generation_mode}'
        )
        return self.sanitize_file_prefix(output_file_name)

    def run_benchmark(
        self,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        sampling_params: Dict[str, Any],
    ) -> tuple:
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
        if num_input_tokens < 40:
            raise ValueError(
                'The minimum number of input tokens that will be sent is 40' ' because of the prompting logic right now'
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

    def get_token_throughput_latencies(
        self,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        sampling_params: dict,
    ) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], str, RequestConfig]]]:
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
                            including the model name, number of concurrent workers,
                            results, number of input tokens, number of output tokens,
                            and additional sampling parameters.
            completed_requests (list): A list of completed requests.

        Raises:
            Exception: If an unexpected error occurs during the execution of requests.
        """
        random.seed(11111)
        start_time = time.monotonic()

        # Build the request config objects that are to be sent to the LLM API endpoint
        request_configs = self.build_request_configs(num_requests, num_input_tokens, num_output_tokens, sampling_params)

        # Get the request counts in order to place them into threads to be executed in batches
        total_request_count = len(request_configs)
        requests_per_thread = (total_request_count) // self.num_workers
        remainder = (total_request_count) % self.num_workers

        # Set up empty batch array and index for a sliding window of request selection
        request_config_batches = []
        idx = 0

        # Create batches of requests for each worker
        for worker in range(self.num_workers):
            num_requests_for_thread = requests_per_thread + (1 if worker < remainder else 0)
            request_config_batch = request_configs[idx : idx + num_requests_for_thread].copy()
            idx += num_requests_for_thread
            request_config_batches.append(request_config_batch)

        # Create empty `threads` and `completed_requests` arrays to be populated with execution threads and
        # completed requests respectively
        threads = []
        llm_responses = []
        progress_bar = stqdm(total=total_request_count, desc='Running Requests', mininterval=1)
        # progress_bar = tqdm(total=total_request_count, desc="Running Requests")

        # Send request threads and add to the threads array
        for request_config_batch in request_config_batches:
            thread = threading.Thread(
                target=self.send_requests,
                args=(
                    request_config_batch,
                    llm_responses,
                    progress_bar,
                    start_time,
                ),
            )
            threads.append(thread)
            add_script_run_ctx(thread)  # Add Streamlit context to thread
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            add_script_run_ctx(thread)
            thread.join()

        # Error handling
        if llm_responses[0].metrics['error_code']:
            raise Exception(
                f"""Unexpected error happened when executing requests: {llm_responses[0].metrics['error_code']}.
                  Additional message: {llm_responses[0].metrics['error_msg']}"""
            )

        # Capture end time and notify user
        end_time = time.monotonic()
        logger.info('Tasks Executed!')
        logger.info(f'Results for token benchmark for {self.model_name} queried with the {self.llm_api} api.')

        # Build a metrics summary for the results of the benchmarking run
        results = self.build_metrics_summary(
            metrics=[response.metrics for response in llm_responses],
            start_time=start_time,
            end_time=end_time,
        )

        # Construct metadata payload to be returned
        metadata = {
            'model': self.model_name,
            'num_concurrent_workers': self.num_workers,
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
        sampling_params: dict,
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
            parameters, LLM API, generation mode, and number of concurrent workers.
        """
        # Empty list to be filled with valid request configs and then returned
        request_configs = []

        # Iterate through data points and build a request config for each
        for _ in range(num_requests):
            # Build input prompt to be sent in LLM request
            prompt_tuple = self.build_prompt(input_token_count)

            # Add max_tokens_to_generate to `sampling_params` dictionary
            if self.llm_api == 'sncloud':
                updated_sampling_params = {
                    'max_tokens': output_token_count,
                }
            else:
                updated_sampling_params = {
                    'max_tokens_to_generate': output_token_count,
                }
            updated_sampling_params.update(sampling_params)

            # Create request config object
            request_config = RequestConfig(
                model=self.model_name,
                prompt_tuple=prompt_tuple,
                sampling_params=updated_sampling_params,
                llm_api=self.llm_api,
                is_stream_mode=self.is_stream_mode,
                num_concurrent_workers=self.num_workers,
            )

            request_configs.append(request_config)

        return request_configs

    def build_prompt(self, num_input_tokens: int) -> Tuple[str, int]:
        """Synthesizes an input prompt for the LLM to be queried. This prompt is created by repeating a prompt_template
        multiple times to reach a user set input_token_count.

        Args:
            num_input_tokens (int): The user specified length of the input prompt.

        Returns:
            Tuple[str, int]: A tuple containing the generated prompt and its length in tokens.
        """

        # Load from prompt files
        prompt_template = yaml.safe_load(PromptTemplate.from_file(USER_PROMPT_PATH).template)['template']

        #  Adjust prompt according to desired input tokens
        full_input_prompt = self.adjust_to_exact_tokens(prompt_template, num_input_tokens)

        return (full_input_prompt, self.get_token_length(full_input_prompt))
