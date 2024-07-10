import abc
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from llmperf import common_metrics
from llmperf.sambanova_client import llm_request
from llmperf.models import RequestConfig
from llmperf.utils import LLMPerfResults, flatten, get_tokenizer

MODEL_TYPE_IDENTIFIER = {"mistral": "mistral", "llama3": "llama-3"}
PROMPT_GENERATION_OFFSET = 3

class BasePerformanceEvaluator(abc.ABC):
    def __init__(
            self,
            model_name: str,
            results_dir: str,
            num_workers: int,
            user_metadata: Dict[str, Any] = {},
            llm_api: str = "sambastudio",
            generation_mode: str = "stream",
            timeout: int = 600,
    ):
        self.model_name = model_name
        self.results_dir = results_dir
        self.num_workers = num_workers
        self.user_metadata = user_metadata
        self.llm_api = llm_api
        self.generation_mode = generation_mode
        self.timeout = timeout
        self.tokenizer = get_tokenizer(self.model_name)

        # To be set upon saving of results
        self.summary_file_path = None
        self.individual_responses_file_path = None

    def get_token_length(self, input_text):
        return len(self.tokenizer.encode(input_text))
    
    @abc.abstractmethod
    def create_output_filename(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def run_benchmark(
        self, 
        sampling_params: Dict = {},
        *args, 
        **kwargs
    ):
        pass

    @abc.abstractmethod
    def get_token_throughput_latencies(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def build_request_configs(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def build_prompt(self, *args, **kwargs):
        pass

    def send_requests(
        self,
        request_config_batch: list,
        completed_requests: list,
        progress_bar: tqdm,
        start_time: float,
    ) -> None:
        """Sends multiple requests to LLM and collects results

        Args:
            request_configs_for_thread (list): list of request configs for LLM calls
            tokenizer (AutoTokenizer): HuggingFace tokenizer
            completed_requests (list): list of completed outputs from requests
            pbar (tqdm): progress bar
            start_time (float): start time of the process
            timeout_s (int): time out in seconds
        """
        for request_config in request_config_batch:
            if time.monotonic() - start_time >= self.timeout:
                break
            req_metrics = llm_request(request_config, self.tokenizer)
            completed_requests.extend([req_metrics[0]])
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
            print(f"Building Metrics Summary for metric: {metric}")
            metrics_summary[metric] = {}
            
            # Get flattened list from metric column in metrics df
            series = pd.Series(list(flatten(metrics_df[metric]))).dropna()

            # Generate statistics for specific metric
            quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
            quantiles_reformatted_keys = {}
            for quantile, value in quantiles.items():
                reformatted_key = f"p{int(quantile * 100)}"
                print(f"    {reformatted_key} = {value}")
                quantiles_reformatted_keys[reformatted_key] = value
            metrics_summary[metric]["quantiles"] = quantiles_reformatted_keys
            
            series_mean = series.mean()
            print(f"    mean = {series_mean}")
            metrics_summary[metric]["mean"] = series_mean
            
            series_min = series.min()
            print(f"    min = {series_min}")
            metrics_summary[metric]["min"] = series_min

            series_max = series.max()
            print(f"    max = {series_max}")
            metrics_summary[metric]["max"] = series_max
            
            series_std = series.std()
            print(f"    stddev = {series_std}")
            metrics_summary[metric]["stddev"] = series_std
        
        # Record number of requests started
        metrics_summary[common_metrics.NUM_REQ_STARTED] = len(metrics)
        
        # Record error count and rate
        error_codes = raw_df[common_metrics.ERROR_CODE].dropna()
        num_errors = len(error_codes)
        metrics_summary[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
        metrics_summary[common_metrics.NUM_ERRORS] = num_errors
        print(f"Number Of Errored Requests: {num_errors}")

        # Record specific error code frequencies
        error_code_frequency = dict(error_codes.value_counts())
        if num_errors:
            error_code_frequency = dict(error_codes.value_counts())
            print("Error Code Frequency")
            print(error_code_frequency)
        metrics_summary[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)
        
        # Record overall throughput
        overall_output_throughput = metrics_df[
            common_metrics.NUM_OUTPUT_TOKENS
        ].sum() / (end_time - start_time)
        print(f"Overall Output Throughput: {overall_output_throughput}")
        metrics_summary[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput
        
        # Record number of requests completed
        num_completed_requests = len(metrics_df)
        num_completed_requests_per_min = (
            num_completed_requests / (end_time - start_time) * 60
        )
        print(f"Number Of Completed Requests: {num_completed_requests}")
        print(f"Number Of Concurrent Workers: {self.num_workers}")
        print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")

        metrics_summary[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
        metrics_summary[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

        return metrics_summary
    
    def save_results(
        self,
        filename,
        summary,
        individual_responses
    ):
        """Save the performance evaluation results to a file.

        Args:
            filename (str): The base name of the file to save the results to.
            summary (dict): A dictionary containing the summary of the performance evaluation.
            individual_responses (list): A list of individual responses from the performance evaluation.

        Returns:
            None

        Raises:
            ValueError: If the results directory does not exist or is not a directory.
        """
        filename = re.sub(r"[^\w\d-]+", "-", filename)
        filename = re.sub(r"-{2,}", "-", filename)
        summary_filename = f"{filename}_summary"
        individual_responses_filename = f"{filename}_individual_responses"

        # Update to metadata.
        summary.update(self.user_metadata)

        results = LLMPerfResults(name=summary_filename, metadata=summary)
        results_dir = Path(self.results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")

        try:
            self.summary_file_path = f"{results_dir}/{summary_filename}.json"
            with open(self.summary_file_path, "w") as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e

        try:
            self.individual_responses_file_path = f"{results_dir}/{individual_responses_filename}.json"
            with open(self.individual_responses_file_path, "w") as f:
                json.dump(individual_responses, f, indent=4)
        except Exception as e:
            print(individual_responses)
            raise e


class CustomPerformanceEvaluator(BasePerformanceEvaluator):
    def __init__(
        self, 
        input_file_path: str,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.file_name = os.path.basename(input_file_path)
        self.dataset = self.read_dataset(input_file_path)
        self.prompt_key = list(self.dataset[0].keys())[0]
    
    @staticmethod
    def read_dataset(input_file_path: str):
        """Utility function for reading in the `.jsonl` file provided by the user for custom dataset evaluation.
        
        Args:
            input_file_path (str): The absolute file path of the input file provided by the user

        Returns:
            List[Dict]: A list of json objects (python dictionaries) containing the individual prompts the user wants to evaluate on
        """
        with open(input_file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return data

    def create_output_filename(self):
        """Utility for creating a unique filename for a custom benchmarking experiment with a dataset.

        Returns:
            str: Filename for the custom benchmark run.
        """
        return f"{self.model_name}_{self.file_name}_{self.num_workers}_{self.generation_mode}"

    def run_benchmark(
            self, 
            sampling_params: Dict[str, Any]
    ):
        """Run a benchmark test for the specified LLM using a custom dataset provided by the user.

        Args:
            sampling_params (str): The sampling parameters in JSON format.

        Raises:
            ValueError: If the number of input tokens is less than 40.

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
                individual_responses
            )

    def get_token_throughput_latencies(
            self,
            sampling_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], str, RequestConfig]]]:
        """This function is used to measure the token throughput and latencies.

        Args:
            sampling_params (Dict[str, Any]): A dictionary containing the parameters for sampling.

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, Any]]]: A tuple containing metadata and completed requests.

        Raises:
            Exception: If an unexpected error happens when executing requests.

        Note:
            This function uses threading to send requests concurrently. It splits the total request count evenly among the threads.
            If there is a remainder, it assigns one extra request to the first threads.
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
        completed_requests = []
        progress_bar = tqdm(total=total_request_count, desc="Running Requests")

        for request_config_batch in request_config_batches:
            thread = threading.Thread(
                target=self.send_requests,
                args=(
                    request_config_batch,
                    completed_requests,
                    progress_bar,
                    start_time
                )
            )
            threads.append(thread)
            thread.start()
        

        for thread in threads:
            thread.join()

        if completed_requests[0]["error_code"]:
            raise Exception(
                f"Unexpected error happened when executing requests: {completed_requests[0]['error_code']}. Additional message: {completed_requests[0]['error_msg']}"
            )
        
        end_time = time.monotonic()
        print("Tasks Executed!")

        print(f"\Results for token benchmark for {self.model_name} queried with the {self.llm_api} api.\n")
        results = self.build_metrics_summary(
            completed_requests,
            start_time,
            end_time,
        )

        metadata = {
            "model": self.model_name,
            "num_concurrent_workers": self.num_workers,
            "results": results,
            "request_count": len(self.dataset),
            "sampling_params": sampling_params,
        }

        metadata["results"]
        
        return metadata, completed_requests

    def build_request_configs(
            self,
            sampling_params: Dict[str, Any]
    ) -> List[RequestConfig]:
        """Builds a list of request configs for the LLM API. This method iterates through the provided dataset and builds a 
        RequestConfig object for each data point. The RequestConfig object contains the necessary information to send a 
        request to the LLM API, including the model name, prompt, sampling parameters, LLM API endpoint, generation mode, 
        and number of concurrent workers. The method returns a list of these RequestConfig objects.

        Args:
            sampling_params (Dict[str, Any]): A dictionary of sampling parameters to be passed into the RequestConfig constructor.

        Returns:
            List[RequestConfig]: A list of RequestConfig objects, each representing a request to the LLM API.
        """
        # Empty list to be filled with valid request configs and then returned
        request_configs = []
        
        # Iterate through data points and build a request config for each
        for data_point in self.dataset:
            
            # Apply prompt templating to get final prompt to send to LLM API along with tokenized prompt length
            prompt_tuple = self.build_prompt(
                raw_prompt=data_point[self.prompt_key]
            )
            
            request_config = RequestConfig(
                model=self.model_name,
                prompt_tuple=prompt_tuple,
                sampling_params=sampling_params,
                llm_api=self.llm_api,
                mode=self.generation_mode,
                num_concurrent_workers=self.num_workers,
            )

            request_configs.append(request_config)
        
        return request_configs
    

    def build_prompt(
        self, 
        raw_prompt: str
    ) -> Tuple[str, int]:
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
        # Specific prompt templating for mistral models
        if MODEL_TYPE_IDENTIFIER["mistral"] in self.model_name.lower():
            prompt = "[INST]" + raw_prompt + "[/INST]"
        
        # Specific prompt templating for Llama-3 models
        elif MODEL_TYPE_IDENTIFIER["llama3"] in self.model_name.lower():
            system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant that provides concise and helpful assistance on a variety of subjects<|eot_id|>"
            
            prompt = (
                system_prompt
                + "<|start_header_id|>user<|end_header_id|>"
                + raw_prompt
                + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )

        # Prompt templating for Llama-2 and all other models
        else:
            system_prompt = "[INST]<<SYS>>You are a helpful assistant that provides concise and helpful assistance on a variety of subjects<</SYS>>"
            prompt = system_prompt + raw_prompt + "[/INST]"

        return (prompt, self.get_token_length(prompt))


class SyntheticPerformanceEvaluator(BasePerformanceEvaluator):
    
    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def create_output_filename(
        self,
        num_input_tokens: int,
        num_output_tokens: int
    ) -> str:
        """Utility for creating a unique filename for a synthetic benchmarking experiment given user specified params.

        Returns:
            str: Filename for the synthetic benchmark run.
        """
        return f"{self.model_name}_{num_input_tokens}_{num_output_tokens}_{self.num_workers}_{self.generation_mode}"

    def run_benchmark(
            self,
            num_input_tokens: int,
            num_output_tokens: int,
            num_requests: int,
            sampling_params: Dict[str, Any],
    ):
        """Run a benchmark test for the specified LLM using synthetically generated data.

        Args:
            num_input_tokens (int): The number of input tokens to be sent.
            num_output_tokens (int): The number of output tokens to be received.
            num_requests (int): The number of requests to be made.
            num_workers (int): The number of workers to be used.
            sampling_params (str): The sampling parameters in JSON format.

        Raises:
            ValueError: If the number of input tokens is less than 40.

        Returns:
            None
        """
        if num_input_tokens < 40:
            raise ValueError(
                "The minimum number of input tokens that will be sent is 40"
                " because of the prompting logic right now"
            )

        # Calculate performance metrics individually and summary
        summary, individual_responses = self.get_token_throughput_latencies(
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            num_requests=num_requests,
            sampling_params=sampling_params,
        )
        
        if self.results_dir:
            filename = self.create_output_filename(
                num_input_tokens,
                num_output_tokens
            )
            self.save_results(
                filename,
                summary,
                individual_responses
            )

    def get_token_throughput_latencies(
        self, 
        num_input_tokens: int, 
        num_output_tokens: int, 
        num_requests, 
        sampling_params
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
        request_configs = self.build_request_configs(
            num_requests,
            num_input_tokens,
            num_output_tokens,
            sampling_params
        )
        
        # Get the request counts in order to place them into threads to be executed in batches
        total_request_count = len(request_configs)
        requests_per_thread = total_request_count // self.num_workers
        remainder = total_request_count % self.num_workers
        
        # Set up empty batch array and index for a sliding window of request selection
        request_config_batches = []
        idx = 0

        # Create batches of requests for each worker
        for worker in range(self.num_workers):
            num_requests_for_thread = requests_per_thread + (1 if worker < remainder else 0)
            request_config_batch = request_configs[idx : idx + num_requests_for_thread].copy()
            idx = idx + num_requests_for_thread
            request_config_batches.append(request_config_batch)
        
        # Create empty `threads` and `completed_requests` arrays to be populated with execution threads and 
        # completed requests respectively
        threads = []
        completed_requests = []
        progress_bar = tqdm(total=total_request_count, desc="Running Requests")
        
        # Send request threads and add to the threads array
        for request_config_batch in request_config_batches:
            thread = threading.Thread(
                target=self.send_requests,
                args=(
                    request_config_batch,
                    completed_requests,
                    progress_bar,
                    start_time
                )
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Error handling
        if completed_requests[0]["error_code"]:
            raise Exception(
                f"Unexpected error happened when executing requests: {completed_requests[0]['error_code']}. Additional message: {completed_requests[0]['error_msg']}"
            )
        
        # Capture end time and notify user
        end_time = time.monotonic()
        print("Tasks Executed!")
        print(f"\Results for token benchmark for {self.model_name} queried with the {self.llm_api} api.\n")

        # Build a metrics summary for the results of the benchmarking run
        results = self.build_metrics_summary(
            completed_requests,
            start_time,
            end_time,
        )
        
        # Construct metadata payload to be returned
        metadata = {
            "model": self.model_name,
            "num_concurrent_workers": self.num_workers,
            "results": results,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "additional_sampling_params": sampling_params,
        }
        
        return metadata, completed_requests


    def build_request_configs(
        self, 
        num_requests: int, 
        input_token_count: int,
        output_token_count: int,
        sampling_params: dict, 
    )-> List[RequestConfig]:
        """Builds a list of request configuration objects used to send requests to the LLM. It iterates through the specified 
        number of requests, builds an input prompt for each request, updates the sampling parameters with the maximum number 
        of tokens to generate, and then creates the request configuration object. The request configurations are then returned 
        as a list.

        Args:
            num_requests (int): The number of request configurations to build.
            input_token_count (int): The number of input tokens to use when building the prompt.
            output_token_count (int): The number of output tokens each request should return.
            sampling_params (dict): A dictionary of sampling parameters for the LLM.

        Returns:
            List[RequestConfig]: A list of request configurations, each containing the model name, prompt, sampling parameters, 
            LLM API, generation mode, and number of concurrent workers.
        """
        # Empty list to be filled with valid request configs and then returned
        request_configs = []

        # Iterate through data points and build a request config for each
        for _ in range(num_requests):
            
            # Build input prompt to be sent in LLM request
            prompt_tuple = self.build_prompt(
                input_token_count, 
                output_token_count
            )
            
            # Add max_tokens_to_generate to `sampling_params` dictionary
            updated_sampling_params = {
                "max_tokens_to_generate": output_token_count,
            }
            updated_sampling_params.update(sampling_params)
            
            # Create request config object
            request_config = RequestConfig(
                model=self.model_name,
                prompt_tuple=prompt_tuple,
                sampling_params=updated_sampling_params,
                llm_api=self.llm_api,
                mode=self.generation_mode,
                num_concurrent_workers=self.num_workers,
            )

            request_configs.append(request_config)

        return request_configs

    def build_prompt(
        self, 
        num_input_tokens: int, 
        num_output_tokens: int
    ) -> Tuple[str, int]:
        """Synthesizes an input prompt for the LLM to be queried. This prompt is created by repeating a prompt_template 
        multiple times to reach a user set input_token_count. Depending on the LLM being queried, there may be an added
        system prompt.

        The method generates a prompt based on the model type and the provided input and output token counts.
        For Mistral models, a fixed prompt template is used.
        For Llama3 models, a system prompt is generated and the user input is appended.
        For other models, a system prompt is generated and the user input is appended.

        Args:
            num_input_tokens (int): The user specified length of the input prompt.
            num_output_tokens (int): The number of tokens for the model to generate.

        Returns:
            Tuple[str, int]: A tuple containing the generated prompt and its length in tokens.
        """
        prompt_template = "Create a movie script of the whole Star Wars movie with details. Describe how every character felt, \
                           include environment details and onomatopoeias."
        
         # Prompt for Mistral models
        if MODEL_TYPE_IDENTIFIER["mistral"] in self.model_name.lower():

            prompt_content = self.generate_prompt_content(
                prompt_template,
                num_input_tokens,
            )

            full_input_prompt = "[INST]" + prompt_content + "[/INST]"

        # Prompt for Llama3 models
        elif MODEL_TYPE_IDENTIFIER["llama3"] in self.model_name.lower():

            system_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant that generates movie scripts with at least {num_output_tokens} words<|eot_id|>"
            prompt_content_length = num_input_tokens - self.get_token_length(system_prompt)

            prompt_content = self.generate_prompt_content(
                prompt_template,
                prompt_content_length
            )

            full_input_prompt = (
                system_prompt
                + "<|start_header_id|>user<|end_header_id|>"
                + prompt_content
                + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
        
        # Prompt for Llama2 and other models
        else:
            
            system_prompt = f"[INST]<<SYS>>You are a helpful assistant that generates movie scripts with at least {num_output_tokens} words<</SYS>>"
            prompt_content_length = num_input_tokens - self.get_token_length(system_prompt)

            prompt_content = self.generate_prompt_content(
                prompt_template,
                prompt_content_length
            )

            full_input_prompt = system_prompt + prompt_content + "[/INST]"

        return (full_input_prompt, self.get_token_length(full_input_prompt))

    
    def generate_prompt_content(
        self,
        prompt_template: int,
        input_prompt_length: int
    ) -> str:
        """Generate prompt content based on the given prompt template and input prompt length.

        This function works by first determining the length of the prompt template and the total number of tokens needed to generate 
        from the prompt template. It then generates the prompt content by repeating the prompt template n times, where n is the 
        number of times the prompt template fits into the total token count. After that, it adds on the first m tokens of the prompt 
        template to the prompt content generated above until the total token count is reached, where m is the number of tokens that 
        didn't fit in the full repetitions of the prompt template.

        Args:
            prompt_template (int): The template from which to generate the prompt.
            input_prompt_length (int): The user selected length of the input prompt.

        Returns:
            str: The generated prompt content.
        """
        # Get the length of the prompt template
        prompt_template_length = self.get_token_length(prompt_template)

        # Determine the number of tokens needed to generate from the prompt template
        total_token_count = input_prompt_length + prompt_template_length * PROMPT_GENERATION_OFFSET
        
        # Generate prompt contnet by repeating `prompt_template` n times, where n is the number of times the prompt template fits 
        # into  the total token count calculated above
        prompt_content = "".join(
            [prompt_template * (total_token_count // prompt_template_length)]
        )
        
        # Add on the first m tokens of the prompt template to the `prompt_content` generated above up until the total token count 
        # is reached, where m is calculated by using the modulo operator to see how many tokens didn't fit in the repetitions above.
        prompt_content += prompt_template[: (total_token_count % prompt_template_length)]

        return prompt_content
