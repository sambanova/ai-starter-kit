import argparse
from collections.abc import Iterable
import json
import os
from pathlib import Path
import re
import time
import random
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from typing import Generator
import threading

import transformers
from llmperf import common_metrics
from llmperf.clients.sambanova_client import llm_request

# from llmperf.common import SUPPORTED_APIS, construct_clients
from llmperf.models import RequestConfig

# from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    get_tokenizer,
    build_prompt,
    LLMPerfResults,
    sample_random_positive_int,
)
from transformers import LlamaTokenizerFast
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def send_requests(
    model: str,
    tokenizer: LlamaTokenizerFast,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    additional_sampling_params: dict,
    num_requests_for_thread: int,
    num_concurrent_workers: int,
    llm_api: str,
    mode: str,
    completed_requests: list,
    pbar: tqdm,
    start_time: float,
    timeout_s: int,
) -> None:
    """Sends multiple requests to LLM and collects results

    Args:
        model (str): model name
        tokenizer (LlamaTokenizerFast): HuggingFace tokenizer
        mean_output_tokens (int): mean output tokens
        stddev_output_tokens (int): output tokens standard deviation
        mean_input_tokens (int): mean input tokens
        stddev_input_tokens (int): input tokens standard deviation
        additional_sampling_params (dict): additional sampling parameters
        num_requests_for_thread (int): number of requests per worker
        num_concurrent_workers (int): number of concurrent workers
        llm_api (str): LLM API type
        mode (str): mode of process
        completed_requests (list): list of completed outputs from requests
        pbar (tqdm): progress bar
        start_time (float): start time of the process
        timeout_s (int): time out in seconds
    """

    for _ in range(num_requests_for_thread):
        if time.monotonic() - start_time >= timeout_s:
            break

        # Set request config
        num_output_tokens = sample_random_positive_int(
            mean_output_tokens, stddev_output_tokens
        )

        prompt = build_prompt(
            model_name=model,
            prompt_tokens_mean=mean_input_tokens,
            prompt_tokens_stddev=stddev_input_tokens,
            num_output_tokens=num_output_tokens,
        )

        default_sampling_params = {
            "max_tokens_to_generate": num_output_tokens,
        }

        default_sampling_params.update(additional_sampling_params)
        request_config = RequestConfig(
            model=model,
            prompt=prompt,
            sampling_params=default_sampling_params,
            llm_api=llm_api,
            mode=mode,
            num_concurrent_workers=num_concurrent_workers,
        )

        req_metrics = llm_request(request_config, tokenizer)
        completed_requests.extend([req_metrics[0]])
        pbar.update(1)


def get_token_throughput_latencies(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    timeout_s=90,
    num_concurrent_workers: int = 1,
    max_num_completed_requests: int = 32,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    llm_api="sambastudio",
    mode="stream",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model (str): The name of the model to query.
        mean_input_tokens (int): The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens (int): The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens (int): The mean number of tokens to generate per request.
        stddev_output_tokens (int): The standard deviation of the number of tokens to generate per request.
        timeout_s (int): The amount of time to run the test for before reporting results. Defaults to 90.
        num_concurrent_workers (int): The number of concurrent workers to make. Increase
            this to increase the amount of load and vice versa. Defaults to 1.
        max_num_completed_requests (int): The maximum number of completed requests. Defaults to 32.
        additional_sampling_params (Optional[Dict[str, Any]]): Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions. Defaults to None.
        llm_api (str): The name of the llm api to use. Static for now. Defaults to "sambastudio".
        mode (str): mode of the API. Either "stream" or "batch". Defaults to "stream".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    random.seed(11111)

    if not additional_sampling_params:
        additional_sampling_params = {}

    tokenizer = get_tokenizer(model)

    start_time = time.monotonic()

    # Calculate requests per thread

    requests_per_thread = max_num_completed_requests // num_concurrent_workers
    remainder = max_num_completed_requests % num_concurrent_workers

    # Create threads
    threads = []
    completed_requests = []
    pbar = tqdm(total=max_num_completed_requests, desc="Running requests")
    for i in range(num_concurrent_workers):

        print(f"Executing Thread {i}")

        num_requests_for_thread = requests_per_thread + (1 if i < remainder else 0)
        thread = threading.Thread(
            target=send_requests,
            args=(
                model,
                tokenizer,
                mean_output_tokens,
                stddev_output_tokens,
                mean_input_tokens,
                stddev_input_tokens,
                additional_sampling_params,
                num_requests_for_thread,
                num_concurrent_workers,
                llm_api,
                mode,
                completed_requests,
                pbar,
                start_time,
                timeout_s,
            ),
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    pbar.close()

    if completed_requests[0]["error_code"]:
        raise Exception(
            f"Unexpected error happened when executing requests: {completed_requests[0]['error_code']}. Additional message: {completed_requests[0]['error_msg']}"
        )

    end_time = time.monotonic()
    print("Tasks executed!!!")

    print(f"\Results for token benchmark for {model} queried with the {llm_api} api.\n")
    ret = metrics_summary(
        completed_requests, start_time, end_time, num_concurrent_workers
    )

    metadata = {
        "model": model,
        "mean_input_tokens": mean_input_tokens,
        "stddev_input_tokens": stddev_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
        "num_concurrent_workers": num_concurrent_workers,
        "additional_sampling_params": additional_sampling_params,
    }

    metadata["results"] = ret

    return metadata, completed_requests


def metrics_summary(
    metrics: List[Dict[str, Any]],
    start_time: int,
    end_time: int,
    num_concurrent_workers: int,
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics (List[Dict[str, Any]]): The metrics to summarize.
        start_time (int): The time the test started.
        end_time (int): The time the test ended.
        num_concurrent_workers (int): number of concurrent workers

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}

    def flatten(item):
        for sub_item in item:
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]

    for key in [
        common_metrics.TTFT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_OUTPUT_THROUGHPUT,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS,
    ]:
        print(key)
        ret[key] = {}
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            print(f"    {reformatted_key} = {value}")
            quantiles_reformatted_keys[reformatted_key] = value
        ret[key]["quantiles"] = quantiles_reformatted_keys
        mean = series.mean()
        print(f"    mean = {mean}")
        ret[key]["mean"] = mean
        print(f"    min = {series.min()}")
        ret[key]["min"] = series.min()
        print(f"    max = {series.max()}")
        ret[key]["max"] = series.max()
        print(f"    stddev = {series.std()}")
        ret[key]["stddev"] = series.std()

    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)

    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    print(f"Number Of Errored Requests: {num_errors}")
    error_code_frequency = dict(error_codes.value_counts())
    if num_errors:
        error_code_frequency = dict(error_codes.value_counts())
        print("Error Code Frequency")
        print(error_code_frequency)
    ret[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)

    overall_output_throughput = df_without_errored_req[
        common_metrics.NUM_OUTPUT_TOKENS
    ].sum() / (end_time - start_time)

    print(f"Overall Output Throughput: {overall_output_throughput}")
    ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
        num_completed_requests / (end_time - start_time) * 60
    )
    print(f"Number Of Completed Requests: {num_completed_requests}")
    print(f"Number Of Concurrent Workers: {num_concurrent_workers}")
    print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")

    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

    return ret


def run_token_benchmark(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_workers: int,
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, Any],
    llm_api: str = "sambastudio",
    mode: str = "stream",
):
    """
    Args:

        model (str): The name of the model to query.
        mean_input_tokens (int): The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens (int): The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens (int): The mean number of tokens to generate per request.
        stddev_output_tokens (int): The standard deviation of the number of tokens to generate per request.
        timeout_s (int): The amount of time to run the test for before reporting results.
        max_num_completed_requests (int): The number of requests to complete before finishing the test.
        num_concurrent_workers (int): The number of concurrent workers to make. Increase
            this to increase the amount of load and vice versa.
        additional_sampling_params (str): Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir (str): The directory to save the results to.
        user_metadata (Dict[str, Any]): Additional metadata to include in the results.
        llm_api (str): The name of the llm api to use. Static for now. Defaults to "sambastudio".
        mode (str): mode of the API. Either "stream" or "batch". Defaults to "stream".
    """

    if mean_input_tokens < 40:
        raise ValueError(
            "The minimum number of input tokens that will be sent is 40"
            " because of the prompting logic right now"
        )

    # Calculate performance metrics individually and summary
    summary, individual_responses = get_token_throughput_latencies(
        model=model,
        mean_input_tokens=mean_input_tokens,
        stddev_input_tokens=stddev_input_tokens,
        mean_output_tokens=mean_output_tokens,
        stddev_output_tokens=stddev_output_tokens,
        timeout_s=timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        num_concurrent_workers=num_concurrent_workers,
        additional_sampling_params=json.loads(additional_sampling_params),
        llm_api=llm_api,
        mode=mode,
    )

    # Build and output performance reports
    if results_dir:
        filename = f"{model}_{mean_input_tokens}_{mean_output_tokens}_{num_concurrent_workers}_{mode}"
        filename = re.sub(r"[^\w\d-]+", "-", filename)
        filename = re.sub(r"-{2,}", "-", filename)
        summary_filename = f"{filename}_summary"
        individual_responses_filename = f"{filename}_individual_responses"

        # Update to metadata.
        summary.update(user_metadata)

        results = LLMPerfResults(name=summary_filename, metadata=summary)
        results_dir = Path(results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")

        try:
            with open(results_dir / f"{summary_filename}.json", "w") as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e

        try:
            with open(results_dir / f"{individual_responses_filename}.json", "w") as f:
                json.dump(individual_responses, f, indent=4)
        except Exception as e:
            print(individual_responses)
            raise e


# Define process arguments
args = argparse.ArgumentParser(
    description="Run a token throughput and latency benchmark."
)

args.add_argument(
    "--model", type=str, required=True, help="The model to use for this load test."
)
args.add_argument(
    "--mean-input-tokens",
    type=int,
    default=550,
    help=(
        "The mean number of tokens to send in the prompt for the request. "
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-input-tokens",
    type=int,
    default=150,
    help=(
        "The standard deviation of number of tokens to send in the prompt for the request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--mean-output-tokens",
    type=int,
    default=150,
    help=(
        "The mean number of tokens to generate from each llm request. This is the max_tokens param "
        "for the completions API. Note that this is not always the number of tokens returned. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-output-tokens",
    type=int,
    default=80,
    help=(
        "The stdandard deviation on the number of tokens to generate per llm request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="The amount of time to run the load test for. (default: %(default)s)",
)
args.add_argument(
    "--max-num-completed-requests",
    type=int,
    default=10,
    help=(
        "The number of requests to complete before finishing the test. Note "
        "that its possible for the test to timeout first. (default: %(default)s)"
    ),
)
args.add_argument(
    "--num-concurrent-workers",
    type=int,
    default=10,
    help=("The number of concurrent workers to send (default: %(default)s)"),
)
args.add_argument(
    "--additional-sampling-params",
    type=str,
    default="{}",
    help=(
        "Additional sampling params to send with the each request to the LLM API. "
        "(default: %(default)s) No additional sampling params are sent."
    ),
)
args.add_argument(
    "--results-dir",
    type=str,
    default="",
    help=(
        "The directory to save the results to. "
        "(`default: %(default)s`) No results are saved)"
    ),
)
args.add_argument(
    "--metadata",
    type=str,
    default="",
    help=(
        "A comma separated list of metadata to include in the results, e.g. "
        "name=foo,bar=1. These will be added to the metadata field of the results. "
    ),
)

if __name__ == "__main__":

    load_dotenv("../.env", override=True)
    env_vars = dict(os.environ)

    args = args.parse_args()

    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    # Call benchmarking process. Static param values are intentional and still WIP.
    run_token_benchmark(
        model=args.model,
        mean_input_tokens=args.mean_input_tokens,
        stddev_input_tokens=args.stddev_input_tokens,
        mean_output_tokens=args.mean_output_tokens,
        stddev_output_tokens=args.stddev_output_tokens,
        timeout_s=args.timeout,
        max_num_completed_requests=args.max_num_completed_requests,
        num_concurrent_workers=args.num_concurrent_workers,
        additional_sampling_params=args.additional_sampling_params,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
        llm_api="sambastudio",
        mode="stream",
    )
