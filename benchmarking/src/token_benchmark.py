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
import ray
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
    request_configs_for_thread: list,
    # tokenizer,
    completed_requests: list,
):
    for request_config in request_configs_for_thread:
        # req_metrics = llm_request(request_config, tokenizer)
        req_metrics = llm_request(request_config)
        completed_requests.extend([req_metrics[0]])


# def build_prompt_2(prompt_tokens_mean, prompt_tokens_stddev):
#     # num_input_tokens = sample_random_positive_int(
#     #     prompt_tokens_mean, prompt_tokens_stddev
#     # )
#     template = "In a small, forgotten village nestled deep within the dense, ancient forests of a remote region, life continued much as it had for centuries, largely untouched by the passage of time or the march of modernity. The villagers, a tight-knit community of fewer than a hundred souls, lived simple lives, their routines dictated by the rhythms of nature and the changing seasons. The village itself, a cluster of weather-worn cottages with thatched roofs and stone walls, stood as a testament to a bygone era. The cobblestone streets, winding and narrow, were lined with wildflowers in the warmer months and blanketed with snow in the winter, creating a picturesque yet melancholic scene. At the heart of the village stood a grand, ancient oak tree, its gnarled branches spreading wide like a protective canopy over the village square. This tree, known as the Elder Oak, was not just a physical landmark but also a central figure in the village's lore and traditions. It was said to be over a thousand years old, having witnessed countless generations come and go. Beneath its boughs, villagers gathered for festivals, markets, and communal meetings. The Elder Oak was believed to possess a deep wisdom and a connection to the spiritual realm, often serving as a place of reflection and meditation for the villagers. One crisp autumn morning, as the first golden rays of sunlight filtered through the mist, a stranger arrived in the village. He was a tall, enigmatic figure, clad in a long, dark coat and a wide-brimmed hat that cast a shadow over his face. His appearance was striking, with sharp, angular features and piercing green eyes that seemed to hold secrets untold. He introduced himself as Alaric, a traveler and scholar of ancient lore, on a quest to uncover forgotten knowledge and hidden truths. His arrival sparked a wave of curiosity and apprehension among the villagers, who were not accustomed to outsiders. Alaric's presence in the village quickly became a subject of both fascination and suspicion. Some saw him as a harbinger of change, a catalyst that could disrupt the delicate balance of their secluded lives. Others were intrigued by his tales of distant lands and lost civilizations, eager to learn more about the world beyond their forested haven. Despite the mixed reactions, Alaric was granted permission to stay in the village, taking up residence in a small, abandoned cottage on the outskirts. As days turned into weeks, Alaric's influence on the village began to grow. He spent his days exploring the surrounding forests, often disappearing for hours, only to return with ancient artifacts and strange, cryptic texts. He would spend his evenings in the village tavern, sharing stories and knowledge with the villagers, who gathered around him in rapt attention. Among his most ardent listeners was Elara, the village healer, a woman of great wisdom and curiosity. Elara was particularly fascinated by Alaric's knowledge of medicinal plants and ancient healing practices, and the two quickly formed a bond, exchanging knowledge and insights. One night, as a fierce storm raged outside, Alaric confided in Elara about the true nature of his quest. He revealed that he was searching for the fabled Crystal of Eternity, an artifact of immense power said to grant its possessor eternal life and unparalleled wisdom. According to ancient texts, the crystal was hidden somewhere within the forest, guarded by ancient spirits and protected by powerful enchantments. Alaric believed that the key to finding the crystal lay in deciphering the runes inscribed on the Elder Oak, which he suspected held clues to its location. Elara, intrigued by the possibility of uncovering such a powerful artifact, agreed to help Alaric. Together, they began to study the runes on the Elder Oak, spending countless hours in its shadow, poring over ancient manuscripts and comparing notes. Their quest soon became the talk of the village, with many villagers offering their assistance and sharing local legends that had been passed down through generations. The village, once a place of quiet routine, was now abuzz with excitement and anticipation. As they delved deeper into their research, Alaric and Elara uncovered a series of cryptic prophecies that seemed to hint at the challenges they would face on their quest. The prophecies spoke of trials of courage, wisdom, and heart, each guarded by ancient spirits who would test their worthiness. Determined to succeed, they prepared themselves for the journey ahead, gathering supplies and seeking guidance from the village elders. Their journey took them deep into the heart of the forest, where they encountered both wonders and dangers. They faced treacherous terrain, battled fierce creatures, and solved intricate puzzles, all while deciphering the cryptic clues that led them closer to the Crystal of Eternity. Along the way, they forged a deep bond, their shared experiences and mutual respect blossoming into a profound friendship. As they neared the end of their quest, they faced their greatest challenge yet: a final confrontation with the guardian spirit of the crystal, a powerful entity of ancient magic and wisdom. In a climactic battle of wits and strength, Alaric and Elara drew upon all they had learned, their combined knowledge and determination proving to be the key to their victory. With the Crystal of Eternity finally in their grasp, Alaric and Elara returned to the village as heroes. Their journey had not only uncovered a powerful artifact but also brought the village closer together, reigniting a sense of community and shared purpose. The villagers celebrated their return with a grand festival beneath the Elder Oak, honoring their bravery and the newfound knowledge they had brought back with them. In the end, Alaric chose to leave the Crystal of Eternity in the care of the village, believing that its power was best kept safe and used for the greater good. He bid farewell to the villagers and continued his travels, ever the seeker of knowledge and adventure. Elara, now a revered figure in the village, continued to share her wisdom and healing practices, her life forever changed by the journey she had undertaken. The village, once a quiet and forgotten place, now thrived with a renewed sense of purpose and wonder. The Elder Oak, standing tall and proud in the village square, continued to watch over the villagers, its ancient runes a reminder of the timeless wisdom and the enduring spirit of those who had come before. However, "
#     random_number = random.randint(100, 1000)
#     new_prompt = template[:random_number]
#     return new_prompt, random_number


def get_request_configs(
    model: str,
    max_num_completed_requests: int,
    num_concurrent_workers,
    mean_output_tokens,
    stddev_output_tokens,
    mean_input_tokens,
    stddev_input_tokens,
    additional_sampling_params,
    llm_api,
    mode,
):
    request_configs = []

    for _ in range(max_num_completed_requests):
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

        # prompt = build_prompt_2(
        #     prompt_tokens_mean=mean_input_tokens,
        #     prompt_tokens_stddev=stddev_input_tokens,
        # )

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

        # collect request config
        request_configs.append(request_config)

    return request_configs


def get_token_throughput_latencies(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    test_timeout_s=90,
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
        test_timeout_s (int): The amount of time to run the test for before reporting results. Defaults to 90.
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

    # tokenizer = get_tokenizer(model)

    start_time = time.monotonic()

    # Get all request configs

    all_request_configs = get_request_configs(
        model,
        max_num_completed_requests,
        num_concurrent_workers,
        mean_output_tokens,
        stddev_output_tokens,
        mean_input_tokens,
        stddev_input_tokens,
        additional_sampling_params,
        llm_api,
        mode,
    )

    # Get request config batches

    requests_per_thread = max_num_completed_requests // num_concurrent_workers
    remainder = max_num_completed_requests % num_concurrent_workers

    request_config_batches = []
    idx = 0
    for i in range(num_concurrent_workers):
        num_requests_for_thread = requests_per_thread + (1 if i < remainder else 0)
        request_config_batch = all_request_configs[
            idx : idx + num_requests_for_thread
        ].copy()
        idx = idx + num_requests_for_thread
        request_config_batches.append(request_config_batch)

    # Create threads

    threads = []
    completed_requests = []
    for i, request_configs_for_thread in enumerate(request_config_batches):
        print(f"Executing Thread {i}")
        thread = threading.Thread(
            target=send_requests,
            args=(
                request_configs_for_thread,
                # tokenizer,
                completed_requests,
            ),
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete

    for thread in threads:
        thread.join()

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
    test_timeout_s: int,
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
        test_timeout_s (int): The amount of time to run the test for before reporting results.
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
    # TODO: change according to new prompt
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
        test_timeout_s=test_timeout_s,
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
        test_timeout_s=args.timeout,
        max_num_completed_requests=args.max_num_completed_requests,
        num_concurrent_workers=args.num_concurrent_workers,
        additional_sampling_params=args.additional_sampling_params,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
        llm_api="sambastudio",
        mode="stream",
    )
