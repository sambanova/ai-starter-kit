"""
This test measures the latency difference between the request establishing the first HTTPS connection
and the subsequent requests reusing the same HTTPS connection (also known as HTTP connection pooling).
The test issues multiple requests. The first request should take longer than the others due to
connection establishment overhead.

To run this test:
python3 ./test_connection_pooling.py --api-url <URL> --api-key <KEY> --model <MODEL>

Optional arguments:
--test-duration: Test duration in seconds (default: 50)
--sleep-duration: Sleep duration between requests in seconds (default: 10)
--max-tokens: Maximum tokens to generate (default: 500)
--prompt: Custom prompt to use for testing
"""

import argparse
import json
import logging
import statistics
import time

import requests

# DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True
# DEBUG

"""
Request parameters documented here. Please change according to your use case
"""
_PROMPT = "Tell me a short story. The story should have two main characters: A prince and a princess."
_MAX_TOKENS_TO_GENERATE = 500


def parse_arguments():
    """Parse command line arguments and return args structure
       Returns:
           Parsed arguments
"""
    parser = argparse.ArgumentParser(description='Test connection pooling for API endpoints')
    parser.add_argument("--api-url", required=True, help="API URL to test against")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--model", required=True, help="Model name to use for testing")
    parser.add_argument("--test-duration", type=int, default=50,
                        help="Test duration in seconds (default: 50)")
    parser.add_argument("--sleep-duration", type=int, default=10,
                        help="Sleep duration between requests in seconds (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Maximum tokens to generate (default: 500)")
    parser.add_argument("--prompt", default=_PROMPT, help="Prompt to use for testing")
    return parser.parse_args()


def create_session():
    """Create and configure a requests session for connection pooling
       Returns:
           instance of a session
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=10)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def create_payload(prompt, max_tokens, model):
    """Create the request payload
       Args:
           prompt: Input prompt
           max_tokens: Maximum tokens to generate
           model: Model to be queried

       Returns:
           json structured payload
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_queue_metrics": True,
        "max_tokens": max_tokens,
        "stop": ["<|eot_id|>"],
        "model": model
    }


def run_connection_pooling_test(api_url, api_key, model, test_duration, sleep_duration, max_tokens, prompt):
    """
    Run the connection pooling test

    Args:
        api_url (str): The API URL to test against
        api_key (str): The API key for authentication
        model (str): The model name to use
        test_duration (int): Test duration in seconds
        sleep_duration (int): Sleep duration between requests in seconds
        max_tokens (int): Maximum tokens to generate
        prompt (str): Prompt to use for testing

    Returns:
        dict: Dictionary containing test results and statistics
    """
    # Create session for connection pooling
    session = create_session()

    # Setup headers and payload
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    sample_payload = create_payload(prompt, max_tokens, model)
    end_sequence = "data: [DONE]"

    # Initialize result arrays
    ttfts = []
    queuing_times = []
    total_times = []

    # Calculate iterations
    iterations = int(test_duration / sleep_duration)

    print(f"Starting connection pooling test with {iterations} iterations")
    print(f"Test duration: {test_duration}s, Sleep duration: {sleep_duration}s")
    print(f"Model: {model}, Max tokens: {max_tokens}")

    # Measure the time taken for the POST request
    for i in range(iterations):
        start_time = time.time()
        ttft_measured = False
        ttft_time = None
        total_latency = None
        response_time = None

        with session.post(api_url, headers=headers, json=sample_payload, timeout=1800, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_to_process = line.decode('utf-8')
                        if line_to_process.startswith(end_sequence):
                            # Termination of string according to OpenAI Spec
                            # See https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream for details
                            continue
                        if line_to_process:
                            # Line is of the format: data: <JSON formatted output>. We extract the JSON format part out.
                            value = json.loads(line_to_process[6:])
                            data = None
                            if isinstance(value, dict):
                                choices = value.get("choices")
                                if choices and not ttft_measured:
                                    ttft_time = time.time()
                                    ttft_measured = True
                                data = value.get("usage")
                                if data:
                                    total_latency = data.get("total_latency")
                                queue_metrics = value.get("queue_metrics")
                                if queue_metrics:
                                    job_queue_metrics = queue_metrics.get("aggregated_job_queue_metrics")
                                    response_time = job_queue_metrics.get("response_duration")
            else:
                print("Error in processing request")
                print(response)
                continue

        end_time = time.time()
        total_elapsed_time = end_time - start_time

        # Calculate metrics
        if ttft_time and total_latency and response_time:
            ttft = ttft_time - start_time
            queuing_time = response_time/1000 - total_latency

            ttfts.append(ttft)
            queuing_times.append(queuing_time)
            total_times.append(total_elapsed_time)

            print(f"====Iteration #{i+1}====")
            print(f"TTFT: {ttft * 1000:.4f} ms")
            print(f"Queuing Time: {queuing_time*1000:.4f} ms")
            print(f"Time taken for POST request: {total_elapsed_time*1000:.4f} ms")
        else:
            print(f"====Iteration #{i+1} - Skipped (missing metrics)====")

        time.sleep(sleep_duration)

    # Calculate and return statistics
    results = {
        "iterations": iterations,
        "test_duration": test_duration,
        "sleep_duration": sleep_duration,
        "model": model,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "ttfts": ttfts,
        "queuing_times": queuing_times,
        "total_times": total_times
    }

    if ttfts:
        results["median_ttft"] = statistics.median(ttfts)
        results["mean_ttft"] = statistics.mean(ttfts)
        results["min_ttft"] = min(ttfts)
        results["max_ttft"] = max(ttfts)

    if queuing_times:
        results["median_queuing_time"] = statistics.median(queuing_times)
        results["mean_queuing_time"] = statistics.mean(queuing_times)

    if total_times:
        results["median_total_time"] = statistics.median(total_times)
        results["mean_total_time"] = statistics.mean(total_times)

    return results


def print_results(results):
    """Print the test results"""
    print(f"\n====Test Results====")
    print(f"Model: {results['model']}")
    print(f"Max tokens: {results['max_tokens']}")
    print(f"Iterations: {results['iterations']}")
    print(f"Test duration: {results['test_duration']}s")
    print(f"Sleep duration: {results['sleep_duration']}s")

    if 'median_ttft' in results:
        print(f"\n====TTFT Statistics====")
        print(f"Median TTFT: {results['median_ttft']*1000:.4f} ms")
        print(f"Mean TTFT: {results['mean_ttft']*1000:.4f} ms")
        print(f"Min TTFT: {results['min_ttft']*1000:.4f} ms")
        print(f"Max TTFT: {results['max_ttft']*1000:.4f} ms")

    if 'median_queuing_time' in results:
        print(f"\n====Queuing Time Statistics====")
        print(f"Median Queuing Time: {results['median_queuing_time']*1000:.4f} ms")
        print(f"Mean Queuing Time: {results['mean_queuing_time']*1000:.4f} ms")

    if 'median_total_time' in results:
        print(f"\n====Total Response Time Statistics====")
        print(f"Median Response Time: {results['median_total_time']*1000:.4f} ms")
        print(f"Mean Response Time: {results['mean_total_time']*1000:.4f} ms")


def main():
    """Main function to run the connection pooling test"""
    args = parse_arguments()

    # Run the test
    results = run_connection_pooling_test(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        test_duration=args.test_duration,
        sleep_duration=args.sleep_duration,
        max_tokens=args.max_tokens,
        prompt=args.prompt
    )

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
