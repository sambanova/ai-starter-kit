"""
vLLM Benchmark Executor

This module provides functionality to run vLLM benchmarks and parse their results.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import pandas as pd

from benchmarking.benchmarking_utils import get_tokenizer_model_name


class VLLMBenchmarkExecutor:
    """Executor for running vLLM benchmark serve commands and parsing results."""

    def __init__(
        self,
        model_name: str,
        results_dir: str,
        timeout: int = 600,
        user_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the vLLM benchmark executor.

        Args:
            model_name: Name of the model to benchmark
            results_dir: Directory to save benchmark results
            timeout: Timeout for the benchmark in seconds
            user_metadata: Additional metadata to include in results
        """
        self.model_name = model_name
        self.results_dir = results_dir
        self.timeout = timeout
        self.user_metadata = user_metadata or {}
        self.stop_event = threading.Event()
        self.result_file_path = None
        self.individual_responses_file_path = None
        self.summary_file_path = None
        self._progress_completed = 0
        self._progress_total = 0

        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

    def run_benchmark(
        self,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        num_concurrent_requests: Optional[int] = None,
        request_rate: Optional[float] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: str = "chat/completions",
        progress_bar: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run vLLM benchmark serve command against remote API.

        Args:
            num_input_tokens: Number of input tokens per request
            num_output_tokens: Number of output tokens to generate
            num_requests: Total number of requests to send
            num_concurrent_requests: Number of concurrent requests (converted to request rate)
            request_rate: Request rate in queries per second (overrides num_concurrent_requests)
            api_base: API base URL (e.g., https://api.sambanova.ai/)
            api_key: API key for authentication
            endpoint: API endpoint to use
            progress_bar: Optional callback for progress updates

        Returns:
            Dictionary containing benchmark results
        """
        # Get API credentials from environment if not provided
        if api_base is None:
            api_base = os.environ.get('SAMBANOVA_API_BASE', 'https://api.sambanova.ai/')
        if api_key is None:
            api_key = os.environ.get('SAMBANOVA_API_KEY', '')

        # Ensure base URL ends with /
        if not api_base.endswith('/'):
            api_base += '/'

        # Generate unique result filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir_name = f"vllm_{timestamp}"
        result_dir_path = os.path.join(self.results_dir, result_dir_name)
        os.makedirs(result_dir_path, exist_ok=True)

        # vLLM will create files in this directory
        self.result_file_path = result_dir_path

        # Get tokenizer model name for vLLM's --model parameter
        tokenizer_model_name = get_tokenizer_model_name(self.model_name)

        # Build vLLM command using random dataset
        cmd = [
            "vllm", "bench", "serve",
            "--backend", "openai-chat",
            "--base-url", api_base,
            "--dataset-name", "random",
            "--endpoint", endpoint,
            "--model", tokenizer_model_name,  # HuggingFace tokenizer model
            "--served_model_name", self.model_name,  # Actual API model name
            "--random-input-len", str(num_input_tokens),
            "--random-output-len", str(num_output_tokens),
            "--num-prompts", str(num_requests),
            "--save-detailed",
            "--save-result",
            "--result-dir", result_dir_path,
        ]

        # Add API key as environment variable
        env = os.environ.copy()
        if api_key:
            env['OPENAI_API_KEY'] = api_key

        # Add request rate or calculate from concurrent requests
        if request_rate is not None:
            cmd.extend(["--request-rate", str(request_rate)])
        elif num_concurrent_requests is not None and num_concurrent_requests > 1:
            # Use concurrent requests as request rate (approximate)
            # This means we'll send num_concurrent_requests requests per second
            cmd.extend(["--request-rate", str(num_concurrent_requests)])
        else:
            # Default to 1 request per second for sequential execution
            cmd.extend(["--request-rate", "1"])

        # Run the benchmark
        try:
            if progress_bar:
                progress_bar(0, 100)

            # Print the command for debugging
            print("\n" + "="*80)
            print("Running vLLM benchmark command:")
            print("="*80)
            print(" ".join(cmd))
            print("="*80 + "\n")

            # Reset progress tracking for this run
            self._progress_completed = 0
            self._progress_total = 0

            # Pipe stderr to parse tqdm progress; stdout streams directly to terminal
            process = subprocess.Popen(
                cmd,
                stdout=None,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )

            # Background thread: parse tqdm progress from stderr and forward to terminal
            stderr_thread = threading.Thread(
                target=self._monitor_stderr,
                args=(process.stderr,),
                daemon=True,
            )
            stderr_thread.start()

            # Monitor process and update progress bar
            while True:
                if self.stop_event.is_set():
                    process.terminate()
                    raise Exception("Benchmark stopped by user")

                retcode = process.poll()
                if retcode is not None:
                    break

                if progress_bar:
                    if self._progress_total > 0:
                        progress_bar(self._progress_completed, self._progress_total)
                    else:
                        progress_bar(1, 100)  # Waiting for first tqdm output

                time.sleep(0.5)

            stderr_thread.join(timeout=2)

            if progress_bar:
                progress_bar(100, 100)

            # Check for errors
            if retcode != 0:
                print(f"\n{'='*80}")
                print(f"vLLM benchmark failed with return code: {retcode}")
                print(f"{'='*80}\n")
                raise Exception(f"vLLM benchmark failed with return code {retcode}")

            # Find and parse the result file in the directory
            results, result_file_path = self._find_and_parse_results(result_dir_path)

            # Store the actual result file path
            self.result_file_path = result_file_path

            # Log partial failures if any
            num_failed = results.get('failed', 0)
            num_total = results.get('num_prompts', 0)
            if num_failed > 0:
                print(f"\nWarning: {num_failed}/{num_total} requests failed. Charts will show the {num_total - num_failed} successful requests.\n")

            # Create compatible output files
            self._create_compatible_output(results, num_input_tokens, num_output_tokens, num_concurrent_requests)

            return results

        except FileNotFoundError:
            raise Exception(
                "vLLM is not installed or not in PATH.\n\n"
                "To install vLLM (works on all platforms for API benchmarking):\n"
                "  pip install vllm>=0.6.0\n\n"
                "Note: For API benchmarking, vLLM doesn't require GPU/local model.\n"
                "It benchmarks remote APIs like SambaNova Cloud."
            )
        except Exception as e:
            raise Exception(f"Error running vLLM benchmark: {str(e)}")

    def _monitor_stderr(self, stderr_pipe) -> None:
        """Read vLLM stderr, parse tqdm progress (X/Y), and forward output to terminal."""
        buffer = ''
        while True:
            chunk = stderr_pipe.read(256)
            if not chunk:
                break
            sys.stderr.write(chunk)
            sys.stderr.flush()
            buffer += chunk
            # Split on carriage return or newline to isolate tqdm lines
            parts = re.split(r'[\r\n]', buffer)
            buffer = parts[-1]  # keep incomplete last segment
            for part in parts[:-1]:
                match = re.search(r'(\d+)/(\d+)', part)
                if match:
                    self._progress_completed = int(match.group(1))
                    self._progress_total = int(match.group(2))

    def _find_and_parse_results(self, result_dir: str) -> tuple[Dict[str, Any], str]:
        """
        Find and parse vLLM benchmark result JSON file in the results directory.

        Args:
            result_dir: Directory containing vLLM results

        Returns:
            Tuple of (parsed results dictionary, result file path)
        """
        # vLLM creates a JSON file with benchmark results
        # Look for JSON files in the directory
        json_files = [f for f in os.listdir(result_dir) if f.endswith('.json')]

        if not json_files:
            raise Exception(f"No result JSON files found in: {result_dir}")

        # Use the first JSON file (vLLM typically creates one main results file)
        result_file = os.path.join(result_dir, json_files[0])

        print(f"\n{'='*80}")
        print(f"Found vLLM result file: {json_files[0]}")
        print(f"{'='*80}\n")

        with open(result_file, 'r') as f:
            results = json.load(f)

        return results, result_file

    def _create_compatible_output(
        self,
        vllm_results: Dict[str, Any],
        num_input_tokens: int,
        num_output_tokens: int,
        num_concurrent_requests: Optional[int],
    ) -> None:
        """
        Convert vLLM results to format compatible with existing plotting functions.

        Args:
            vllm_results: Raw vLLM benchmark results
            num_input_tokens: Number of input tokens used
            num_output_tokens: Number of output tokens used
            num_concurrent_requests: Number of concurrent requests
        """
        num_completed = vllm_results.get('completed', 0)

        # Extract per-request metrics — all arrays have num_prompts elements (including failed)
        ttfts = vllm_results.get('ttfts', [])
        output_lens = vllm_results.get('output_lens', [])
        input_lens = vllm_results.get('input_lens', [])
        itls = vllm_results.get('itls', [])
        errors_raw = vllm_results.get('errors', None)  # per-request error strings; empty = success

        # num_prompts covers all requests (successful + failed)
        num_prompts = vllm_results.get('num_prompts', len(ttfts))

        # Aggregate fallback (vLLM reports stats only for completed requests)
        mean_ttft_s = vllm_results.get('mean_ttft_ms', 0) / 1000

        def _get_error(i: int) -> Optional[str]:
            """Return error string for request i, or None if it succeeded."""
            if errors_raw is not None and i < len(errors_raw) and errors_raw[i]:
                return str(errors_raw[i])
            # Fallback heuristic when errors field is absent: zero TTFT + zero output = failed
            if (
                i < len(ttfts) and i < len(output_lens)
                and ttfts[i] == 0.0 and output_lens[i] == 0
            ):
                return 'REQUEST_FAILED'
            return None

        # Compute E2E latencies and ITL sums for successful requests (for summary stats)
        calculated_e2e_latencies = []
        calculated_itl_sums = []
        for i in range(num_prompts):
            if _get_error(i):
                continue
            request_ttft_s = ttfts[i] if i < len(ttfts) else mean_ttft_s
            if i < len(itls) and itls[i]:
                itl_sum = sum(itls[i])
                request_e2e_s = request_ttft_s + itl_sum
                calculated_itl_sums.append(itl_sum)
            else:
                request_e2e_s = vllm_results.get('duration', 1) / num_completed if num_completed > 0 else 1
                calculated_itl_sums.append(0)
            calculated_e2e_latencies.append(request_e2e_s)

        if calculated_e2e_latencies:
            mean_e2e_s = sum(calculated_e2e_latencies) / len(calculated_e2e_latencies)
            median_e2e_s = sorted(calculated_e2e_latencies)[len(calculated_e2e_latencies) // 2]
            variance = sum((x - mean_e2e_s) ** 2 for x in calculated_e2e_latencies) / len(calculated_e2e_latencies)
            std_e2e_s = variance ** 0.5
        else:
            mean_e2e_s = 0
            median_e2e_s = 0
            std_e2e_s = 0

        mean_itl_sum_s = sum(calculated_itl_sums) / len(calculated_itl_sums) if calculated_itl_sums else 0

        # Build per-request records for ALL requests (successful and failed)
        individual_responses = []
        for i in range(num_prompts):
            error_str = _get_error(i)

            if error_str:
                # Failed request: set error fields, leave metrics as None
                record = {
                    'client_ttft_s': None,
                    'client_end_to_end_latency_s': None,
                    'client_output_token_per_s_per_request': None,
                    'number_input_tokens': input_lens[i] if i < len(input_lens) else num_input_tokens,
                    'number_output_tokens': 0,
                    'client_inter_token_latencies_s': [],
                    'client_mean_inter_token_latency_s': None,
                    'server_ttft_s': None,
                    'server_end_to_end_latency_s': None,
                    'server_output_token_per_s_per_request': None,
                    'server_number_output_tokens': None,
                    'server_number_input_tokens': None,
                    'batch_size_used': None,
                    'error_code': error_str,
                    'error_msg': error_str,
                }
            else:
                request_ttft_s = ttfts[i] if i < len(ttfts) else mean_ttft_s
                request_output_tokens = output_lens[i] if i < len(output_lens) else num_output_tokens
                request_input_tokens = input_lens[i] if i < len(input_lens) else num_input_tokens
                request_itls = itls[i] if i < len(itls) and itls[i] else []
                request_itl_sum = sum(request_itls) if request_itls else 0
                request_e2e_s = (
                    request_ttft_s + request_itl_sum
                    if request_itls
                    else vllm_results.get('duration', 1) / num_completed if num_completed > 0 else 1
                )
                output_tokens_per_s = request_output_tokens / request_itl_sum if request_itl_sum > 0 else 0

                if len(request_itls) > 1:
                    mean_itl = sum(request_itls[1:]) / len(request_itls[1:])
                elif len(request_itls) == 1:
                    mean_itl = request_itls[0]
                else:
                    mean_itl = None

                record = {
                    'client_ttft_s': request_ttft_s,
                    'client_end_to_end_latency_s': request_e2e_s,
                    'client_output_token_per_s_per_request': output_tokens_per_s,
                    'number_input_tokens': request_input_tokens,
                    'number_output_tokens': request_output_tokens,
                    'client_inter_token_latencies_s': request_itls,
                    'client_mean_inter_token_latency_s': mean_itl,
                    'server_ttft_s': None,
                    'server_end_to_end_latency_s': None,
                    'server_output_token_per_s_per_request': None,
                    'server_number_output_tokens': None,
                    'server_number_input_tokens': None,
                    'batch_size_used': None,
                    'error_code': None,
                    'error_msg': None,
                }
            individual_responses.append(record)

        # Save individual responses file
        individual_file_name = self.result_file_path.replace('.json', '_individual_responses.json')
        with open(individual_file_name, 'w') as f:
            json.dump(individual_responses, f, indent=2)
        self.individual_responses_file_path = individual_file_name

        # Create summary file
        num_failed = vllm_results.get('failed', 0)
        summary = {
            'name': f'vllm_{self.model_name}',
            'model': self.model_name,
            'num_concurrent_requests': num_concurrent_requests or 1,
            'results_client_ttft_s_mean': mean_ttft_s,
            'results_client_ttft_s_median': vllm_results.get('median_ttft_ms', 0) / 1000,
            'results_client_ttft_s_stddev': vllm_results.get('std_ttft_ms', 0) / 1000,
            'results_client_end_to_end_latency_s_mean': mean_e2e_s,
            'results_client_end_to_end_latency_s_median': median_e2e_s,
            'results_client_end_to_end_latency_s_stddev': std_e2e_s,
            'results_client_output_token_per_s_per_request_mean': (
                num_output_tokens / mean_itl_sum_s if mean_itl_sum_s > 0 else 0
            ),
            'results_error_rate': num_failed / num_prompts if num_prompts > 0 else 0,
            'results_num_completed_requests': num_completed,
            'results_num_failed_requests': num_failed,
            'request_throughput': vllm_results.get('request_throughput', 0),
            'output_throughput': vllm_results.get('output_throughput', 0),
        }

        # Save summary file
        summary_file_name = self.result_file_path.replace('.json', '_summary.json')
        with open(summary_file_name, 'w') as f:
            json.dump(summary, f, indent=2)
        self.summary_file_path = summary_file_name

    def stop_benchmark(self) -> None:
        """Stop the running benchmark."""
        self.stop_event.set()

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame compatible with existing plotting functions.

        Returns:
            DataFrame with benchmark results
        """
        if self.individual_responses_file_path is None:
            raise Exception("No results available. Run benchmark first.")

        df = pd.read_json(self.individual_responses_file_path)
        return df


def parse_vllm_results_to_dataframe(result_file_path: str) -> pd.DataFrame:
    """
    Parse vLLM result file and convert to DataFrame format.

    Args:
        result_file_path: Path to vLLM result JSON file

    Returns:
        DataFrame with parsed results
    """
    # Look for the individual responses file
    individual_file_path = result_file_path.replace('.json', '_individual_responses.json')

    if os.path.exists(individual_file_path):
        df = pd.read_json(individual_file_path)
        return df
    else:
        # Parse raw vLLM output and convert
        with open(result_file_path, 'r') as f:
            results = json.load(f)

        ttfts = results.get('ttfts', [])
        output_lens = results.get('output_lens', [])
        itls = results.get('itls', [])
        errors_raw = results.get('errors', None)
        mean_ttft_s = results.get('mean_ttft_ms', 0) / 1000
        num_completed = results.get('completed', 0)
        num_prompts = results.get('num_prompts', len(ttfts))

        rows = []
        for i in range(num_prompts):
            # Detect failures
            error_str = None
            if errors_raw is not None and i < len(errors_raw) and errors_raw[i]:
                error_str = str(errors_raw[i])
            elif i < len(ttfts) and i < len(output_lens) and ttfts[i] == 0.0 and output_lens[i] == 0:
                error_str = 'REQUEST_FAILED'

            if error_str:
                rows.append({
                    'client_ttft_s': None,
                    'client_end_to_end_latency_s': None,
                    'server_ttft_s': None,
                    'server_end_to_end_latency_s': None,
                    'error_code': error_str,
                })
            else:
                request_ttft_s = ttfts[i] if i < len(ttfts) else mean_ttft_s
                request_itls = itls[i] if i < len(itls) and itls[i] else []
                request_itl_sum = sum(request_itls)
                request_e2e_s = (
                    request_ttft_s + request_itl_sum
                    if request_itls
                    else results.get('duration', 1) / num_completed if num_completed > 0 else 1
                )
                rows.append({
                    'client_ttft_s': request_ttft_s,
                    'client_end_to_end_latency_s': request_e2e_s,
                    'server_ttft_s': None,
                    'server_end_to_end_latency_s': None,
                    'error_code': None,
                })

        df = pd.DataFrame(rows)
        return df
