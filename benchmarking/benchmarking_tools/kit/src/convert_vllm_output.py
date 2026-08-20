"""Converts vLLM's native `vllm bench serve --save-result --save-detailed` output JSON into the
Kit's own two-file convention (`<raw>_individual_responses.json` + `<raw>_summary.json`),
written alongside the raw file so downstream analysis (notebooks, the bundles consolidator, etc.)
can treat a vLLM run the same way as a Kit run.

This is a POST-HOC converter over vLLM's already-finished, already-written output file -- it does
not wrap or intercept the live benchmark run itself. An earlier version of this integration did
wrap execution live (mismatched kwargs, thin schema drift); converting one known, finished file
after the fact is a much smaller, lower-risk surface, while still reusing schemas.py's exact same
canonical model + `to_legacy_flat_dict()` so the on-disk shape is genuinely Kit-equivalent.

Usage:
    python convert_vllm_output.py --raw-result-path ./data/results/vllm/openai-chat-....json \
        --model-name Meta-Llama-3.3-70B-Instruct --num-input-tokens 1000 --num-output-tokens 1000
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
benchmarking_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
repo_dir = os.path.abspath(os.path.join(benchmarking_dir, '..'))
sys.path.append(benchmarking_dir)
sys.path.append(repo_dir)

from benchmarking.benchmarking_tools.kit.src.schemas import (
    BenchmarkSummary,
    QuantileStats,
    RequestMetric,
    compute_quantile_stats,
)


def _get_error(errors_raw: Optional[List[Any]], ttfts: List[float], output_lens: List[int], i: int) -> Optional[str]:
    if errors_raw is not None and i < len(errors_raw) and errors_raw[i]:
        return str(errors_raw[i])
    # Fallback heuristic when errors[] is absent or empty: zero TTFT + zero output = failed.
    if i < len(ttfts) and i < len(output_lens) and ttfts[i] == 0.0 and output_lens[i] == 0:
        return 'REQUEST_FAILED'
    return None


def build_individual_responses(
    raw: Dict[str, Any], num_input_tokens: int, num_output_tokens: int
) -> List[RequestMetric]:
    ttfts = raw.get('ttfts', [])
    output_lens = raw.get('output_lens', [])
    input_lens = raw.get('input_lens', [])
    itls = raw.get('itls', [])
    errors_raw = raw.get('errors')
    num_prompts = raw.get('num_prompts', len(ttfts))
    num_completed = raw.get('completed', 0)
    mean_ttft_s = raw.get('mean_ttft_ms', 0) / 1000

    responses: List[RequestMetric] = []
    for i in range(num_prompts):
        error_str = _get_error(errors_raw, ttfts, output_lens, i)
        if error_str:
            responses.append(
                RequestMetric(
                    number_input_tokens=input_lens[i] if i < len(input_lens) else num_input_tokens,
                    number_output_tokens=0,
                    client_inter_token_latencies_s=[],
                    error_code=error_str,
                    error_msg=error_str,
                )
            )
            continue

        request_ttft_s = ttfts[i] if i < len(ttfts) else mean_ttft_s
        request_output_tokens = output_lens[i] if i < len(output_lens) else num_output_tokens
        request_input_tokens = input_lens[i] if i < len(input_lens) else num_input_tokens
        request_itls = itls[i] if i < len(itls) and itls[i] else []
        request_itl_sum = sum(request_itls) if request_itls else 0
        request_e2e_s = (
            request_ttft_s + request_itl_sum
            if request_itls
            else raw.get('duration', 1) / num_completed
            if num_completed > 0
            else 1
        )
        output_tokens_per_s = request_output_tokens / request_itl_sum if request_itl_sum > 0 else 0
        if len(request_itls) > 1:
            mean_itl = sum(request_itls[1:]) / len(request_itls[1:])
        elif len(request_itls) == 1:
            mean_itl = request_itls[0]
        else:
            mean_itl = None

        responses.append(
            RequestMetric(
                client_ttft_s=request_ttft_s,
                client_end_to_end_latency_s=request_e2e_s,
                client_output_token_per_s_per_request=output_tokens_per_s,
                number_input_tokens=request_input_tokens,
                number_output_tokens=request_output_tokens,
                client_inter_token_latencies_s=request_itls,
                client_mean_inter_token_latency_s=mean_itl,
            )
        )
    return responses


def build_summary(
    raw: Dict[str, Any],
    individual_responses: List[RequestMetric],
    model_name: str,
    workload_mode: str,
    num_input_tokens: int,
    num_output_tokens: int,
) -> BenchmarkSummary:
    successful = [r for r in individual_responses if r.error_code is None]
    results: Dict[str, QuantileStats] = {}
    for metric_name in (
        'client_ttft_s',
        'client_end_to_end_latency_s',
        'client_output_token_per_s_per_request',
        'client_mean_inter_token_latency_s',
        'number_input_tokens',
        'number_output_tokens',
    ):
        stats = compute_quantile_stats([getattr(r, metric_name) for r in successful])
        if stats is not None:
            results[metric_name] = stats

    num_prompts = raw.get('num_prompts', len(individual_responses))
    num_completed = raw.get('completed', 0)
    num_failed = raw.get('failed', 0)

    qps: Optional[float] = None
    request_rate = raw.get('request_rate')
    if request_rate is not None:
        try:
            qps = float(request_rate)
        except (TypeError, ValueError):
            qps = None

    return BenchmarkSummary(
        tool='vllm',
        workload_mode=workload_mode,
        name=f'vllm_{model_name}',
        model=model_name,
        num_concurrent_requests=raw.get('max_concurrency'),
        qps=qps,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        num_requests=num_prompts,
        results=results,
        num_completed_requests=num_completed,
        number_errors=num_failed,
        error_rate=(num_failed / num_prompts) if num_prompts else 0,
        request_throughput=raw.get('request_throughput'),
        client_total_output_throughput=raw.get('output_throughput'),
        # vLLM doesn't echo the served/API model name back into its own output (only the
        # tokenizer id, under model_id) -- keep the tokenizer id + a couple of other tool-native
        # fields for reference, rather than silently dropping them.
        extra={
            'tokenizer_id': raw.get('tokenizer_id'),
            'burstiness': raw.get('burstiness'),
            'backend': raw.get('backend'),
        },
    )


def convert(
    raw_result_path: str, model_name: str, num_input_tokens: int, num_output_tokens: int, workload_mode: str
) -> None:
    with open(raw_result_path) as f:
        raw: Dict[str, Any] = json.load(f)

    individual_responses = build_individual_responses(raw, num_input_tokens, num_output_tokens)
    summary = build_summary(raw, individual_responses, model_name, workload_mode, num_input_tokens, num_output_tokens)

    base = raw_result_path[: -len('.json')] if raw_result_path.endswith('.json') else raw_result_path
    individual_path = f'{base}_individual_responses.json'
    summary_path = f'{base}_summary.json'

    with open(individual_path, 'w') as f:
        json.dump([m.model_dump() for m in individual_responses], f, indent=2)
    with open(summary_path, 'w') as f:
        json.dump(summary.to_legacy_flat_dict(), f, indent=2)

    print(f'Wrote {individual_path}')
    print(f'Wrote {summary_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw-result-path', type=str, required=True, help="vLLM's own native result JSON file.")
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Actual API model name (not in the raw file -- vLLM only echoes the tokenizer id).',
    )
    parser.add_argument(
        '--num-input-tokens', type=int, required=True, help='Configured input token count for this run.'
    )
    parser.add_argument(
        '--num-output-tokens', type=int, required=True, help='Configured output token count for this run.'
    )
    parser.add_argument(
        '--workload-mode',
        type=str,
        default='synthetic',
        help="Kit-style workload label to record ('synthetic', 'real_workload', 'custom', ...). "
        '(default: %(default)s)',
    )
    args = parser.parse_args()
    convert(args.raw_result_path, args.model_name, args.num_input_tokens, args.num_output_tokens, args.workload_mode)


if __name__ == '__main__':
    main()
