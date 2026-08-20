"""Converts NVIDIA aiperf's native output (`profile_export.jsonl` + `profile_export_aiperf.json`,
written by `aiperf profile ...` into `--artifact-dir`) into the Kit's own two-file convention
(`aiperf_individual_responses.json` + `aiperf_summary.json`), written alongside them in the same
artifact directory.

This is a POST-HOC converter over aiperf's already-finished, already-written output files -- it
does not wrap or intercept the live benchmark run itself (see `convert_vllm_output.py`'s
docstring for why that distinction matters). Field mappings are the same ones verified live
against a real aiperf 0.12.0 run in this repo's history: per-request metric names/units in
`profile_export.jsonl` (`time_to_first_token`/`request_latency`/`inter_token_latency` in ms,
`input_sequence_length`/`output_sequence_length` in tokens), a top-level `error` object
(`{"code": ..., "type": ..., "message": ...}`) on failed requests, and an aggregate file whose
per-metric entries already carry a full `p5`-`p99` percentile set plus `avg`/`min`/`max`/`std` --
richer than vLLM's aggregate, so this converter reads percentiles directly from it rather than
recomputing them from individual records (falling back to recomputing only for a metric the
aggregate happens to omit).

Usage:
    python convert_aiperf_output.py --artifact-dir ./data/aiperf_results/aiperf_.../ \
        --model-name Meta-Llama-3.3-70B-Instruct
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

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

# aiperf per-request metric names (profile_export.jsonl `metrics.<name>.value`, ms/tokens) that
# map onto RequestMetric's common vocabulary. Values are converted ms -> s where applicable.
_MS_TO_S_FIELDS = {
    'time_to_first_token': 'client_ttft_s',
    'request_latency': 'client_end_to_end_latency_s',
    'inter_token_latency': 'client_mean_inter_token_latency_s',
}
_TOKEN_COUNT_FIELDS = {
    'input_sequence_length': 'number_input_tokens',
    'output_sequence_length': 'number_output_tokens',
}
_QUANTILE_KEYS = ('p5', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99')


def find_output_files(artifact_dir: str, num_profile_runs: Optional[int] = None) -> Tuple[str, str]:
    """Locates aiperf's aggregate summary and per-request JSONL within the artifact dir.
    `num_profile_runs > 1` puts the aggregate under an `aggregate/` subdirectory; single-run
    outputs sit at the artifact dir's top level.
    """
    candidates = [artifact_dir]
    if num_profile_runs and num_profile_runs > 1:
        candidates.insert(0, os.path.join(artifact_dir, 'aggregate'))

    for base in candidates:
        aggregate_path = os.path.join(base, 'profile_export_aiperf.json')
        individual_path = os.path.join(base, 'profile_export.jsonl')
        if os.path.exists(aggregate_path) and os.path.exists(individual_path):
            return aggregate_path, individual_path

    raise FileNotFoundError(
        f'Could not find profile_export_aiperf.json/profile_export.jsonl under {artifact_dir} '
        f'(checked: {candidates}).'
    )


def parse_individual_record(raw_record: Dict[str, Any]) -> RequestMetric:
    """Maps one `profile_export.jsonl` line to a `RequestMetric`."""
    metadata = raw_record.get('metadata', {})
    metrics = raw_record.get('metrics', {})

    kwargs: Dict[str, Any] = {}
    for aiperf_name, field_name in _MS_TO_S_FIELDS.items():
        entry = metrics.get(aiperf_name)
        if entry is not None and entry.get('value') is not None:
            kwargs[field_name] = entry['value'] / 1000.0
    for aiperf_name, field_name in _TOKEN_COUNT_FIELDS.items():
        entry = metrics.get(aiperf_name)
        if entry is not None and entry.get('value') is not None:
            kwargs[field_name] = entry['value']
    output_token_count = metrics.get('output_token_count')
    if 'number_output_tokens' not in kwargs and output_token_count is not None:
        kwargs['number_output_tokens'] = output_token_count.get('value')

    # A failed request carries a top-level "error": {"code": ..., "type": ..., "message": ...}
    # (confirmed live: a 401 auth failure produced exactly this shape). Fall back to "missing
    # latency metrics" only if there's no explicit error object either.
    error = raw_record.get('error')
    if error is not None:
        kwargs['error_code'] = str(error.get('code', error.get('type', 'AIPERF_ERROR')))
        kwargs['error_msg'] = error.get('message', '')
    elif 'client_ttft_s' not in kwargs and 'client_end_to_end_latency_s' not in kwargs:
        kwargs['error_code'] = 'AIPERF_MISSING_METRICS'
        kwargs['error_msg'] = 'No latency metrics present in profile_export.jsonl record'

    kwargs['extra'] = {
        k: v
        for k, v in {
            'session_num': metadata.get('session_num'),
            'turn_index': metadata.get('turn_index'),
            'benchmark_phase': metadata.get('benchmark_phase'),
            'x_request_id': metadata.get('x_request_id'),
        }.items()
        if v is not None
    }
    return RequestMetric(**kwargs)


def build_summary(
    aggregate: Dict[str, Any],
    individual_responses: List[RequestMetric],
    model_name: str,
    workload_mode: str,
    num_concurrent_requests: Optional[int],
    qps: Optional[float],
    num_input_tokens: Optional[int],
    num_output_tokens: Optional[int],
) -> BenchmarkSummary:
    # aiperf's aggregate already carries a full p5-p99 percentile set plus avg/min/max/std per
    # metric -- read it directly rather than recomputing, only falling back to computing our own
    # stats from individual_responses for a metric the aggregate happens to omit.
    results: Dict[str, QuantileStats] = {}
    for aiperf_name, field_name in {**_MS_TO_S_FIELDS, **_TOKEN_COUNT_FIELDS}.items():
        entry = aggregate.get(aiperf_name)
        if not entry:
            continue
        scale = 1 / 1000.0 if aiperf_name in _MS_TO_S_FIELDS else 1.0
        quantiles = {q: entry[q] * scale for q in _QUANTILE_KEYS if entry.get(q) is not None}
        stats_kwargs: Dict[str, Any] = {'quantiles': quantiles}
        if entry.get('avg') is not None:
            stats_kwargs['mean'] = entry['avg'] * scale
        if entry.get('min') is not None:
            stats_kwargs['min'] = entry['min'] * scale
        if entry.get('max') is not None:
            stats_kwargs['max'] = entry['max'] * scale
        if entry.get('std') is not None:
            stats_kwargs['stddev'] = entry['std'] * scale
        results[field_name] = QuantileStats(**stats_kwargs)

    for field_name in ('client_ttft_s', 'client_end_to_end_latency_s', 'client_mean_inter_token_latency_s'):
        if field_name not in results:
            stats = compute_quantile_stats([getattr(r, field_name) for r in individual_responses])
            if stats is not None:
                results[field_name] = stats

    num_completed = sum(1 for r in individual_responses if r.error_code is None)
    num_errors = len(individual_responses) - num_completed

    request_throughput = None
    request_throughput_entry = aggregate.get('request_throughput')
    if request_throughput_entry is not None:
        request_throughput = request_throughput_entry.get('avg')

    effective_concurrency = None
    concurrency_entry = aggregate.get('effective_concurrency')
    if concurrency_entry is not None:
        effective_concurrency = concurrency_entry.get('avg')

    # aiperf's aggregate-level scalar metrics (benchmark_duration, request_throughput,
    # effective_concurrency, ...) are all {"unit": ..., "avg": ...} dicts, not bare numbers --
    # confirmed live (a bare-float assumption here raised a pydantic ValidationError).
    benchmark_duration_s = None
    benchmark_duration_entry = aggregate.get('benchmark_duration')
    if benchmark_duration_entry is not None:
        benchmark_duration_s = benchmark_duration_entry.get('avg')

    return BenchmarkSummary(
        tool='aiperf',
        workload_mode=workload_mode,
        name=f'aiperf_{model_name}',
        model=model_name,
        num_concurrent_requests=num_concurrent_requests
        or (round(effective_concurrency) if effective_concurrency is not None else None),
        qps=qps,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        num_requests=len(individual_responses),
        benchmark_duration_s=benchmark_duration_s,
        results=results,
        num_completed_requests=num_completed,
        number_errors=num_errors,
        error_rate=(num_errors / len(individual_responses)) if individual_responses else None,
        request_throughput=request_throughput,
        extra={
            k: v
            for k, v in {
                'schema_version': aggregate.get('schema_version'),
                'aiperf_version': aggregate.get('aiperf_version'),
                'benchmark_id': aggregate.get('benchmark_id'),
            }.items()
            if v is not None
        },
    )


def convert(
    artifact_dir: str,
    model_name: str,
    workload_mode: str,
    num_concurrent_requests: Optional[int],
    qps: Optional[float],
    num_input_tokens: Optional[int],
    num_output_tokens: Optional[int],
    num_profile_runs: Optional[int],
) -> None:
    aggregate_path, individual_path = find_output_files(artifact_dir, num_profile_runs)

    with open(aggregate_path) as f:
        aggregate: Dict[str, Any] = json.load(f)

    individual_responses: List[RequestMetric] = []
    with open(individual_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            individual_responses.append(parse_individual_record(json.loads(line)))

    summary = build_summary(
        aggregate,
        individual_responses,
        model_name,
        workload_mode,
        num_concurrent_requests,
        qps,
        num_input_tokens,
        num_output_tokens,
    )

    individual_out_path = os.path.join(artifact_dir, 'aiperf_individual_responses.json')
    summary_out_path = os.path.join(artifact_dir, 'aiperf_summary.json')

    with open(individual_out_path, 'w') as f:
        json.dump([m.model_dump() for m in individual_responses], f, indent=2)
    with open(summary_out_path, 'w') as f:
        json.dump(summary.to_legacy_flat_dict(), f, indent=2)

    print(f'Wrote {individual_out_path}')
    print(f'Wrote {summary_out_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--artifact-dir', type=str, required=True, help="aiperf's --artifact-dir from the run to convert."
    )
    parser.add_argument('--model-name', type=str, required=True, help='Actual API model name.')
    parser.add_argument(
        '--workload-mode',
        type=str,
        default='synthetic',
        help="Kit-style workload label to record ('synthetic', 'real_workload', 'agentic_coding', ...). "
        '(default: %(default)s)',
    )
    parser.add_argument('--num-concurrent-requests', type=int, default=None, help='--concurrency used for the run.')
    parser.add_argument('--qps', type=float, default=None, help='--request-rate used for the run, if rate-paced.')
    parser.add_argument('--num-input-tokens', type=int, default=None, help='Configured input token count, if fixed.')
    parser.add_argument(
        '--num-output-tokens', type=int, default=None, help='Configured output token count, if fixed.'
    )
    parser.add_argument(
        '--num-profile-runs', type=int, default=None, help='--num-profile-runs used for the run, if set.'
    )
    args = parser.parse_args()
    convert(
        args.artifact_dir,
        args.model_name,
        args.workload_mode,
        args.num_concurrent_requests,
        args.qps,
        args.num_input_tokens,
        args.num_output_tokens,
        args.num_profile_runs,
    )


if __name__ == '__main__':
    main()
