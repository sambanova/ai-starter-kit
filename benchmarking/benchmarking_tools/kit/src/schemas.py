"""Canonical, cross-tool result schema.

Every benchmarking tool (Kit-native, vLLM, aiperf, ...) writes its per-request and per-run
output through these pydantic models instead of hand-built dicts. Field names are copied
verbatim from `benchmarking/benchmarking_tools/kit/src/llmperf/common_metrics.py`'s string constants, so Kit's
existing metric dicts validate against `RequestMetric` with no translation table.

`extra='forbid'` on both models means a typo'd or unexpected field raises `ValidationError`
at construction time instead of silently producing a differently-shaped result file (the bug
that let vLLM's "compatible" summary drift into a much thinner shape than Kit's). Anything
that is genuinely tool-native and has no cross-tool equivalent (aiperf's `scenario`,
`submission_valid`, etc.) belongs in the `extra` dict, not as a new top-level field.
"""

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# Per-request client-side metric field names that get a {quantiles, mean, min, max, stddev}
# entry in BenchmarkSummary.results. Mirrors the first loop in
# BasePerformanceEvaluator.build_metrics_summary (performance_evaluation.py).
CLIENT_METRIC_FIELDS: List[str] = [
    'client_ttft_s',
    'client_end_to_end_latency_s',
    'client_output_token_per_s_per_request',
    'number_input_tokens',
    'number_output_tokens',
    'client_mean_inter_token_latency_s',
    'client_network_latency_ttft_s',
    'client_network_latency_e2e_s',
]

# Per-request server-side metric field names, same treatment. Mirrors the second loop in
# build_metrics_summary. Only ever populated by the 'kit' tool (SambaNova server telemetry).
SERVER_METRIC_FIELDS: List[str] = [
    'server_ttft_s',
    'server_end_to_end_latency_s',
    'server_output_token_per_s_per_request',
    'server_output_token_after_first_per_s_first_ten_per_request',
    'server_number_input_tokens',
    'server_number_output_tokens',
    'server_number_reasoning_tokens',
    'server_number_cached_tokens',
    'acceptance_rate',
]

ALL_METRIC_FIELDS: List[str] = CLIENT_METRIC_FIELDS + SERVER_METRIC_FIELDS

_QUANTILE_SUFFIXES = ('p5', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99')
_QUANTILE_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Recursively flattens a dict with `sep`-joined keys — same algorithm as
    `benchmarking.benchmarking_tools.kit.src.llmperf.llmperf_utils.flatten_dict`, duplicated locally (rather than
    imported) since `schemas.py` is meant to stay tool-agnostic/foundational, not depend on
    Kit-native-specific plumbing.
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


class QuantileStats(BaseModel):
    model_config = ConfigDict(extra='forbid')

    quantiles: Dict[str, float] = Field(default_factory=dict)  # p5,p25,p50,p75,p90,p95,p99
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    stddev: Optional[float] = None


def compute_quantile_stats(values: Sequence[Optional[float]]) -> Optional['QuantileStats']:
    """Computes {quantiles: {p5..p99}, mean, min, max, stddev} from a list of numeric values —
    same quantile set and 4-decimal rounding as `BasePerformanceEvaluator.build_metrics_summary`
    (performance_evaluation.py), so any tool computing its own per-metric stats this way produces
    a `BenchmarkSummary.results` shape genuinely equivalent to the Kit's, not just similarly named.
    None values (e.g. failed requests) are dropped. Returns None if nothing is left to summarize.
    """
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    series = pd.Series(clean)
    quantiles = {f'p{int(q * 100)}': round(float(series.quantile(q)), 4) for q in _QUANTILE_QUANTILES}
    return QuantileStats(
        quantiles=quantiles,
        mean=round(float(series.mean()), 4),
        min=round(float(series.min()), 4),
        max=round(float(series.max()), 4),
        stddev=round(float(series.std()), 4) if len(series) > 1 else 0.0,
    )


class RequestMetric(BaseModel):
    """Per-request metrics. Field names match common_metrics.py's string constants exactly."""

    model_config = ConfigDict(extra='forbid')

    # identity / bookkeeping
    prompt_name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_code: Optional[str] = None
    error_msg: Optional[str] = None

    # client-side (populated whenever a tool can time a single HTTP round trip)
    client_ttft_s: Optional[float] = None
    client_end_to_end_latency_s: Optional[float] = None
    client_output_token_per_s_per_request: Optional[float] = None
    client_total_tokens_per_s_per_request: Optional[float] = None
    client_inter_token_latencies_s: Optional[List[float]] = None
    client_mean_inter_token_latency_s: Optional[float] = None
    client_network_latency_ttft_s: Optional[float] = None
    client_network_latency_e2e_s: Optional[float] = None
    number_input_tokens: Optional[int] = None
    number_output_tokens: Optional[int] = None
    number_total_tokens: Optional[int] = None

    # server-side — structurally unpopulatable by vLLM/aiperf (no SambaNova telemetry access);
    # populated only by the 'kit' tool.
    server_ttft_s: Optional[float] = None
    server_end_to_end_latency_s: Optional[float] = None
    server_output_token_per_s_per_request: Optional[float] = None
    server_output_token_after_first_per_s_first_ten_per_request: Optional[float] = None
    server_total_tokens_per_s_per_request: Optional[float] = None
    server_number_input_tokens: Optional[int] = None
    server_number_output_tokens: Optional[int] = None
    server_number_total_tokens: Optional[int] = None
    server_number_reasoning_tokens: Optional[int] = None
    server_number_cached_tokens: Optional[int] = None
    batch_size_used: Optional[int] = None
    queue_time: Optional[float] = None
    acceptance_rate: Optional[float] = None

    extra: Dict[str, Any] = Field(default_factory=dict)  # tool-native per-request fields


class BenchmarkSummary(BaseModel):
    """Per-run summary. Constructed by every tool's executor before being written to disk."""

    model_config = ConfigDict(extra='forbid')

    # provenance
    tool: str  # 'kit' | 'vllm' | 'aiperf' | ...
    workload_mode: str  # 'custom' | 'synthetic' | 'real_workload' | 'agentic_coding' | ...
    name: str
    timestamp: Optional[int] = None
    model: str
    run_uuid: Optional[str] = None

    # workload shape — Optional because each mode/tool populates a different subset
    num_concurrent_requests: Optional[int] = None
    qps: Optional[float] = None
    qps_distribution: Optional[str] = None
    num_input_tokens: Optional[int] = None
    num_output_tokens: Optional[int] = None
    num_requests: Optional[int] = None
    benchmark_duration_s: Optional[float] = None  # duration-based runs (e.g. aiperf agentic_coding)

    # per-metric aggregate stats, keyed by RequestMetric field name
    results: Dict[str, QuantileStats] = Field(default_factory=dict)

    # scalar summary fields
    num_requests_started: Optional[int] = None
    num_completed_requests: Optional[int] = None
    num_completed_requests_per_min: Optional[float] = None
    error_rate: Optional[float] = None
    number_errors: Optional[int] = None
    error_code_frequency: Optional[str] = None
    client_total_output_throughput: Optional[float] = None
    mean_output_throughput_token_per_s: Optional[float] = None
    request_throughput: Optional[float] = None  # req/s — natively reported by some tools (vLLM, aiperf)

    additional_sampling_params: Dict[str, Any] = Field(default_factory=dict)
    user_metadata: Dict[str, Any] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)  # tool-native fields (aiperf's scenario, etc.)

    def to_legacy_flat_dict(self) -> Dict[str, Any]:
        """Emits today's on-disk `_summary.json` flattened key shape (`results_<metric>_<stat>`,
        `results_<metric>_quantiles_p50`, etc.) — the same shape `LLMPerfResults.to_dict()`
        produced — so existing readers (`benchmarking/utils.py::read_perf_eval_json_files`,
        notebooks, `ResultsConsolidator`) keep working unmodified now that
        `BasePerformanceEvaluator.save_results` constructs a `BenchmarkSummary` before
        serializing. Only genuinely new top-level fields (`tool`, `workload_mode`, `run_uuid`)
        are additive; everything else matches key-for-key, including recursively flattening
        nested dicts in `additional_sampling_params`/`user_metadata`/`extra` exactly like the
        original `flatten_dict` did.
        """
        flat: Dict[str, Any] = {
            'name': self.name,
            'model': self.model,
            'tool': self.tool,
            'workload_mode': self.workload_mode,
        }
        if self.timestamp is not None:
            flat['timestamp'] = self.timestamp
        if self.run_uuid is not None:
            flat['run_uuid'] = self.run_uuid
        for field_name in (
            'num_concurrent_requests', 'qps', 'qps_distribution', 'num_input_tokens',
            'num_output_tokens', 'benchmark_duration_s', 'request_throughput',
        ):
            value = getattr(self, field_name)
            if value is not None:
                flat[field_name] = value
        if self.num_requests is not None:
            # Kit's pre-existing on-disk convention names this field `request_count` for
            # 'custom' mode specifically (no fixed-token-count axis to call it num_requests
            # against) and `num_requests` for every other workload mode/tool.
            key = 'request_count' if self.workload_mode == 'custom' else 'num_requests'
            flat[key] = self.num_requests

        for metric_name, stats in self.results.items():
            prefix = f'results_{metric_name}'
            for stat_name in ('mean', 'min', 'max', 'stddev'):
                stat_value = getattr(stats, stat_name)
                if stat_value is not None:
                    flat[f'{prefix}_{stat_name}'] = stat_value
            for q_name, q_value in stats.quantiles.items():
                flat[f'{prefix}_quantiles_{q_name}'] = q_value

        for field_name in (
            'num_requests_started', 'num_completed_requests', 'num_completed_requests_per_min',
            'error_rate', 'number_errors', 'error_code_frequency', 'client_total_output_throughput',
            'mean_output_throughput_token_per_s',
        ):
            value = getattr(self, field_name)
            if value is not None:
                flat[f'results_{field_name}'] = value

        # Recursively flattened (not just one level) so a nested dict anywhere in
        # additional_sampling_params/user_metadata/extra matches today's `flatten_dict`-based
        # behavior exactly — e.g. custom mode's `extra['sampling_params']` (a dict) flattens to
        # `sampling_params_<key>` keys, same as the pre-schema on-disk shape.
        flat.update(_flatten_dict(self.additional_sampling_params, parent_key='additional_sampling_params'))
        flat.update(_flatten_dict(self.user_metadata))
        flat.update(_flatten_dict(self.extra))
        return flat

    @classmethod
    def from_kit_legacy_summary_dict(
        cls, flat: Dict[str, Any], tool: str = 'kit', workload_mode: str = 'synthetic'
    ) -> 'BenchmarkSummary':
        """Parses today's on-disk Kit `_summary.json` shape (flattened by
        `LLMPerfResults`/`flatten_dict`) into a validated `BenchmarkSummary`.
        """
        remaining = dict(flat)
        known_scalars = {
            'name': remaining.pop('name', None),
            'model': remaining.pop('model', None),
            'timestamp': remaining.pop('timestamp', None),
            'num_concurrent_requests': remaining.pop('num_concurrent_requests', None),
            'qps': remaining.pop('qps', None),
            'qps_distribution': remaining.pop('qps_distribution', None),
            'num_input_tokens': remaining.pop('num_input_tokens', None),
            'num_output_tokens': remaining.pop('num_output_tokens', None),
            # 'custom' mode's pre-existing on-disk convention names this field `request_count`;
            # every other mode/tool uses `num_requests` — see to_legacy_flat_dict.
            'num_requests': remaining.pop('num_requests', None) or remaining.pop('request_count', None),
        }

        results: Dict[str, QuantileStats] = {}
        for metric_name in ALL_METRIC_FIELDS:
            prefix = f'results_{metric_name}'
            stats_kwargs: Dict[str, Any] = {}
            for stat_name in ('mean', 'min', 'max', 'stddev'):
                key = f'{prefix}_{stat_name}'
                if key in remaining:
                    stats_kwargs[stat_name] = remaining.pop(key)
            quantiles = {}
            for q in _QUANTILE_SUFFIXES:
                key = f'{prefix}_quantiles_{q}'
                if key in remaining:
                    quantiles[q] = remaining.pop(key)
            if quantiles:
                stats_kwargs['quantiles'] = quantiles
            if stats_kwargs:
                results[metric_name] = QuantileStats(**stats_kwargs)

        summary_scalars = {}
        for field_name in (
            'num_requests_started', 'num_completed_requests', 'num_completed_requests_per_min',
            'error_rate', 'number_errors', 'error_code_frequency', 'client_total_output_throughput',
            'mean_output_throughput_token_per_s',
        ):
            key = f'results_{field_name}'
            if key in remaining:
                summary_scalars[field_name] = remaining.pop(key)

        additional_sampling_params = {}
        for key in list(remaining):
            if key.startswith('additional_sampling_params_'):
                additional_sampling_params[key[len('additional_sampling_params_'):]] = remaining.pop(key)

        return cls(
            tool=tool,
            workload_mode=workload_mode,
            name=known_scalars['name'] or '',
            model=known_scalars['model'] or '',
            timestamp=known_scalars['timestamp'],
            run_uuid=remaining.pop('run_uuid', None),
            num_concurrent_requests=known_scalars['num_concurrent_requests'],
            qps=known_scalars['qps'],
            qps_distribution=known_scalars['qps_distribution'],
            num_input_tokens=known_scalars['num_input_tokens'],
            num_output_tokens=known_scalars['num_output_tokens'],
            num_requests=known_scalars['num_requests'],
            results=results,
            additional_sampling_params=additional_sampling_params,
            extra=remaining,  # anything left over (user metadata, etc.) is tool-native/extra
            **summary_scalars,
        )

    @classmethod
    def from_vllm_legacy_summary_dict(cls, flat: Dict[str, Any]) -> 'BenchmarkSummary':
        """Parses today's `VLLMBenchmarkExecutor._create_compatible_output` flat dict shape
        into a validated `BenchmarkSummary`. That shape is thinner than Kit's (only
        mean/median/stddev, no full quantile set; `median` instead of `p50`).
        """
        remaining = dict(flat)
        results: Dict[str, QuantileStats] = {}
        for metric_name in ('client_ttft_s', 'client_end_to_end_latency_s', 'client_output_token_per_s_per_request'):
            prefix = f'results_{metric_name}'
            stats_kwargs: Dict[str, Any] = {}
            mean_key, median_key, stddev_key = f'{prefix}_mean', f'{prefix}_median', f'{prefix}_stddev'
            if mean_key in remaining:
                stats_kwargs['mean'] = remaining.pop(mean_key)
            if median_key in remaining:
                stats_kwargs['quantiles'] = {'p50': remaining.pop(median_key)}
            if stddev_key in remaining:
                stats_kwargs['stddev'] = remaining.pop(stddev_key)
            if stats_kwargs:
                results[metric_name] = QuantileStats(**stats_kwargs)

        return cls(
            tool='vllm',
            workload_mode='synthetic',
            name=remaining.pop('name', ''),
            model=remaining.pop('model', ''),
            num_concurrent_requests=remaining.pop('num_concurrent_requests', None),
            results=results,
            error_rate=remaining.pop('results_error_rate', None),
            num_completed_requests=remaining.pop('results_num_completed_requests', None),
            number_errors=remaining.pop('results_num_failed_requests', None),
            request_throughput=remaining.pop('request_throughput', None),
            client_total_output_throughput=remaining.pop('output_throughput', None),
            extra=remaining,
        )
