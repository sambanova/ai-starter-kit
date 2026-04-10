"""
Shared comparison utilities for Kit vs vLLM benchmarking.

No Streamlit dependency — safe to import from both CLI (evaluator.py)
and UI (streamlit_utils.py) contexts.
"""

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def get_vllm_summary_metrics(vllm_result_file_path: str) -> Dict[str, Any]:
    """
    Extract summary metrics from a vLLM raw result JSON file.

    Args:
        vllm_result_file_path: Path to the vLLM bench-serve output JSON.

    Returns:
        Dictionary of summary metrics with consistent keys shared with
        calculate_kit_summary_metrics output.
    """
    with open(vllm_result_file_path, 'r') as f:
        vllm_results = json.load(f)

    duration = vllm_results.get('duration', 0)
    completed = vllm_results.get('completed', 0)
    total_input_tokens = vllm_results.get('total_input_tokens', 0)
    total_output_tokens = vllm_results.get('total_output_tokens', 0)

    request_throughput = completed / duration if duration > 0 else 0
    output_throughput = total_output_tokens / duration if duration > 0 else 0
    total_token_throughput = (total_input_tokens + total_output_tokens) / duration if duration > 0 else 0

    return {
        'duration': duration,
        'completed_requests': completed,
        'failed_requests': vllm_results.get('failed', 0),
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'request_throughput': request_throughput,
        'output_throughput': output_throughput,
        'total_token_throughput': total_token_throughput,
        'mean_ttft_ms': vllm_results.get('mean_ttft_ms', 0),
        'median_ttft_ms': vllm_results.get('median_ttft_ms', 0),
        'mean_tpot_ms': vllm_results.get('mean_tpot_ms', 0),
        'median_tpot_ms': vllm_results.get('median_tpot_ms', 0),
        'mean_itl_ms': vllm_results.get('mean_itl_ms', 0),
        'median_itl_ms': vllm_results.get('median_itl_ms', 0),
    }


def calculate_kit_summary_metrics(df_individual: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary metrics from the Kit individual-responses DataFrame.

    Args:
        df_individual: DataFrame loaded from *_individual_responses.json.

    Returns:
        Dictionary of summary metrics with keys consistent with
        get_vllm_summary_metrics output.
    """
    valid_df = df_individual[df_individual['error_code'].isnull()]

    start_times = pd.to_datetime(valid_df['start_time'])
    end_times = pd.to_datetime(valid_df['end_time'])
    duration = (end_times.max() - start_times.min()).total_seconds()

    completed = len(valid_df)
    failed = len(df_individual) - completed

    total_input_tokens = valid_df['number_input_tokens'].sum()
    total_output_tokens = valid_df['number_output_tokens'].sum()

    request_throughput = completed / duration if duration > 0 else 0
    output_throughput = total_output_tokens / duration if duration > 0 else 0
    total_token_throughput = (total_input_tokens + total_output_tokens) / duration if duration > 0 else 0

    mean_ttft_ms = valid_df['client_ttft_s'].mean() * 1000
    median_ttft_ms = valid_df['client_ttft_s'].median() * 1000
    mean_e2e_latency_ms = valid_df['client_end_to_end_latency_s'].mean() * 1000
    median_e2e_latency_ms = valid_df['client_end_to_end_latency_s'].median() * 1000
    mean_output_throughput_per_request = valid_df['client_output_token_per_s_per_request'].mean()
    median_output_throughput_per_request = valid_df['client_output_token_per_s_per_request'].median()

    if 'client_mean_inter_token_latency_s' in valid_df.columns:
        mean_itl_ms: Optional[float] = valid_df['client_mean_inter_token_latency_s'].mean() * 1000
        median_itl_ms: Optional[float] = valid_df['client_mean_inter_token_latency_s'].median() * 1000
    else:
        mean_itl_ms = None
        median_itl_ms = None

    return {
        'duration': duration,
        'completed_requests': completed,
        'failed_requests': failed,
        'total_input_tokens': int(total_input_tokens),
        'total_output_tokens': int(total_output_tokens),
        'request_throughput': request_throughput,
        'output_throughput': output_throughput,
        'total_token_throughput': total_token_throughput,
        'mean_ttft_ms': mean_ttft_ms,
        'median_ttft_ms': median_ttft_ms,
        'mean_e2e_latency_ms': mean_e2e_latency_ms,
        'median_e2e_latency_ms': median_e2e_latency_ms,
        'mean_output_throughput_per_request': mean_output_throughput_per_request,
        'median_output_throughput_per_request': median_output_throughput_per_request,
        'mean_itl_ms': mean_itl_ms,
        'median_itl_ms': median_itl_ms,
    }


def build_comparison_dataframe(kit_metrics: Dict[str, Any], vllm_metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a side-by-side Kit vs vLLM summary metrics DataFrame.

    Usable by both CLI (printed via to_string) and Streamlit (rendered via st.dataframe).

    Args:
        kit_metrics:  Output of calculate_kit_summary_metrics.
        vllm_metrics: Output of get_vllm_summary_metrics.

    Returns:
        DataFrame with columns ['Metric', 'Kit', 'vLLM'].
    """

    def fmt(val: Any, fmt_str: str = '.2f') -> str:
        if val is None:
            return 'N/A'
        try:
            return format(float(val), fmt_str)
        except (TypeError, ValueError):
            return str(val)

    rows = [
        ('Duration (s)',                         fmt(kit_metrics['duration']),                          fmt(vllm_metrics['duration'])),
        ('Completed Requests',                   str(kit_metrics['completed_requests']),                str(vllm_metrics['completed_requests'])),
        ('Failed Requests',                      str(kit_metrics['failed_requests']),                   str(vllm_metrics['failed_requests'])),
        ('Total Input Tokens',                   str(kit_metrics['total_input_tokens']),                str(vllm_metrics['total_input_tokens'])),
        ('Total Output Tokens',                  str(kit_metrics['total_output_tokens']),               str(vllm_metrics['total_output_tokens'])),
        ('Request Throughput (req/s)',            fmt(kit_metrics['request_throughput'],   '.4f'),       fmt(vllm_metrics['request_throughput'],   '.4f')),
        ('Output Throughput (tokens/s)',          fmt(kit_metrics['output_throughput']),                 fmt(vllm_metrics['output_throughput'])),
        ('Total Token Throughput (tokens/s)',     fmt(kit_metrics['total_token_throughput']),            fmt(vllm_metrics['total_token_throughput'])),
        ('Mean TTFT (ms)',                        fmt(kit_metrics['mean_ttft_ms']),                      fmt(vllm_metrics['mean_ttft_ms'])),
        ('Median TTFT (ms)',                      fmt(kit_metrics['median_ttft_ms']),                    fmt(vllm_metrics['median_ttft_ms'])),
        ('Mean ITL (ms)',                         fmt(kit_metrics.get('mean_itl_ms')),                   fmt(vllm_metrics.get('mean_itl_ms'))),
        ('Median ITL (ms)',                       fmt(kit_metrics.get('median_itl_ms')),                 fmt(vllm_metrics.get('median_itl_ms'))),
    ]

    return pd.DataFrame(rows, columns=['Metric', 'Kit', 'vLLM'])


def _percentile_row(
    label: str,
    kit_series: pd.Series,
    vllm_series: pd.Series,
    scale: float = 1.0,
) -> Tuple[str, str, str]:
    """Compute p5/p50/p95 for one metric and return a formatted table row."""

    def pct(series: pd.Series, p: int) -> str:
        vals = series.dropna()
        if vals.empty:
            return 'N/A'
        return f'{float(np.percentile(vals, p)) * scale:.2f}'

    kit_str = f'p5={pct(kit_series, 5)} / p50={pct(kit_series, 50)} / p95={pct(kit_series, 95)}'
    vllm_str = f'p5={pct(vllm_series, 5)} / p50={pct(vllm_series, 50)} / p95={pct(vllm_series, 95)}'
    return (label, kit_str, vllm_str)


def build_percentile_dataframe(kit_df: pd.DataFrame, vllm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a TTFT and ITL percentile distribution DataFrame (p5/p50/p95).

    Args:
        kit_df:  Individual responses DataFrame from Kit.
        vllm_df: Individual responses DataFrame from vLLM (compatible format).

    Returns:
        DataFrame with columns ['Metric', 'Kit', 'vLLM'].
    """
    kit_valid = kit_df[kit_df['error_code'].isna()] if 'error_code' in kit_df.columns else kit_df
    vllm_valid = vllm_df[vllm_df['error_code'].isna()] if 'error_code' in vllm_df.columns else vllm_df

    rows = []

    # TTFT (convert s → ms)
    rows.append(_percentile_row(
        'TTFT (ms)',
        kit_valid['client_ttft_s'],
        vllm_valid['client_ttft_s'],
        scale=1000.0,
    ))

    # ITL (convert s → ms)
    kit_itl = (
        kit_valid['client_mean_inter_token_latency_s']
        if 'client_mean_inter_token_latency_s' in kit_valid.columns
        else pd.Series(dtype=float)
    )
    vllm_itl = (
        vllm_valid['client_mean_inter_token_latency_s']
        if 'client_mean_inter_token_latency_s' in vllm_valid.columns
        else pd.Series(dtype=float)
    )
    rows.append(_percentile_row('ITL (ms)', kit_itl, vllm_itl, scale=1000.0))

    return pd.DataFrame(rows, columns=['Metric', 'Kit', 'vLLM'])


def print_comparison_report(
    kit_metrics: Dict[str, Any],
    vllm_metrics: Dict[str, Any],
    kit_df: pd.DataFrame,
    vllm_df: pd.DataFrame,
    model_name: str,
) -> None:
    """
    Print a comprehensive Kit vs vLLM comparison report to stdout.

    Includes a summary metrics table and latency distribution percentiles,
    matching the information shown in the UI's comparison view.

    Args:
        kit_metrics:  Output of calculate_kit_summary_metrics.
        vllm_metrics: Output of get_vllm_summary_metrics.
        kit_df:       Individual responses DataFrame from Kit.
        vllm_df:      Individual responses DataFrame from vLLM.
        model_name:   Model name string, used in the report header.
    """
    sep = '=' * 80

    print(f'\n{sep}')
    print(f'  BENCHMARK COMPARISON: {model_name}')
    print(sep)

    print('\n--- Summary Metrics ---\n')
    df_summary = build_comparison_dataframe(kit_metrics, vllm_metrics)
    print(df_summary.to_string(index=False))

    print('\n--- Latency Distribution (p5 / p50 / p95) ---\n')
    df_pct = build_percentile_dataframe(kit_df, vllm_df)
    print(df_pct.to_string(index=False))

    print(f'\n{sep}\n')
