"""
Comparison utilities for the Kit's Streamlit UI (kit vs. vLLM side-by-side display).

No Streamlit dependency — safe to import from UI (streamlit_utils.py) contexts.
"""

import json
from typing import Any, Dict, Optional

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


