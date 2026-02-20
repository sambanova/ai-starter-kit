import base64
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.graph_objs import Figure

from benchmarking.utils import SAMBANOVA_API_BASE
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

LLM_API_OPTIONS = {'sncloud': 'SambaNova Cloud'}
MULTIMODAL_IMAGE_SIZE_OPTIONS = {'na': 'N/A', 'small': 'Small', 'medium': 'Medium', 'large': 'Large'}
QPS_DISTRIBUTION_OPTIONS = {'constant': 'Constant', 'uniform': 'Uniform', 'exponential': 'Exponential'}
APP_PAGES = {
    'synthetic_eval': {
        'file_path': 'pages/synthetic_performance_eval_st.py',
        'page_label': 'Synthetic Performance Evaluation',
        'page_icon': ':material/analytics:',
    },
    'real_workload_eval': {
        'file_path': 'pages/real_workload_eval_st.py',
        'page_label': 'Real Workload Evaluation',
        'page_icon': ':material/speed:',
    },
    'custom_eval': {
        'file_path': 'pages/custom_performance_eval_st.py',
        'page_label': 'Custom Performance Evaluation',
        'page_icon': ':material/instant_mix:',
    },
    'chat_eval': {
        'file_path': 'pages/chat_performance_st.py',
        'page_label': 'Performance on Chat',
        'page_icon': ':material/chat:',
    },
}


def render_logo() -> None:
    # Inject HTML to display the logo in the sidebar at 70% width
    logo_path = os.path.join(repo_dir, 'images', 'dark-logo.png')
    with open(logo_path, 'rb') as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{encoded}" style="width:60%; display: block; max-width:100%;">
        </div>
    """,
        unsafe_allow_html=True,
    )


def set_font() -> None:
    # Load Inter font from Google Fonts and apply globally
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">

        <style>
            /* Apply Exile font to all elements on the page */
            html, body, [class^="css"] :not(.material-icons) {
                font-family: 'Inter', sans-serif !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_title_icon(title: str, icon: Optional[str] = None) -> None:
    # add title and icon
    if icon is not None:
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            st.image(icon)
    st.markdown(
        f"""
        <style>
            .kit-title {{
                text-align: center;
                color: #250E36 !important;
                font-size: 3.0em;
                font-weight: bold;
                margin-bottom: 0.5em;
            }}
        </style>
        <div class="kit-title">{title}</div>
    """,
        unsafe_allow_html=True,
    )


def setup_credentials() -> None:
    """Sets up the credentials for the application."""

    st.title('Setup')

    # Callout to get SambaNova API Key
    st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')

    # Set the llm_api to sncloud (only option for now)
    st.session_state.llm_api = 'sncloud'

    additional_env_vars: Dict[str, Any] = {}
    additional_env_vars = {'SAMBANOVA_API_BASE': SAMBANOVA_API_BASE}

    initialize_env_variables(st.session_state.prod_mode, additional_env_vars)

    if not are_credentials_set():
        api_key, additional_vars = env_input_fields(additional_env_vars)
        if st.button('Save Credentials', key='save_credentials_sidebar'):
            message = save_credentials(api_key, additional_vars, st.session_state.prod_mode)
            st.session_state.mp_events.api_key_saved()
            st.success(message)
            st.rerun()
    else:
        st.success('Credentials are set')
        if st.button('Clear Credentials', key='clear_credentials'):
            if st.session_state.llm_api == 'sncloud':
                save_credentials('', None, st.session_state.prod_mode)
            else:
                save_credentials('', {var: '' for var in additional_env_vars}, st.session_state.prod_mode)
            st.rerun()


def save_uploaded_file(internal_save_path: str) -> str:
    uploaded_file = st.session_state.uploaded_file
    temp_file_path = '.'
    if st.session_state.uploaded_file is not None:
        # Save the uploaded file to a temporary location
        save_dir = os.path.join(os.getcwd(), internal_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        temp_file_path = os.path.join(save_dir, uploaded_file.name)
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.getbuffer())
    return temp_file_path


def find_pages_to_show() -> List[Any]:
    pages = st.session_state.pages_to_show
    pages_to_show = []

    for page_k, _ in APP_PAGES.items():
        if page_k in pages:
            pages_to_show.append(
                st.Page(
                    APP_PAGES[page_k]['file_path'],
                    title=APP_PAGES[page_k]['page_label'],
                    icon=APP_PAGES[page_k]['page_icon'],
                )
            )
    return pages_to_show


def update_progress_bar(step: int, total_steps: int) -> None:
    """Update the progress bar."""
    st.session_state.progress_bar.progress(value=step / total_steps, text=f'Running requests: {step}/{total_steps}')


def set_api_variables() -> Dict[str, Any]:
    if st.session_state.prod_mode:
        # SambaNova Cloud
        if st.session_state.llm_api == 'sncloud':
            api_variables = {
                'SAMBANOVA_API_BASE': st.session_state.SAMBANOVA_API_BASE,
                'SAMBANOVA_API_KEY': st.session_state.SAMBANOVA_API_KEY,
            }
        else:
            raise Exception('Only sncloud supported.')
    else:
        api_variables = {}

    return api_variables


def plot_dataframe_summary(df_req_info: pd.DataFrame) -> Figure:
    """
    Plots a throughput summary across all batch sizes

    Args:
        df_req_info (pd.DataFrame): The DataFrame containing the data to plot.

    Returns:
        fig (go.Figure): The plotly figure container
    """
    df_req_summary = (
        df_req_info.groupby('batch_size_used')[
            [
                'server_output_token_per_s_per_request',
                'client_output_token_per_s_per_request',
            ]
        ]
        .mean()
        .reset_index()
    ).rename(
        columns={
            'server_output_token_per_s_per_request': 'server_output_token_per_s_mean',
            'client_output_token_per_s_per_request': 'client_output_token_per_s_mean',
        }
    )
    df_req_summary['server_throughput_token_per_s'] = (
        df_req_summary['server_output_token_per_s_mean'] * df_req_summary['batch_size_used']
    )
    df_req_summary['client_throughput_token_per_s'] = (
        df_req_summary['client_output_token_per_s_mean'] * df_req_summary['batch_size_used']
    )
    df_req_summary.rename(
        columns={
            'batch_size_used': 'Batch size',
            'server_throughput_token_per_s': 'Server',
            'client_throughput_token_per_s': 'Client',
        },
        inplace=True,
    )
    df_melted = pd.melt(
        df_req_summary,
        id_vars='Batch size',
        value_vars=['Server', 'Client'],
        var_name='Side type',
        value_name='Total output throughput (tokens per second)',
    )

    df_melted['Total output throughput (tokens per second)'] = df_melted[
        'Total output throughput (tokens per second)'
    ].round(2)

    df_melted['Batch size'] = [str(x) for x in df_melted['Batch size']]
    fig = px.bar(
        df_melted,
        x='Batch size',
        y='Total output throughput (tokens per second)',
        color='Side type',
        barmode='group',
        color_discrete_sequence=['#325c8c', '#ee7625'],
        text='Total output throughput (tokens per second)',
    )

    fig.update_traces(textposition='outside')  # Set text position outside bars

    fig.update_layout(
        title_text='Total output throughput per batch size',
        template='plotly_dark',
    )
    return fig


def plot_client_vs_server_barplots(
    df_user: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    legend_labels: List[str],
    title: str,
    ylabel: str,
    xlabel: str,
    batching_exposed: bool,
    colors: Optional[List[str]] = None,
) -> Figure:
    """
    Plots bar plots for client vs server metrics from a DataFrame.

    Args:
        df_user (pd.DataFrame): The DataFrame containing the data to plot.
        x_col (str): The column name to be used as the x-axis.
        y_cols (List[str]): A list of column names to be used as the y-axis.
        legend_labels (List[str]): Human-readable labels for each grouping in y_cols.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        xlabel (str): The label for the x-axis.
        batching_exposed (bool): boolean identifying if batching was exposed.

    Returns:
        fig (go.Figure): The plotly figure container
    """
    colors = colors or ['#325c8c', '#ee7625']
    value_vars = y_cols
    title_text = title
    yaxis_title = ylabel
    xaxis_title = xlabel if batching_exposed else ''

    df_melted = df_user.melt(
        id_vars=[x_col],
        value_vars=value_vars,
        var_name='Metric',
        value_name='Value',
    )
    xgroups = [str(x) for x in sorted(pd.unique(df_melted[x_col]))]
    df_melted[x_col] = [str(x) for x in df_melted[x_col]]

    valsl = {}
    valsr = {}
    for i in xgroups:
        maskl = (df_melted['Metric'] == value_vars[0]) & (df_melted[x_col] == i)
        valsl[i] = np.percentile(df_melted['Value'][maskl], [5, 50, 95])
        # Only compute right values if we have two metrics
        if len(value_vars) > 1:
            maskr = (df_melted['Metric'] == value_vars[1]) & (df_melted[x_col] == i)
            valsr[i] = np.percentile(df_melted['Value'][maskr], [5, 50, 95])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=xgroups,
            y=[0 for _ in xgroups],
            base=[round(valsl[i][1], 2) for i in xgroups],
            customdata=[legend_labels[0] for _ in xgroups],
            marker={'color': colors[0], 'line': {'color': colors[0], 'width': 2}},
            offsetgroup=0,
            legendgroup=legend_labels[0],
            name=legend_labels[0],
            showlegend=False,
            hovertemplate='<extra></extra><b>%{customdata}</b> median: %{base:.2f}',
            text=[round(valsl[i][1], 2) for i in xgroups],
            textposition='outside',
        )
    )
    fig.add_trace(
        go.Bar(
            x=xgroups,
            y=[valsl[i][2] - valsl[i][0] for i in xgroups],
            base=[valsl[i][0] for i in xgroups],
            customdata=[valsl[i][2] for i in xgroups],
            marker={'color': colors[0]},
            opacity=0.5,
            offsetgroup=0,
            legendgroup=legend_labels[0],
            name=legend_labels[0],
            hovertemplate='<extra></extra>5–95 pctile range: %{base:.2f}–%{customdata:.2f}',
        )
    )
    # Only add right metric bars if we have two metrics
    if len(value_vars) > 1:
        fig.add_trace(
            go.Bar(
                x=xgroups,
                y=[0 for _ in xgroups],
                base=[round(valsr[i][1], 2) for i in xgroups],
                customdata=[legend_labels[1] for _ in xgroups],
                marker={'color': colors[1], 'line': {'color': colors[1], 'width': 2}},
                offsetgroup=1,
                legendgroup=legend_labels[1],
                name=legend_labels[1],
                showlegend=False,
                hovertemplate='<extra></extra><b>%{customdata}</b> median: %{base:.2f}',
                text=[round(valsr[i][1], 2) for i in xgroups],
                textposition='outside',
            )
        )
        fig.add_trace(
            go.Bar(
                x=xgroups,
                y=[valsr[i][2] - valsr[i][0] for i in xgroups],
                base=[valsr[i][0] for i in xgroups],
                customdata=[valsr[i][2] for i in xgroups],
                marker={'color': colors[1]},
                opacity=0.5,
                offsetgroup=1,
                legendgroup=legend_labels[1],
                name=legend_labels[1],
                hovertemplate='<extra></extra>5–95 pctile range: %{base:.2f}–%{customdata:.2f}',
            )
        )

    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        barmode='group',
        template='plotly_dark',
        hovermode='x unified',
    )

    fig.update_xaxes(hoverformat='foo', showticklabels=batching_exposed)

    return fig


def plot_requests_gantt_chart(df_user: pd.DataFrame) -> Figure:
    """
    Plots a Gantt chart of response timings across all requests

    Args:
        df_user (pd.DataFrame): The DataFrame containing the data to plot.

    Returns:
        fig (go.Figure): The plotly figure container
    """
    requests = df_user.index + 1

    # Normalize timestamps to start at 0 for relative comparison
    # Convert start_time to datetime and find the minimum
    start_times = pd.to_datetime(df_user['start_time'])
    min_start_time = start_times.min()

    # Calculate relative start times in seconds from the first request
    relative_start_times_s = (start_times - min_start_time).dt.total_seconds()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=requests,
            x=df_user['client_ttft_s'],
            base=relative_start_times_s,
            name='TTFT',
            orientation='h',
            marker_color='#ee7625',
        )
    )
    fig.add_trace(
        go.Bar(
            y=requests,
            x=df_user['client_end_to_end_latency_s'],
            base=relative_start_times_s,
            name='End-to-end latency',
            orientation='h',
            marker_color='#325c8c',
        )
    )
    for i in range(0, len(df_user.index), 2):
        fig.add_hrect(y0=i + 0.5, y1=i + 1.5, line_width=0, fillcolor='grey', opacity=0.1)
    fig.update_xaxes(
        tickformat='.2f',
        hoverformat='.3f',
    )
    fig.update_layout(
        title_text='LLM requests across time',
        xaxis_title='Relative time (seconds from start)',
        yaxis_title='Request index',
        template='plotly_dark',
    )
    return fig


def get_vllm_summary_metrics(vllm_result_file_path: str) -> Dict[str, Any]:
    """
    Extract summary metrics from vLLM raw result file.

    Args:
        vllm_result_file_path (str): Path to the vLLM raw result JSON file.

    Returns:
        Dict[str, Any]: Dictionary containing summary metrics.
    """
    import json

    with open(vllm_result_file_path, 'r') as f:
        vllm_results = json.load(f)

    duration = vllm_results.get('duration', 0)
    completed = vllm_results.get('completed', 0)
    total_input_tokens = vllm_results.get('total_input_tokens', 0)
    total_output_tokens = vllm_results.get('total_output_tokens', 0)

    # Calculate throughput metrics
    request_throughput = completed / duration if duration > 0 else 0
    output_throughput = total_output_tokens / duration if duration > 0 else 0
    total_token_throughput = (total_input_tokens + total_output_tokens) / duration if duration > 0 else 0

    # Additional metrics
    mean_ttft_ms = vllm_results.get('mean_ttft_ms', 0)
    median_ttft_ms = vllm_results.get('median_ttft_ms', 0)
    mean_tpot_ms = vllm_results.get('mean_tpot_ms', 0)
    median_tpot_ms = vllm_results.get('median_tpot_ms', 0)
    mean_itl_ms = vllm_results.get('mean_itl_ms', 0)
    median_itl_ms = vllm_results.get('median_itl_ms', 0)

    return {
        'duration': duration,
        'completed_requests': completed,
        'failed_requests': vllm_results.get('failed', 0),
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'request_throughput': request_throughput,
        'output_throughput': output_throughput,
        'total_token_throughput': total_token_throughput,
        'mean_ttft_ms': mean_ttft_ms,
        'median_ttft_ms': median_ttft_ms,
        'mean_tpot_ms': mean_tpot_ms,
        'median_tpot_ms': median_tpot_ms,
        'mean_itl_ms': mean_itl_ms,
        'median_itl_ms': median_itl_ms,
    }


def calculate_kit_summary_metrics(df_individual: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary metrics from kit individual responses DataFrame.

    Args:
        df_individual (pd.DataFrame): DataFrame with individual request responses.

    Returns:
        Dict[str, Any]: Dictionary containing summary metrics.
    """
    # Filter out failed requests
    valid_df = df_individual[df_individual['error_code'].isnull()]

    # Calculate duration from timestamps
    start_times = pd.to_datetime(valid_df['start_time'])
    end_times = pd.to_datetime(valid_df['end_time'])
    duration = (end_times.max() - start_times.min()).total_seconds()

    # Count completed and failed requests
    completed = len(valid_df)
    failed = len(df_individual) - completed

    # Calculate token totals
    total_input_tokens = valid_df['number_input_tokens'].sum()
    total_output_tokens = valid_df['number_output_tokens'].sum()

    # Calculate throughput metrics
    request_throughput = completed / duration if duration > 0 else 0
    output_throughput = total_output_tokens / duration if duration > 0 else 0
    total_token_throughput = (total_input_tokens + total_output_tokens) / duration if duration > 0 else 0

    # Additional metrics (convert to milliseconds for consistency with vLLM)
    mean_ttft_ms = valid_df['client_ttft_s'].mean() * 1000
    median_ttft_ms = valid_df['client_ttft_s'].median() * 1000
    mean_e2e_latency_ms = valid_df['client_end_to_end_latency_s'].mean() * 1000
    median_e2e_latency_ms = valid_df['client_end_to_end_latency_s'].median() * 1000
    mean_output_throughput_per_request = valid_df['client_output_token_per_s_per_request'].mean()
    median_output_throughput_per_request = valid_df['client_output_token_per_s_per_request'].median()

    # Add ITL metrics with fallback for backward compatibility
    if 'client_mean_inter_token_latency_s' in valid_df.columns:
        mean_itl_ms = valid_df['client_mean_inter_token_latency_s'].mean() * 1000
        median_itl_ms = valid_df['client_mean_inter_token_latency_s'].median() * 1000
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


def display_summary_metrics_comparison(kit_metrics: Dict[str, Any], vllm_metrics: Dict[str, Any]) -> None:
    """
    Display summary metrics comparison between Kit and vLLM in Streamlit.

    Args:
        kit_metrics (Dict[str, Any]): Summary metrics from Kit benchmark.
        vllm_metrics (Dict[str, Any]): Summary metrics from vLLM benchmark.
    """
    st.markdown('### Summary Metrics Comparison')

    # Create comparison DataFrame
    comparison_data = {
        'Metric': [
            'Duration (s)',
            'Completed Requests',
            'Failed Requests',
            'Total Input Tokens',
            'Total Output Tokens',
            'Request Throughput (req/s)',
            'Output Throughput (tokens/s)',
            'Total Token Throughput (tokens/s)',
            'Mean TTFT (ms)',
            'Median TTFT (ms)',
            'Mean ITL (ms)',
            'Median ITL (ms)',
        ],
        'Kit': [
            f"{kit_metrics['duration']:.2f}",
            kit_metrics['completed_requests'],
            kit_metrics['failed_requests'],
            kit_metrics['total_input_tokens'],
            kit_metrics['total_output_tokens'],
            f"{kit_metrics['request_throughput']:.4f}",
            f"{kit_metrics['output_throughput']:.2f}",
            f"{kit_metrics['total_token_throughput']:.2f}",
            f"{kit_metrics['mean_ttft_ms']:.2f}",
            f"{kit_metrics['median_ttft_ms']:.2f}",
            f"{kit_metrics['mean_itl_ms']:.2f}" if kit_metrics['mean_itl_ms'] is not None else 'N/A',
            f"{kit_metrics['median_itl_ms']:.2f}" if kit_metrics['median_itl_ms'] is not None else 'N/A',
        ],
        'vLLM': [
            f"{vllm_metrics['duration']:.2f}",
            vllm_metrics['completed_requests'],
            vllm_metrics['failed_requests'],
            vllm_metrics['total_input_tokens'],
            vllm_metrics['total_output_tokens'],
            f"{vllm_metrics['request_throughput']:.4f}",
            f"{vllm_metrics['output_throughput']:.2f}",
            f"{vllm_metrics['total_token_throughput']:.2f}",
            f"{vllm_metrics['mean_ttft_ms']:.2f}",
            f"{vllm_metrics['median_ttft_ms']:.2f}",
            f"{vllm_metrics['mean_itl_ms']:.2f}",
            f"{vllm_metrics['median_itl_ms']:.2f}",
        ],
    }

    df_comparison = pd.DataFrame(comparison_data)

    # Display the comparison table
    st.dataframe(
        df_comparison,
        use_container_width=True,
        hide_index=True,
    )


def calculate_sum_itl_per_request(df: pd.DataFrame, itl_column: str) -> pd.Series:
    """
    Calculate sum of all ITLs for each request.

    Args:
        df (pd.DataFrame): DataFrame containing the metrics
        itl_column (str): Column name containing inter-token latencies list

    Returns:
        pd.Series: Sum of ITLs in seconds per request
    """
    def calc_sum_itl(row):
        itls = row[itl_column]
        if itls is None or not isinstance(itls, list) or len(itls) == 0:
            return None
        return sum(itls)

    return df.apply(calc_sum_itl, axis=1)


def calculate_tpot_per_request(df: pd.DataFrame, itl_column: str, output_tokens_column: str) -> pd.Series:
    """
    Calculate Time Per Output Token (TPOT) for each request using ITLs and output tokens.

    TPOT is calculated as: sum(ITLs[1:]) / (output_tokens - 1)
    This excludes the first ITL which typically includes TTFT overhead.

    Args:
        df (pd.DataFrame): DataFrame containing the metrics
        itl_column (str): Column name containing inter-token latencies list
        output_tokens_column (str): Column name containing output token counts

    Returns:
        pd.Series: TPOT values in seconds per token
    """
    def calc_tpot(row):
        itls = row[itl_column]
        output_tokens = row[output_tokens_column]

        # Handle cases where ITL data might be missing or invalid
        if itls is None or not isinstance(itls, list) or len(itls) == 0:
            return None
        if output_tokens is None or output_tokens <= 1:
            return None

        # Calculate TPOT: sum of ITLs (after first) divided by number of tokens (after first)
        if len(itls) > 1:
            # Exclude first ITL chunk which includes TTFT, divide by actual tokens generated
            tpot = sum(itls[1:]) / (output_tokens - 1)
        else:
            # Single chunk case - divide by output tokens
            tpot = itls[0] / output_tokens if output_tokens > 0 else None

        return tpot

    return df.apply(calc_tpot, axis=1)


def plot_per_request_comparison(
    kit_df: pd.DataFrame,
    vllm_df: pd.DataFrame,
    kit_metric_column: str,
    vllm_metric_column: str,
    metric_name: str,
    y_axis_label: str,
    title: str,
) -> Figure:
    """
    Create a per-request comparison plot between Kit and vLLM for a given metric.

    Args:
        kit_df (pd.DataFrame): Kit benchmark results DataFrame
        vllm_df (pd.DataFrame): vLLM benchmark results DataFrame
        kit_metric_column (str): Column name in Kit DataFrame
        vllm_metric_column (str): Column name in vLLM DataFrame
        metric_name (str): Display name for the metric
        y_axis_label (str): Y-axis label
        title (str): Plot title

    Returns:
        Figure: Plotly figure object
    """
    # Filter valid data (non-null, non-error)
    kit_valid = kit_df[kit_df['error_code'].isna()].copy()
    vllm_valid = vllm_df[vllm_df['error_code'].isna()].copy()

    # Get metric values
    kit_values = kit_valid[kit_metric_column].dropna()
    vllm_values = vllm_valid[vllm_metric_column].dropna()

    # Create request indices
    kit_requests = list(range(1, len(kit_values) + 1))
    vllm_requests = list(range(1, len(vllm_values) + 1))

    # Create figure
    fig = go.Figure()

    # Add Kit trace
    fig.add_trace(
        go.Scatter(
            x=kit_requests,
            y=kit_values,
            mode='lines+markers',
            name='Kit',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
        )
    )

    # Add vLLM trace
    fig.add_trace(
        go.Scatter(
            x=vllm_requests,
            y=vllm_values,
            mode='lines+markers',
            name='vLLM',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=6),
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Request Number',
        yaxis_title=y_axis_label,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
    )

    return fig


def plot_ttft_per_request_comparison(kit_df: pd.DataFrame, vllm_df: pd.DataFrame) -> Figure:
    """
    Create a per-request TTFT comparison plot between Kit and vLLM.

    Args:
        kit_df (pd.DataFrame): Kit benchmark results DataFrame
        vllm_df (pd.DataFrame): vLLM benchmark results DataFrame

    Returns:
        Figure: Plotly figure object
    """
    return plot_per_request_comparison(
        kit_df=kit_df,
        vllm_df=vllm_df,
        kit_metric_column='client_ttft_s',
        vllm_metric_column='client_ttft_s',  # vLLM now uses same column name
        metric_name='TTFT',
        y_axis_label='Time to First Token (seconds)',
        title='TTFT Comparison: Kit vs vLLM (Per Request)',
    )


def plot_itl_per_request_comparison(kit_df: pd.DataFrame, vllm_df: pd.DataFrame) -> Figure:
    """
    Create a per-request sum ITL comparison plot between Kit and vLLM.

    Args:
        kit_df (pd.DataFrame): Kit benchmark results DataFrame
        vllm_df (pd.DataFrame): vLLM benchmark results DataFrame

    Returns:
        Figure: Plotly figure object
    """
    # Calculate sum of ITLs for both Kit and vLLM
    kit_df_copy = kit_df.copy()
    vllm_df_copy = vllm_df.copy()

    kit_df_copy['sum_itl_s'] = calculate_sum_itl_per_request(kit_df_copy, 'client_inter_token_latencies_s')
    vllm_df_copy['sum_itl_s'] = calculate_sum_itl_per_request(vllm_df_copy, 'client_inter_token_latencies_s')

    return plot_per_request_comparison(
        kit_df=kit_df_copy,
        vllm_df=vllm_df_copy,
        kit_metric_column='sum_itl_s',
        vllm_metric_column='sum_itl_s',
        metric_name='Sum ITL',
        y_axis_label='Sum of Inter-Token Latencies (seconds)',
        title='Sum ITL Comparison: Kit vs vLLM (Per Request)',
    )


def plot_tpot_per_request_comparison(kit_df: pd.DataFrame, vllm_df: pd.DataFrame) -> Figure:
    """
    Create a per-request TPOT comparison plot between Kit and vLLM.
    TPOT is calculated from ITLs and output tokens for each request.

    Args:
        kit_df (pd.DataFrame): Kit benchmark results DataFrame
        vllm_df (pd.DataFrame): vLLM benchmark results DataFrame

    Returns:
        Figure: Plotly figure object
    """
    # Calculate TPOT for both Kit and vLLM
    kit_df_copy = kit_df.copy()
    vllm_df_copy = vllm_df.copy()

    kit_df_copy['tpot_s'] = calculate_tpot_per_request(
        kit_df_copy, 'client_inter_token_latencies_s', 'number_output_tokens'
    )
    vllm_df_copy['tpot_s'] = calculate_tpot_per_request(
        vllm_df_copy, 'client_inter_token_latencies_s', 'number_output_tokens'  # vLLM now uses same column names
    )

    return plot_per_request_comparison(
        kit_df=kit_df_copy,
        vllm_df=vllm_df_copy,
        kit_metric_column='tpot_s',
        vllm_metric_column='tpot_s',
        metric_name='TPOT',
        y_axis_label='Time Per Output Token (seconds)',
        title='TPOT Comparison: Kit vs vLLM (Per Request)',
    )
