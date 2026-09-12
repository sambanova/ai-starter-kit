import base64
import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.graph_objs import Figure

from benchmarking.benchmarking_utils import DEFAULT_MODEL
from benchmarking.benchmarking_tools.kit.src.tool_runners import TOOL_RUNNERS, ToolRunResult
from benchmarking.utils import SAMBANOVA_API_BASE
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

LLM_API_OPTIONS = {'sncloud': 'SambaNova Cloud'}
MULTIMODAL_IMAGE_SIZE_OPTIONS = {'na': 'N/A', 'small': 'Small', 'medium': 'Medium', 'large': 'Large'}
QPS_DISTRIBUTION_OPTIONS = {'constant': 'Constant', 'exponential': 'Exponential'}
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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_available_models(api_base: str, api_key: str) -> Optional[List[str]]:
    """Fetch model IDs from GET {api_base}/models. Returns None on any failure."""
    try:
        url = api_base.rstrip('/') + '/models'
        resp = requests.get(url, headers={'Authorization': f'Bearer {api_key}'}, timeout=10)
        resp.raise_for_status()
        models = [m['id'] for m in resp.json().get('data', [])]
        return sorted(models) if models else None
    except Exception:
        return None


def model_selector_widget(disabled: bool = False) -> str:
    """Dropdown populated from /models endpoint; falls back to text input on failure."""
    if are_credentials_set():
        if st.session_state.prod_mode:
            api_base = st.session_state.SAMBANOVA_API_BASE
            api_key = st.session_state.SAMBANOVA_API_KEY
        else:
            api_base = os.environ.get('SAMBANOVA_API_BASE', SAMBANOVA_API_BASE)
            api_key = os.environ.get('SAMBANOVA_API_KEY', '')
        models = fetch_available_models(api_base, api_key) if api_key else None
    else:
        models = None

    default = DEFAULT_MODEL
    help_text = 'Select or type the model name'

    if models:
        current = st.session_state.get('llm') or default
        idx = models.index(current) if current in models else 0
        return st.selectbox('Model Name', options=models, index=idx, disabled=disabled, help=help_text)
    else:
        return st.text_input(
            'Model Name', value=st.session_state.get('llm') or default, disabled=disabled, help=help_text
        )


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


def create_progress_callback(progress_bar: Any) -> Callable[[int, int, str], None]:
    """Builds a `progress_cb` for `ToolRunner.run_*` bound to one specific `st.progress()` widget.

    Each tool run should get its own progress bar (created inside its own `st.status()` block)
    rather than all tools sharing/overwriting a single `st.session_state.progress_bar` -- that
    previously meant one tool's progress bar reset/hid another's when running more than one tool
    per pass.

    Args:
        progress_bar: an `st.progress()` widget to update.
    """

    def _update(step: int, total_steps: int, phase: str = 'Running requests') -> None:
        fraction = step / total_steps if total_steps else 0
        progress_bar.progress(value=fraction, text=f'{phase}: {step}/{total_steps}')

    return _update


def create_log_callback(
    placeholder: Any, log_lines: List[str], tool_display_name: str, max_lines: int = 200
) -> Callable[[str], None]:
    """Builds a `log_cb` for `ToolRunner.run_*` that streams a subprocess tool's (vLLM/aiperf)
    raw stdout into a live-updating code block, so the user sees real progress even though these
    tools don't expose the same granular, per-request progress bar Kit does.

    Args:
        placeholder: an `st.empty()` (or similar) container to re-render into on every line.
        log_lines: shared, growing list of all lines seen so far this run (across every tool) --
            passed in rather than created here so a whole run's log survives past any single
            tool's call.
        tool_display_name: prefixed onto each line so lines from different tools stay distinguishable
            when more than one is selected.
        max_lines: only the most recent lines are rendered, so a long-running benchmark doesn't
            grow the DOM/log view unboundedly.
    """

    def _append(line: str) -> None:
        log_lines.append(f'[{tool_display_name}] {line}')
        placeholder.code('\n'.join(log_lines[-max_lines:]), language='bash')

    return _append


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


def calculate_tool_summary_metrics(df_individual: pd.DataFrame, summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate summary metrics for one tool's run, combining its individual-responses DataFrame
    with its already-standardized *_summary.json.

    Request/token counts and TTFT/ITL stats are computed from df_individual -- tool-agnostic,
    since every tool's *_individual_responses.json shares the same columns by construction (the
    RequestMetric schema in kit/src/schemas.py). Duration and throughput are read from `summary`
    instead: `start_time`/`end_time` per request are Kit-only fields (vLLM/aiperf's converters
    never populate them), so deriving duration from individual-response timestamps silently gives
    NaN/0 for vLLM/aiperf. Each tool's own *_summary.json already carries this correctly, just
    under different field names depending on what that tool natively reports:
      - benchmark_duration_s: set directly by aiperf/vLLM (from their own native duration/
        benchmark_duration fields); Kit doesn't report a native duration, so it's derived from
        Kit's own results_num_completed_requests_per_min instead (same derivation
        benchmarking_bundles/synthetic_performance_eval_script.py's BundleSummaryCalculator uses).
      - request_throughput: natively reported by vLLM/aiperf; Kit has no equivalent field, so it's
        derived from results_num_completed_requests_per_min (Kit's own native rate) instead.
      - client_total_output_throughput: populated by all three (Kit natively; vLLM/aiperf's
        converters map their own output-throughput field onto it) -- read directly, though note
        it's flattened as `results_client_total_output_throughput` (grouped with the other
        `results_*` scalar fields), not a bare top-level key.

    Args:
        df_individual: DataFrame loaded from this tool's *_individual_responses.json.
        summary: Dict loaded from this tool's *_summary.json (the flattened on-disk shape).

    Returns:
        Dictionary of summary metrics, comparable across tools.
    """
    valid_df = df_individual[df_individual['error_code'].isnull()]

    completed = len(valid_df)
    failed = len(df_individual) - completed

    total_input_tokens = valid_df['number_input_tokens'].sum()
    total_output_tokens = valid_df['number_output_tokens'].sum()

    completed_per_min = summary.get('results_num_completed_requests_per_min')

    duration = summary.get('benchmark_duration_s')
    if duration is None and completed_per_min:
        duration = completed / (completed_per_min / 60.0)

    request_throughput = summary.get('request_throughput')
    if request_throughput is None and completed_per_min:
        request_throughput = completed_per_min / 60.0
    if request_throughput is None and duration:
        request_throughput = completed / duration

    # client_total_output_throughput is flattened with a `results_` prefix by
    # BenchmarkSummary.to_legacy_flat_dict() (grouped with num_completed_requests_per_min et al.),
    # not as a bare top-level key -- confirmed live against a real vLLM-converted summary.
    output_throughput = summary.get('results_client_total_output_throughput')
    if output_throughput is None and duration:
        output_throughput = total_output_tokens / duration

    total_token_throughput = (total_input_tokens + total_output_tokens) / duration if duration else None

    mean_ttft_ms = valid_df['client_ttft_s'].mean() * 1000
    median_ttft_ms = valid_df['client_ttft_s'].median() * 1000

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
        'mean_itl_ms': mean_itl_ms,
        'median_itl_ms': median_itl_ms,
    }


def display_summary_metrics_comparison(tool_metrics: Dict[str, Dict[str, Any]]) -> None:
    """
    Display an N-way summary metrics comparison table in Streamlit, one column per tool.

    Args:
        tool_metrics: Mapping of tool display name -> calculate_tool_summary_metrics(...) output.
    """
    st.markdown('### Summary Metrics Comparison')

    rows = [
        ('Duration (s)', 'duration', '{:.2f}'),
        ('Completed Requests', 'completed_requests', '{}'),
        ('Failed Requests', 'failed_requests', '{}'),
        ('Total Input Tokens', 'total_input_tokens', '{}'),
        ('Total Output Tokens', 'total_output_tokens', '{}'),
        ('Request Throughput (req/s)', 'request_throughput', '{:.4f}'),
        ('Output Throughput (tokens/s)', 'output_throughput', '{:.2f}'),
        ('Total Token Throughput (tokens/s)', 'total_token_throughput', '{:.2f}'),
        ('Mean TTFT (ms)', 'mean_ttft_ms', '{:.2f}'),
        ('Median TTFT (ms)', 'median_ttft_ms', '{:.2f}'),
        ('Mean ITL (ms)', 'mean_itl_ms', '{:.2f}'),
        ('Median ITL (ms)', 'median_itl_ms', '{:.2f}'),
    ]

    comparison_data: Dict[str, List[Any]] = {'Metric': [label for label, _, _ in rows]}
    for tool_display_name, metrics in tool_metrics.items():
        column = []
        for _, key, fmt in rows:
            value = metrics.get(key)
            column.append(fmt.format(value) if value is not None else 'N/A')
        comparison_data[tool_display_name] = column

    st.dataframe(pd.DataFrame(comparison_data), width='stretch', hide_index=True)


def display_single_tool_results(
    df_req_info: pd.DataFrame,
    batching_exposed: bool,
    expected_output_tokens: int,
    tool_display_name: str,
    show_server_metrics: bool,
) -> None:
    """Display benchmark results plots for a single tool's run.

    Args:
        df_req_info: DataFrame with request information (valid rows only).
        batching_exposed: Whether batching info is available for this run.
        expected_output_tokens: Expected number of output tokens.
        tool_display_name: Label for the benchmark (e.g. 'Kit', 'vLLM', 'aiperf').
        show_server_metrics: Whether to show server-side metrics (kit-only) alongside client ones.
    """
    st.markdown('**Performance metrics plots**')

    if df_req_info.empty:
        st.warning('No successful requests to display. All requests failed.')
        return

    unique_vals = df_req_info.server_number_output_tokens.dropna().unique() if show_server_metrics else []
    if len(unique_vals) > 0 and not pd.isnull(unique_vals[0]):
        generated_output_tokens = unique_vals[0]
        st.markdown(
            f"""Difference between expected output tokens ({expected_output_tokens}) and generated output
            tokens ({generated_output_tokens}) is {abs(expected_output_tokens - generated_output_tokens)}
                token(s)"""
        )

    by_batch_size_suffix = ' by batch size' if batching_exposed else ''

    if not show_server_metrics:
        metrics_ttft = ['client_ttft_s']
        labels_ttft = ['Client']
    else:
        metrics_ttft = ['server_ttft_s', 'client_ttft_s']
        labels_ttft = ['Server', 'Client']
        metrics_latency = ['server_end_to_end_latency_s', 'client_end_to_end_latency_s']
        labels_latency = ['Server', 'Client']
        metrics_throughput = ['server_output_token_per_s_per_request', 'client_output_token_per_s_per_request']
        labels_throughput = ['Server', 'Client']

    st.plotly_chart(
        plot_client_vs_server_barplots(
            df_req_info,
            'batch_size_used',
            metrics_ttft,
            labels_ttft,
            f'{tool_display_name}: Distribution of Time to First Token (TTFT)' + by_batch_size_suffix,
            'TTFT (s), per request',
            'Batch size',
            batching_exposed,
            colors=['#ee7625'] if not show_server_metrics else None,
        ),
        width='stretch',
    )
    if show_server_metrics:
        st.plotly_chart(
            plot_client_vs_server_barplots(
                df_req_info,
                'batch_size_used',
                metrics_latency,
                labels_latency,
                f'{tool_display_name}: Distribution of end-to-end latency' + by_batch_size_suffix,
                'Latency (s), per request',
                'Batch size',
                batching_exposed,
            ),
            width='stretch',
        )
        st.plotly_chart(
            plot_client_vs_server_barplots(
                df_req_info,
                'batch_size_used',
                metrics_throughput,
                labels_throughput,
                f'{tool_display_name}: Distribution of output throughput' + by_batch_size_suffix,
                'Tokens per second, per request',
                'Batch size',
                batching_exposed,
            ),
            width='stretch',
        )
    df_itl = df_req_info[['batch_size_used', 'client_mean_inter_token_latency_s']].copy()
    df_itl['client_mean_inter_token_latency_ms'] = df_itl['client_mean_inter_token_latency_s'] * 1000
    st.plotly_chart(
        plot_client_vs_server_barplots(
            df_itl,
            'batch_size_used',
            ['client_mean_inter_token_latency_ms'],
            ['Client'],
            f'{tool_display_name}: Distribution of Mean Inter-Token Latency (ITL)' + by_batch_size_suffix,
            'Mean ITL (ms), per request',
            'Batch size',
            batching_exposed,
            colors=['#ee7625'],
        ),
        width='stretch',
    )
    if batching_exposed:
        st.plotly_chart(plot_dataframe_summary(df_req_info), width='stretch')
    if show_server_metrics:
        st.plotly_chart(plot_requests_gantt_chart(df_req_info), width='stretch')


def render_tool_logs(results: Dict[str, ToolRunResult], tool_logs: Optional[Dict[str, List[str]]]) -> None:
    """Renders each tool's captured subprocess log (vLLM/aiperf -- Kit has none) as its own
    collapsed expander, so the live logs shown while a run is in progress remain available (but
    out of the way) once results are displayed -- for both a single-tool run and a side-by-side
    comparison alike.

    Args:
        results: Mapping of tool key -> that tool's ToolRunResult, used only to decide which
            tools' logs to show and in what order.
        tool_logs: Mapping of tool key -> that tool's captured log lines (persisted in
            `st.session_state` across the rerun that follows a run's completion, since the live
            `st.status`/log widgets themselves only exist while `st.session_state.running` is
            True). Tools with no captured lines (Kit, or a tool that produced no output before
            failing) are skipped.
    """
    if not tool_logs:
        return
    for tool_name in results:
        lines = tool_logs.get(tool_name)
        if not lines:
            continue
        with st.expander(f'{TOOL_RUNNERS[tool_name].display_name} log', expanded=False):
            st.code('\n'.join(lines), language='bash')


def render_multi_tool_results(
    results: Dict[str, ToolRunResult],
    expected_output_tokens: int,
    tool_logs: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Render this run's results. A single tool gets the full detailed view (as always); running
    more than one tool shows ONLY the summary comparison table and a TTFT distribution comparison
    -- nothing else -- to keep the multi-tool view simple. Shared by the synthetic, custom, and
    real-workload Streamlit pages.

    Args:
        results: Mapping of tool key ('kit'/'vllm'/'aiperf') -> that tool's ToolRunResult.
        expected_output_tokens: Expected number of output tokens (for the token-count sanity note).
        tool_logs: Mapping of tool key -> that tool's captured log lines, shown as collapsed
            per-tool expanders above the results (see `render_tool_logs`). Omitted entirely if
            not passed (e.g. no run happened this session, only a stale `tool_results` reload).
    """
    render_tool_logs(results, tool_logs)

    valid_dfs: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for tool_name, result in results.items():
        df = pd.read_json(result.individual_responses_file_path)
        valid_dfs[tool_name] = df[df['error_code'].isnull()]
        with open(result.summary_file_path) as f:
            summaries[tool_name] = json.load(f)

    if len(results) == 1:
        tool_name = next(iter(results))
        runner = TOOL_RUNNERS[tool_name]
        valid_df = valid_dfs[tool_name]
        st.subheader(f'{runner.display_name} Benchmark Results')
        batching_exposed = (
            runner.supports_batching_info
            and not valid_df.empty
            and not valid_df['batch_size_used'].isnull().all()
        )
        display_single_tool_results(
            valid_df, batching_exposed, expected_output_tokens, runner.display_name, runner.supports_server_metrics
        )
        return

    st.header('Side-by-Side Comparison')

    tool_metrics = {
        TOOL_RUNNERS[t].display_name: calculate_tool_summary_metrics(valid_dfs[t], summaries[t])
        for t in results
        if not valid_dfs[t].empty
    }
    if tool_metrics:
        display_summary_metrics_comparison(tool_metrics)

    st.markdown('### TTFT Distribution Comparison')
    cols = st.columns(len(results))
    for col, tool_name in zip(cols, results.keys()):
        runner = TOOL_RUNNERS[tool_name]
        valid_df = valid_dfs[tool_name]
        with col:
            if valid_df.empty:
                st.warning(f'No successful {runner.display_name} requests to display.')
                continue
            if runner.supports_server_metrics:
                metrics_ttft = ['server_ttft_s', 'client_ttft_s']
                labels_ttft = ['Server', 'Client']
            else:
                metrics_ttft = ['client_ttft_s']
                labels_ttft = ['Client']
            batching_exposed = runner.supports_batching_info and not valid_df['batch_size_used'].isnull().all()
            st.plotly_chart(
                plot_client_vs_server_barplots(
                    valid_df,
                    'batch_size_used',
                    metrics_ttft,
                    labels_ttft,
                    f'{runner.display_name}: Distribution of TTFT',
                    'TTFT (s), per request',
                    'Batch size',
                    batching_exposed,
                    colors=None if runner.supports_server_metrics else ['#ee7625'],
                ),
                width='stretch',
            )
