import io
import json
import os
import warnings
import zipfile
from typing import Any, Dict

import pandas as pd
import streamlit as st
import yaml

from benchmarking.src.performance_evaluation import SyntheticPerformanceEvaluator
from benchmarking.src.vllm_benchmark import VLLMBenchmarkExecutor
from benchmarking.streamlit.streamlit_utils import (
    LLM_API_OPTIONS,
    MULTIMODAL_IMAGE_SIZE_OPTIONS,
    calculate_kit_summary_metrics,
    display_summary_metrics_comparison,
    get_vllm_summary_metrics,
    plot_client_vs_server_barplots,
    plot_dataframe_summary,
    plot_requests_gantt_chart,
    render_logo,
    render_title_icon,
    set_api_variables,
    set_font,
    setup_credentials,
    update_progress_bar,
)
from benchmarking.utils import CONFIG_PATH

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

with open(CONFIG_PATH) as file:
    st.session_state.config = yaml.safe_load(file)
    st.session_state.prod_mode = st.session_state.config['prod_mode']
    st.session_state.pages_to_show = st.session_state.config['pages_to_show']


def _initialize_session_variables() -> None:
    # Clear results when navigating from a different page
    if st.session_state.get('current_page') != 'synthetic':
        st.session_state.df_req_info = None
        st.session_state.df_req_info_vllm = None
        st.session_state.batching_exposed = None
        st.session_state.batching_exposed_vllm = None
        st.session_state.performance_evaluator = None
        st.session_state.vllm_evaluator = None
        st.session_state.optional_download = False
        st.session_state.zip_buffer = None
        st.session_state.current_page = 'synthetic'

    # Initialize llm
    if 'llm' not in st.session_state:
        st.session_state.llm = None

    # Initialize llm params
    if 'multimodal_image_size' not in st.session_state:
        st.session_state.multimodal_image_size = None
    if 'input_tokens' not in st.session_state:
        st.session_state.input_tokens = None
    if 'output_tokens' not in st.session_state:
        st.session_state.output_tokens = None
    if 'number_requests' not in st.session_state:
        st.session_state.number_requests = None
    if 'number_concurrent_requests' not in st.session_state:
        st.session_state.number_concurrent_requests = None
    if 'timeout' not in st.session_state:
        st.session_state.timeout = None
    if 'llm_api' not in st.session_state:
        st.session_state.llm_api = None

    # Benchmark mode selection
    if 'benchmark_mode' not in st.session_state:
        st.session_state.benchmark_mode = 'kit'
    if 'previous_benchmark_mode' not in st.session_state:
        st.session_state.previous_benchmark_mode = st.session_state.benchmark_mode

    # Additional initializations
    if 'optional_download' not in st.session_state:
        st.session_state.optional_download = False
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'run_button' in st.session_state and st.session_state.run_button == True:
        st.session_state.running = True
    else:
        st.session_state.running = False
    if 'zip_buffer' not in st.session_state:
        st.session_state.zip_buffer = None
    if 'performance_evaluator' not in st.session_state:
        st.session_state.performance_evaluator = None
    if 'vllm_evaluator' not in st.session_state:
        st.session_state.vllm_evaluator = None
    if 'df_req_info' not in st.session_state:
        st.session_state.df_req_info = None
    if 'df_req_info_vllm' not in st.session_state:
        st.session_state.df_req_info_vllm = None
    if 'batching_exposed' not in st.session_state:
        st.session_state.batching_exposed = None
    if 'batching_exposed_vllm' not in st.session_state:
        st.session_state.batching_exposed_vllm = None
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = None
    if 'progress_bar' not in st.session_state:
        st.session_state.progress_bar = None
    if 'mp_events' not in st.session_state:
        st.switch_page('app.py')


def read_performance_evaluation_output_files() -> Dict[str, Any]:
    """Reads performance evaluation output files and returns dictionary with each file name and content.

    Returns:
        Dict[str, Any]: dictionary with file names and contents.
    """

    # Get individual file information
    individual_file_name = st.session_state.performance_evaluator.individual_responses_file_path.split('/')[-1]
    with open(st.session_state.performance_evaluator.individual_responses_file_path, 'r') as f:
        individual_file_content = json.loads(f.read())

    # Get summmary file information
    summary_file_name = st.session_state.performance_evaluator.summary_file_path.split('/')[-1]
    with open(st.session_state.performance_evaluator.summary_file_path, 'r') as f:
        summary_file_content = json.loads(f.read())

    # Make output dictionary
    json_data = {individual_file_name: individual_file_content, summary_file_name: summary_file_content}

    return json_data


def create_zip(json_data: Dict[str, Any]) -> io.BytesIO:
    """Creates a zip file out of a JSON data in a dictionary

    Args:
        json_data (Dict[str, Any]): Data in JSON to be zipped.

    Returns:
        io.BytesIO: zip buffer containing all data in json_data
    """

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, data in json_data.items():
            json_content = json.dumps(data, indent=4)  # Convert dictionary to JSON string
            zip_file.writestr(filename, json_content)  # Write JSON content to ZIP
    zip_buffer.seek(0)
    return zip_buffer


def _run_performance_evaluation(progress_bar: Any = None) -> pd.DataFrame:
    """Runs the performance evaluation process for different number of concurrent requests that will run in parallel.

    Returns:
        pd.DataFrame: Dataframe with metrics for each number of concurrent requests.
    """

    results_path = './data/results/llmperf'

    api_variables = set_api_variables()

    # Call benchmarking process
    st.session_state.performance_evaluator = SyntheticPerformanceEvaluator(
        model_name=st.session_state.llm,
        results_dir=results_path,
        multimodal_image_size=st.session_state.multimodal_image_size,
        num_concurrent_requests=st.session_state.number_concurrent_requests,
        timeout=st.session_state.timeout,
        llm_api=st.session_state.llm_api,
        api_variables=api_variables,
        user_metadata={'model_idx': 0},
        config=st.session_state.config,
    )

    st.session_state.performance_evaluator.run_benchmark(
        num_input_tokens=st.session_state.input_tokens,
        num_output_tokens=st.session_state.output_tokens,
        num_requests=st.session_state.number_requests,
        sampling_params={},
        progress_bar=progress_bar,
    )

    # Read generated json and output formatted results
    df_user = pd.read_json(st.session_state.performance_evaluator.individual_responses_file_path)
    df_user['concurrent_requests'] = st.session_state.number_concurrent_requests
    valid_df = df_user[df_user['error_code'].isnull()]

    # For non-batching endpoints, batching_exposed will be False
    st.session_state.batching_exposed = True
    if valid_df['batch_size_used'].isnull().all():
        st.session_state.batching_exposed = False

    return valid_df


def _run_vllm_benchmark(progress_bar: Any = None) -> pd.DataFrame:
    """Runs the vLLM benchmark process.

    Returns:
        pd.DataFrame: Dataframe with metrics for each request.
    """

    results_path = './data/results/vllm'

    # Get API variables for vLLM
    api_variables = set_api_variables()

    # Call vLLM benchmarking process
    st.session_state.vllm_evaluator = VLLMBenchmarkExecutor(
        model_name=st.session_state.llm,
        results_dir=results_path,
        timeout=st.session_state.timeout,
        user_metadata={'model_idx': 0},
    )

    st.session_state.vllm_evaluator.run_benchmark(
        num_input_tokens=st.session_state.input_tokens,
        num_output_tokens=st.session_state.output_tokens,
        num_requests=st.session_state.number_requests,
        num_concurrent_requests=st.session_state.number_concurrent_requests,
        api_base=api_variables.get('SAMBANOVA_API_BASE'),
        api_key=api_variables.get('SAMBANOVA_API_KEY'),
        progress_bar=progress_bar,
    )

    # Read generated json and output formatted results
    df_user = st.session_state.vllm_evaluator.get_results_dataframe()
    df_user['concurrent_requests'] = st.session_state.number_concurrent_requests
    valid_df = df_user[df_user['error_code'].isnull()]

    # vLLM doesn't expose batching info
    st.session_state.batching_exposed_vllm = False

    return valid_df


def _display_benchmark_results(
    df_req_info: pd.DataFrame,
    batching_exposed: bool,
    expected_output_tokens: int,
    label: str,
    is_vllm: bool = False
) -> None:
    """Display benchmark results plots for a single benchmark.

    Args:
        df_req_info: DataFrame with request information
        batching_exposed: Whether batching is exposed
        expected_output_tokens: Expected number of output tokens
        label: Label for the benchmark (e.g., 'Kit' or 'vLLM')
        is_vllm: Whether this is a vLLM benchmark (shows only client metrics)
    """
    st.markdown('**Performance metrics plots**')

    if df_req_info.empty:
        st.warning('No successful requests to display. All requests failed.')
        return

    # Check output tokens
    unique_vals = df_req_info.server_number_output_tokens.dropna().unique()
    if len(unique_vals) > 0 and not pd.isnull(unique_vals[0]):
        generated_output_tokens = unique_vals[0]
        st.markdown(
            f"""Difference between expected output tokens ({expected_output_tokens}) and generated output
            tokens ({generated_output_tokens}) is {abs(expected_output_tokens - generated_output_tokens)}
                token(s)"""
        )

    by_batch_size_suffix = ' by batch size' if batching_exposed else ''

    # For vLLM, only show client metrics since server metrics aren't logged
    if is_vllm:
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
            f'{label}: Distribution of Time to First Token (TTFT)' + by_batch_size_suffix,
            'TTFT (s), per request',
            'Batch size',
            batching_exposed,
            colors=['#ee7625'] if is_vllm else None,
        ),
        width='stretch'
    )
    if not is_vllm:
        st.plotly_chart(
            plot_client_vs_server_barplots(
                df_req_info,
                'batch_size_used',
                metrics_latency,
                labels_latency,
                f'{label}: Distribution of end-to-end latency' + by_batch_size_suffix,
                'Latency (s), per request',
                'Batch size',
                batching_exposed,
            ),
            width='stretch'
        )
        st.plotly_chart(
            plot_client_vs_server_barplots(
                df_req_info,
                'batch_size_used',
                metrics_throughput,
                labels_throughput,
                f'{label}: Distribution of output throughput' + by_batch_size_suffix,
                'Tokens per second, per request',
                'Batch size',
                batching_exposed,
            ),
            width='stretch'
        )
    df_itl = df_req_info[['batch_size_used', 'client_mean_inter_token_latency_s']].copy()
    df_itl['client_mean_inter_token_latency_ms'] = df_itl['client_mean_inter_token_latency_s'] * 1000
    itl_color = '#ee7625'
    st.plotly_chart(
        plot_client_vs_server_barplots(
            df_itl,
            'batch_size_used',
            ['client_mean_inter_token_latency_ms'],
            ['Client'],
            f'{label}: Distribution of Mean Inter-Token Latency (ITL)' + by_batch_size_suffix,
            'Mean ITL (ms), per request',
            'Batch size',
            batching_exposed,
            colors=[itl_color],
        ),
        width='stretch'
    )
    # Compute total throughput per batch
    if batching_exposed:
        st.plotly_chart(plot_dataframe_summary(df_req_info), width='stretch')
    if not is_vllm:
        st.plotly_chart(plot_requests_gantt_chart(df_req_info), width='stretch')


def main() -> None:
    set_font()

    render_title_icon('Synthetic Performance Evaluation', os.path.join(repo_dir, 'images', 'benchmark_icon.png'))

    st.markdown(
        """This performance evaluation assesses the following LLM's performance metrics using concurrent processes.
        _client represents the metrics computed from the client-side (includes queue and round-trip time 
        from host to server and back) and _server represents the metrics computed from the server-side."""
    )
    st.markdown(
        """**Time to first token (TTFT):** This metric is driven by the time required to process the prompt and then
        generate the first output token."""
    )
    st.markdown('**E2E Latency:** TTFT + (Time per Output Token) * (the number of tokens to be generated - 1)')
    st.markdown(
        """**Tokens/sec/request (Output Throughput)**: Number of output tokens generated per second per request 
        for a given batch-size. Client metric is calculated as *Number of Output Tokens / (E2E Latency - TTFT)*"""
    )
    st.markdown("""**Tokens/sec (Throughput)**: Total number of tokens generated per second for a given batch-size.""")

    with st.sidebar:
        # Set up credentials and API variables
        setup_credentials()

        render_logo()
        st.title('Configuration')
        st.markdown('**Modify the following parameters before running the process**')

        # Benchmarking mode selector
        st.session_state.benchmark_mode = st.radio(
            'Benchmarking Mode',
            options=['kit', 'vllm', 'both'],
            format_func=lambda x: {
                'kit': 'Kit Only',
                'vllm': 'vLLM Only',
                'both': 'Both (Side-by-Side Comparison)'
            }[x],
            help='Select which benchmark to run: Kit (current implementation), vLLM (vLLM benchmark serve), or Both for comparison',
            disabled=st.session_state.running or st.session_state.optional_download,
        )

        if st.session_state.benchmark_mode != st.session_state.previous_benchmark_mode:
            st.session_state.df_req_info = None
            st.session_state.df_req_info_vllm = None
            st.session_state.batching_exposed = None
            st.session_state.batching_exposed_vllm = None
            st.session_state.performance_evaluator = None
            st.session_state.vllm_evaluator = None
            st.session_state.optional_download = False
            st.session_state.zip_buffer = None
            st.session_state.previous_benchmark_mode = st.session_state.benchmark_mode

        st.divider()

        llm_model = st.text_input(
            'Model Name',
            value='Meta-Llama-3.3-70B-Instruct',
            help='Look at your model card and introduce the same name \
                of the model/expert',
            disabled=st.session_state.running or st.session_state.optional_download,
        )
        st.session_state.llm = f'{llm_model}'

        if st.session_state.llm_api == 'sncloud':
            st.selectbox(
                'API type',
                options=list(LLM_API_OPTIONS.keys()),
                format_func=lambda x: LLM_API_OPTIONS[x],
                index=0,
                disabled=True,
            )

        vllm_mode = st.session_state.benchmark_mode in ['vllm', 'both']
        st.session_state.multimodal_image_size = st.selectbox(
            'Multimodal image size',
            options=list(MULTIMODAL_IMAGE_SIZE_OPTIONS.keys()),
            format_func=lambda x: MULTIMODAL_IMAGE_SIZE_OPTIONS[x],
            index=0,
            disabled=st.session_state.running or st.session_state.optional_download or vllm_mode,
            help='Select the pre-set image size for multimodal models. '
                'Small: 500x500, Medium: 1024x1024, Large: 2000x2000. Select N/A for non-multimodal models. '
                'Not supported in vLLM mode.',
        )
        if vllm_mode:
            st.session_state.multimodal_image_size = 'na'
            st.caption('ℹ️ Multimodal image size is not supported in vLLM mode.')

        st.session_state.input_tokens = st.number_input(
            'Number of input tokens',
            min_value=50,
            max_value=10000,
            value=1000,
            step=1,
            disabled=st.session_state.running or st.session_state.optional_download,
        )

        st.session_state.output_tokens = st.number_input(
            'Number of output tokens',
            min_value=50,
            max_value=2000,
            value=1000,
            step=1,
            disabled=st.session_state.running or st.session_state.optional_download,
        )

        st.session_state.number_requests = st.number_input(
            'Number of total requests',
            min_value=1,
            max_value=2000,
            value=10,
            step=1,
            disabled=st.session_state.running or st.session_state.optional_download,
        )

        st.session_state.number_concurrent_requests = st.number_input(
            'Number of concurrent requests',
            min_value=1,
            max_value=2000,
            value=1,
            step=1,
            disabled=st.session_state.running or st.session_state.optional_download,
        )

        timeout_blocked = st.session_state.benchmark_mode in ['vllm', 'both']
        st.session_state.timeout = st.number_input(
            'Timeout',
            min_value=60,
            max_value=1800,
            value=600,
            step=1,
            disabled=st.session_state.running or st.session_state.optional_download or timeout_blocked,
            help='Number of seconds before program times out. Not supported in vLLM mode.',
        )
        if st.session_state.benchmark_mode == 'vllm':
            st.caption('ℹ️ Timeout is not supported in vLLM mode.')
        elif st.session_state.benchmark_mode == 'both':
            st.caption('ℹ️ Timeout is disabled — not supported by all frameworks in this mode.')

        st.session_state.running = st.sidebar.button(
            'Run!',
            disabled=st.session_state.running or st.session_state.optional_download,
            key='run_button',
            type='primary',
            width='stretch',
        )

        if st.session_state.optional_download:
            st.sidebar.download_button(
                label='Download Results',
                data=st.session_state.zip_buffer,
                file_name='output_files.zip',
                mime='application/zip',
                width='stretch',
            )
        else:
            st.sidebar.download_button(
                label='Download Results',
                data='',
                disabled=not st.session_state.running or not st.session_state.optional_download,
                width='stretch',
            )

        # Disable stop button if app is not running and download button is not available
        sidebar_stop = st.sidebar.button(
            'Stop',
            disabled=(not st.session_state.running) and (not st.session_state.optional_download),
            type='secondary',
            width='stretch',
        )

    if sidebar_stop:
        st.session_state.optional_download = False
        st.session_state.zip_buffer = None
        if st.session_state.performance_evaluator:
            st.session_state.performance_evaluator.stop_benchmark()
        if st.session_state.vllm_evaluator:
            st.session_state.vllm_evaluator.stop_benchmark()

        st.rerun()

    if st.session_state.running:
        st.session_state.mp_events.input_submitted('synthetic_performance_evaluation ')
        benchmark_mode_label = {
            'kit': 'Kit',
            'vllm': 'vLLM',
            'both': 'Kit and vLLM'
        }[st.session_state.benchmark_mode]
        st.toast(f'{benchmark_mode_label} performance evaluation processing now. It should take few minutes.')
        with st.spinner('Processing'):
            st.session_state.progress_bar = st.progress(0)
            do_rerun = False
            try:
                json_data = {}

                # Run benchmarks based on selected mode
                if st.session_state.benchmark_mode in ['kit', 'both']:
                    st.session_state.df_req_info = _run_performance_evaluation(update_progress_bar)
                    json_data.update(read_performance_evaluation_output_files())

                if st.session_state.benchmark_mode in ['vllm', 'both']:
                    # Reset progress bar for vLLM if running both
                    if st.session_state.benchmark_mode == 'both':
                        st.session_state.progress_bar.progress(0)
                        st.toast('Now running vLLM benchmark...')

                    st.session_state.df_req_info_vllm = _run_vllm_benchmark(update_progress_bar)

                    # Add vLLM results to json_data
                    if st.session_state.vllm_evaluator:
                        vllm_individual_name = st.session_state.vllm_evaluator.individual_responses_file_path.split('/')[-1]
                        with open(st.session_state.vllm_evaluator.individual_responses_file_path, 'r') as f:
                            vllm_individual_content = json.loads(f.read())

                        vllm_summary_name = st.session_state.vllm_evaluator.summary_file_path.split('/')[-1]
                        with open(st.session_state.vllm_evaluator.summary_file_path, 'r') as f:
                            vllm_summary_content = json.loads(f.read())

                        json_data[vllm_individual_name] = vllm_individual_content
                        json_data[vllm_summary_name] = vllm_summary_content

                st.session_state.zip_buffer = create_zip(json_data)
                st.session_state.running = False
                st.session_state.optional_download = True

                # workareound to avoid rerun within try block
                do_rerun = True
            except Exception as e:
                st.error(f'Error:\n{e}.')
                # Cleaning df results in case of error
                st.session_state.df_req_info = None
                st.session_state.df_req_info_vllm = None
            if do_rerun:
                st.rerun()

    # Display results based on benchmark mode
    if st.session_state.benchmark_mode == 'both' and st.session_state.df_req_info is not None and st.session_state.df_req_info_vllm is not None:
        # Side-by-side comparison
        st.header('Side-by-Side Comparison: Kit vs vLLM')

        # Summary metrics comparison table
        kit_metrics = calculate_kit_summary_metrics(
            pd.read_json(st.session_state.performance_evaluator.individual_responses_file_path)
        )
        vllm_metrics = get_vllm_summary_metrics(st.session_state.vllm_evaluator.result_file_path)
        display_summary_metrics_comparison(kit_metrics, vllm_metrics)

        st.markdown('### Metrics Distribution Comparison')

        # TTFT side by side
        col_kit, col_vllm = st.columns(2)
        with col_kit:
            st.plotly_chart(
                plot_client_vs_server_barplots(
                    st.session_state.df_req_info,
                    'batch_size_used',
                    ['server_ttft_s', 'client_ttft_s'],
                    ['Server', 'Client'],
                    'Kit: Distribution of TTFT',
                    'TTFT (s), per request',
                    'Batch size',
                    st.session_state.batching_exposed,
                ),
                width='stretch',
            )
        with col_vllm:
            st.plotly_chart(
                plot_client_vs_server_barplots(
                    st.session_state.df_req_info_vllm,
                    'batch_size_used',
                    ['client_ttft_s'],
                    ['Client'],
                    'vLLM: Distribution of TTFT',
                    'TTFT (s), per request',
                    'Batch size',
                    st.session_state.batching_exposed_vllm,
                    colors=['#ee7625'],
                ),
                width='stretch',
            )

        # ITL side by side (converted to ms)
        df_itl_kit = st.session_state.df_req_info[['batch_size_used', 'client_mean_inter_token_latency_s']].copy()
        df_itl_kit['client_mean_inter_token_latency_ms'] = df_itl_kit['client_mean_inter_token_latency_s'] * 1000
        df_itl_vllm = st.session_state.df_req_info_vllm[['batch_size_used', 'client_mean_inter_token_latency_s']].copy()
        df_itl_vllm['client_mean_inter_token_latency_ms'] = df_itl_vllm['client_mean_inter_token_latency_s'] * 1000

        col_kit, col_vllm = st.columns(2)
        with col_kit:
            st.plotly_chart(
                plot_client_vs_server_barplots(
                    df_itl_kit,
                    'batch_size_used',
                    ['client_mean_inter_token_latency_ms'],
                    ['Client'],
                    'Kit: Distribution of Mean ITL',
                    'Mean ITL (ms), per request',
                    'Batch size',
                    st.session_state.batching_exposed,
                    colors=['#ee7625'],
                ),
                width='stretch',
            )
        with col_vllm:
            st.plotly_chart(
                plot_client_vs_server_barplots(
                    df_itl_vllm,
                    'batch_size_used',
                    ['client_mean_inter_token_latency_ms'],
                    ['Client'],
                    'vLLM: Distribution of Mean ITL',
                    'Mean ITL (ms), per request',
                    'Batch size',
                    st.session_state.batching_exposed_vllm,
                    colors=['#ee7625'],
                ),
                width='stretch',
            )

    elif st.session_state.benchmark_mode == 'kit' and st.session_state.df_req_info is not None:
        st.subheader('Kit Benchmark Results')
        _display_benchmark_results(
            st.session_state.df_req_info,
            st.session_state.batching_exposed,
            st.session_state.output_tokens,
            'Kit',
            is_vllm=False
        )

    elif st.session_state.benchmark_mode == 'vllm' and st.session_state.df_req_info_vllm is not None:
        st.subheader('vLLM Benchmark Results')
        _display_benchmark_results(
            st.session_state.df_req_info_vllm,
            st.session_state.batching_exposed_vllm,
            st.session_state.output_tokens,
            'vLLM',
            is_vllm=True
        )


if __name__ == '__main__':
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'icon.svg'),
    )

    _initialize_session_variables()

    main()
