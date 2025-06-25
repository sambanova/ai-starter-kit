import io
import json
import os
import warnings
import zipfile
from typing import Any, Dict

import pandas as pd
import streamlit as st
import yaml
from st_pages import hide_pages

from benchmarking.src.performance_evaluation import SyntheticPerformanceEvaluator
from benchmarking.streamlit.streamlit_utils import (
    APP_PAGES,
    LLM_API_OPTIONS,
    MULTIMODAL_IMAGE_SIZE_OPTIONS,
    PRIMARY_ST_STYLE,
    SECONDARY_ST_STYLE,
    find_pages_to_hide,
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
    if 'df_req_info' not in st.session_state:
        st.session_state.df_req_info = None
    if 'batching_exposed' not in st.session_state:
        st.session_state.batching_exposed = None
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


def main() -> None:
    hide_pages([APP_PAGES['main']['page_label']])

    set_font()
    if st.session_state.prod_mode:
        pages_to_hide = find_pages_to_hide()
        pages_to_hide.append(APP_PAGES['main']['page_label'])
        hide_pages(pages_to_hide)

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

        llm_model = st.text_input(
            'Model Name',
            value='Meta-Llama-3.3-70B-Instruct',
            help='If using SambaStudio, look at your model card and introduce the same name \
                of the model/expert here following the Readme.',
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
        elif st.session_state.llm_api == 'sambastudio':
            st.selectbox(
                'API type',
                options=list(LLM_API_OPTIONS.keys()),
                format_func=lambda x: LLM_API_OPTIONS[x],
                index=1,
                disabled=True,
            )

        st.session_state.multimodal_image_size = st.selectbox(
            'Multimodal image size',
            options=list(MULTIMODAL_IMAGE_SIZE_OPTIONS.keys()),
            format_func=lambda x: MULTIMODAL_IMAGE_SIZE_OPTIONS[x],
            index=0,
            disabled=st.session_state.running or st.session_state.optional_download,
            help='Select the pre-set image size for multimodal models. \
                Small: 500x500, Medium: 1024x1024, Large: 2000x2000. Select N/A for non-multimodal models.',
        )

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

        st.session_state.timeout = st.number_input(
            'Timeout',
            min_value=60,
            max_value=1800,
            value=600,
            step=1,
            disabled=st.session_state.running or st.session_state.optional_download,
        )

        st.session_state.running = st.sidebar.button(
            'Run!',
            disabled=st.session_state.running or st.session_state.optional_download,
            key='run_button',
            type='primary',
        )

        if st.session_state.optional_download:
            st.sidebar.download_button(
                label='Download Results',
                data=st.session_state.zip_buffer,
                file_name='output_files.zip',
                mime='application/zip',
            )
        else:
            st.sidebar.download_button(
                label='Download Results',
                data='',
                disabled=not st.session_state.running or not st.session_state.optional_download,
            )

        # Disable stop button if app is not running and download button is not available
        sidebar_stop = st.sidebar.button(
            'Stop',
            disabled=(not st.session_state.running) and (not st.session_state.optional_download),
            type='secondary',
        )

    if sidebar_stop:
        st.session_state.optional_download = False
        st.session_state.zip_buffer = None
        st.session_state.performance_evaluator.stop_benchmark()

        st.rerun()

    if st.session_state.running:
        st.session_state.mp_events.input_submitted('synthetic_performance_evaluation ')
        st.toast('Performance evaluation processing now. It should take few minutes.')
        with st.spinner('Processing'):
            st.session_state.progress_bar = st.progress(0)
            do_rerun = False
            try:
                st.session_state.df_req_info = _run_performance_evaluation(update_progress_bar)

                json_data = read_performance_evaluation_output_files()

                st.session_state.zip_buffer = create_zip(json_data)
                st.session_state.running = False
                st.session_state.optional_download = True

                # workareound to avoid rerun within try block
                do_rerun = True
            except Exception as e:
                st.error(f'Error:\n{e}.')
                # Cleaning df results in case of error
                st.session_state.df_req_info = None
            if do_rerun:
                st.rerun()

    if st.session_state.df_req_info is not None:
        st.subheader('Performance metrics plots')
        expected_output_tokens = st.session_state.output_tokens
        generated_output_tokens = st.session_state.df_req_info.server_number_output_tokens.unique()[0]
        if not pd.isnull(generated_output_tokens):
            st.markdown(
                f"""Difference between expected output tokens ({expected_output_tokens}) and generated output
                tokens ({generated_output_tokens}) is {abs(expected_output_tokens - generated_output_tokens)}
                    token(s)"""
            )

        by_batch_size_suffix = ' by batch size' if st.session_state.batching_exposed else ''
        st.plotly_chart(
            plot_client_vs_server_barplots(
                st.session_state.df_req_info,
                'batch_size_used',
                ['server_ttft_s', 'client_ttft_s'],
                ['Server', 'Client'],
                'Distribution of Time to First Token (TTFT)' + by_batch_size_suffix,
                'TTFT (s), per request',
                'Batch size',
                st.session_state.batching_exposed,
            )
        )
        st.plotly_chart(
            plot_client_vs_server_barplots(
                st.session_state.df_req_info,
                'batch_size_used',
                ['server_end_to_end_latency_s', 'client_end_to_end_latency_s'],
                ['Server', 'Client'],
                'Distribution of end-to-end latency' + by_batch_size_suffix,
                'Latency (s), per request',
                'Batch size',
                st.session_state.batching_exposed,
            )
        )
        st.plotly_chart(
            plot_client_vs_server_barplots(
                st.session_state.df_req_info,
                'batch_size_used',
                [
                    'server_output_token_per_s_per_request',
                    'client_output_token_per_s_per_request',
                ],
                ['Server', 'Client'],
                'Distribution of output throughput' + by_batch_size_suffix,
                'Tokens per second, per request',
                'Batch size',
                st.session_state.batching_exposed,
            )
        )
        # Compute total throughput per batch
        if st.session_state.batching_exposed:
            st.plotly_chart(plot_dataframe_summary(st.session_state.df_req_info))
        st.plotly_chart(plot_requests_gantt_chart(st.session_state.df_req_info))

        # Once results are given, reset running state and ending threads just in case.
        sidebar_stop = True


if __name__ == '__main__':
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
    )

    # Defining styles
    st.markdown(PRIMARY_ST_STYLE, unsafe_allow_html=True)
    st.markdown(SECONDARY_ST_STYLE, unsafe_allow_html=True)

    _initialize_session_variables()

    main()