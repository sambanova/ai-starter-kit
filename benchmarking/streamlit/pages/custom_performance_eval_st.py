import os
import warnings
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from benchmarking.src.performance_evaluation import CustomPerformanceEvaluator
from benchmarking.streamlit.streamlit_utils import (
    LLM_API_OPTIONS,
    plot_client_vs_server_barplots,
    plot_dataframe_summary,
    plot_requests_gantt_chart,
    render_logo,
    render_title_icon,
    save_uploaded_file,
    set_api_variables,
    set_font,
    setup_credentials,
    update_progress_bar,
)

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

CONFIG_PATH = './config.yaml'
with open(CONFIG_PATH) as file:
    st.session_state.config = yaml.safe_load(file)
    st.session_state.prod_mode = st.session_state.config['prod_mode']
    st.session_state.pages_to_show = st.session_state.config['pages_to_show']


def _initialize_sesion_variables() -> None:
    # Initialize llm
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'llm_api' not in st.session_state:
        st.session_state.llm_api = None

    # Initialize llm params
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'file_path' not in st.session_state:
        st.session_state.file_path = None
    if 'do_sample' not in st.session_state:
        st.session_state.do_sample = None
    if 'max_tokens_to_generate' not in st.session_state:
        st.session_state.max_tokens_to_generate = None
    if 'repetition_penalty' not in st.session_state:
        st.session_state.repetition_penalty = None
    if 'temperature' not in st.session_state:
        st.session_state.temperature = None
    if 'top_k' not in st.session_state:
        st.session_state.top_k = None
    if 'top_p' not in st.session_state:
        st.session_state.top_p = None
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = None

    # Additional initialization
    if 'run_button' in st.session_state and st.session_state.run_button == True:
        st.session_state.running = True
    else:
        st.session_state.running = False
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



def _run_custom_performance_evaluation(progress_bar: Any = None) -> pd.DataFrame:
    """Runs custom performance evaluation

    Returns:
        pd.DataFrame: valid dataframe containing benchmark results
    """

    api_variables = set_api_variables()

    results_path = './data/results/llmperf'
    st.session_state.performance_evaluator = CustomPerformanceEvaluator(
        model_name=st.session_state.llm,
        results_dir=results_path,
        num_concurrent_requests=st.session_state.number_concurrent_requests,
        timeout=st.session_state.timeout,
        input_file_path=st.session_state.file_path,
        save_response_texts=st.session_state.save_llm_responses,
        llm_api=st.session_state.llm_api,
        api_variables=api_variables,
        config=st.session_state.config,
    )

    # set generic max tokens parameter
    sampling_params = {'max_tokens_to_generate': st.session_state.max_tokens}
    st.session_state.performance_evaluator.run_benchmark(sampling_params=sampling_params, progress_bar=progress_bar)

    df_user = pd.read_json(st.session_state.performance_evaluator.individual_responses_file_path)
    df_user['concurrent_user'] = st.session_state.performance_evaluator.num_concurrent_requests
    valid_df = df_user[df_user['error_code'].isnull()]

    # For non-batching endpoints, batching_exposed will be False
    st.session_state.batching_exposed = True
    if valid_df['batch_size_used'].isnull().all():
        st.session_state.batching_exposed = False

    return valid_df


def main() -> None:

    set_font()

    render_title_icon('Custom Performance Evaluation', os.path.join(repo_dir, 'images', 'benchmark_icon.png'))
    st.markdown(
        'Here you can select a custom dataset that you want to benchmark performance with. Note that with models that \
          support dynamic batching, you are limited to the number of cpus available on your machine to send concurrent \
              requests.'
    )

    with st.sidebar:
        # Set up credentials and API variables
        setup_credentials()

        render_logo()
        ##################
        # File Selection #
        ##################
        st.title('File Selection')
        st.session_state.uploaded_file = st.file_uploader(
            'Upload JSON File', type='jsonl', disabled=st.session_state.running
        )
        st.session_state.file_path = save_uploaded_file(internal_save_path='data/custom_input_files')

        #########################
        # Runtime Configuration #
        #########################
        st.title('Configuration')

        st.text_input(
            'Model Name',
            value='Meta-Llama-3.3-70B-Instruct',
            key='llm',
            help='Look at your model card and introduce the same name \
                of the model/expert',
            disabled=st.session_state.running,
        )

        if st.session_state.llm_api == 'sncloud':
            st.selectbox(
                'API type',
                options=list(LLM_API_OPTIONS.keys()),
                format_func=lambda x: LLM_API_OPTIONS[x],
                index=0,
                disabled=True,
            )

        st.number_input(
            'Num Concurrent Requests',
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            key='number_concurrent_requests',
            disabled=st.session_state.running,
        )

        st.number_input(
            'Timeout', min_value=60, max_value=1800, value=600, step=1, key='timeout', disabled=st.session_state.running
        )

        st.toggle(
            'Save LLM Responses',
            value=False,
            key='save_llm_responses',
            help='Toggle on if you want to save the llm responses to an output JSONL file',
            disabled=st.session_state.running,
        )

        #####################
        # Tuning Parameters #
        #####################
        st.title('Tuning Parameters')

        st.number_input(
            'Max Output Tokens',
            min_value=1,
            max_value=2048,
            value=256,
            step=1,
            key='max_tokens',
            disabled=st.session_state.running,
        )

        # TODO: Add more tuning params below (temperature, top_k, etc.)

        job_submitted = st.sidebar.button(
            'Run!', 
            disabled=st.session_state.running, 
            key='run_button', 
            type='primary', 
            width="stretch")

        sidebar_stop = st.sidebar.button(
            'Stop', 
            disabled=not st.session_state.running, 
            type='secondary', 
            width="stretch")

    if sidebar_stop:
        st.session_state.running = False
        st.session_state.performance_evaluator.stop_benchmark()

    if job_submitted:
        st.session_state.mp_events.input_submitted('custom_performance_evaluation ')
        st.toast(
            """Performance evaluation in progress. This could take a while depending on the dataset size and max tokens
              setting."""
        )
        with st.spinner('Processing'):
            st.session_state.progress_bar = st.progress(0)
            do_rerun = False
            try:
                st.session_state.df_req_info = _run_custom_performance_evaluation(update_progress_bar)
                st.session_state.running = False
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

    _initialize_sesion_variables()

    main()