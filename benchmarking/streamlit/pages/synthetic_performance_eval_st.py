# current_dir = os.path.dirname(os.path.abspath('..'))
# kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
# repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

# sys.path.append(kit_dir)
# sys.path.append(repo_dir)

import warnings

import pandas as pd
import streamlit as st
import yaml
from st_pages import hide_pages

from benchmarking.src.performance_evaluation import SyntheticPerformanceEvaluator
from benchmarking.streamlit.streamlit_utils import (
    APP_PAGES,
    LLM_API_OPTIONS,
    find_pages_to_hide,
    plot_client_vs_server_barplots,
    plot_dataframe_summary,
    plot_requests_gantt_chart,
    set_api_variables,
)

warnings.filterwarnings('ignore')

CONFIG_PATH = './config.yaml'
with open(CONFIG_PATH) as file:
    st.session_state.config = yaml.safe_load(file)
    st.session_state.prod_mode = st.session_state.config['prod_mode']
    st.session_state.pages_to_show = st.session_state.config['pages_to_show']


def _initialize_session_variables() -> None:
    # Initialize llm
    if 'llm' not in st.session_state:
        st.session_state.llm = None

    # Initialize llm params
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
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = None


def _run_performance_evaluation() -> pd.DataFrame:
    """Runs the performance evaluation process for different number of concurrent requests that will run in parallel.

    Returns:
        pd.DataFrame: Dataframe with metrics for each number of concurrent requests.
    """

    results_path = './data/results/llmperf'

    api_variables = set_api_variables()

    # Call benchmarking process
    performance_evaluator = SyntheticPerformanceEvaluator(
        model_name=st.session_state.llm,
        results_dir=results_path,
        num_concurrent_requests=st.session_state.number_concurrent_requests,
        timeout=st.session_state.timeout,
        llm_api=st.session_state.llm_api,
        api_variables=api_variables,
        user_metadata={'model_idx': 0},
    )

    performance_evaluator.run_benchmark(
        num_input_tokens=st.session_state.input_tokens,
        num_output_tokens=st.session_state.output_tokens,
        num_requests=st.session_state.number_requests,
        sampling_params={},
    )

    # Read generated json and output formatted results
    df_user = pd.read_json(performance_evaluator.individual_responses_file_path)
    df_user['concurrent_requests'] = st.session_state.number_concurrent_requests
    valid_df = df_user[df_user['error_code'].isnull()]

    # For non-batching endpoints, batch_size_used will be 1
    if valid_df['batch_size_used'].isnull().all():
        valid_df['batch_size_used'] = 1

    return valid_df


def main() -> None:
    if st.session_state.prod_mode:
        pages_to_hide = find_pages_to_hide()
        pages_to_hide.append(APP_PAGES['setup']['page_label'])
        hide_pages(pages_to_hide)
    else:
        hide_pages([APP_PAGES['setup']['page_label']])

    st.title(':orange[SambaNova] Synthetic Performance Evaluation')
    st.markdown(
        """This performance evaluation assesses the following LLM's performance metrics using concurrent processes.
        _client represent the metrics computed from the client-side and _server represents the metrics computed
        from the server-side."""
    )
    st.markdown(
        """**Time to first token (TTFT):** This metric is driven by the time required to process the prompt and then
        generate the first output token."""
    )
    st.markdown('**E2E Latency:** TTFT + (Time per Output Token) * (the number of tokens to be generated - 1)')
    st.markdown(
        """**Throughput:** Number of output tokens per second across all concurrency requests. Client metric is
        calculated as *Number of Output Tokens / (E2E Latency - TTFT)*"""
    )
    st.markdown('**Total Throughput:** Number of total output tokens per batch and per second')

    with st.sidebar:
        st.title('Configuration')
        st.markdown('**Modify the following parameters before running the process**')

        llm_model = st.text_input(
            'Model Name',
            value='llama3-8b',
            help='Look at your model card in SambaStudio and introduce the same name of the model/expert here.',
        )
        st.session_state.llm = f'{llm_model}'

        if st.session_state.prod_mode:
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
        else:
            st.session_state.llm_api = st.selectbox(
                'API type', options=list(LLM_API_OPTIONS.keys()), format_func=lambda x: LLM_API_OPTIONS[x], index=0
            )

        st.session_state.input_tokens = st.number_input(
            'Number of input tokens', min_value=50, max_value=2000, value=1000, step=1
        )

        st.session_state.output_tokens = st.number_input(
            'Number of output tokens', min_value=50, max_value=2000, value=1000, step=1
        )

        st.session_state.number_requests = st.number_input(
            'Number of total requests', min_value=10, max_value=1000, value=10, step=1
        )

        st.session_state.number_concurrent_requests = st.number_input(
            'Number of concurrent requests', min_value=1, max_value=100, value=1, step=1
        )

        st.session_state.timeout = st.number_input('Timeout', min_value=60, max_value=1800, value=600, step=1)

        sidebar_option = st.sidebar.button('Run!')

        if st.session_state.prod_mode:
            if st.button('Back to Setup'):
                st.session_state.setup_complete = False
                st.switch_page('app.py')

    if sidebar_option:
        st.session_state.mp_events.input_submitted('synthetic_performance_evaluation ')
        st.toast('Performance evaluation processing now. It should take few minutes.')
        with st.spinner('Processing'):
            try:
                df_req_info = _run_performance_evaluation()

                st.subheader('Performance metrics plots')
                expected_output_tokens = st.session_state.output_tokens
                generated_output_tokens = df_req_info.server_number_output_tokens.unique()[0]
                if not pd.isnull(generated_output_tokens):
                    st.markdown(
                        f"""Difference between expected output tokens ({expected_output_tokens}) and generated output
                        tokens ({generated_output_tokens}) is {abs(expected_output_tokens-generated_output_tokens)}
                            token(s)"""
                    )

                st.plotly_chart(
                    plot_client_vs_server_barplots(
                        df_req_info,
                        'batch_size_used',
                        ['server_ttft_s', 'client_ttft_s'],
                        ['Server', 'Client'],
                        'Distribution of Time to First Token (TTFT) by batch size',
                        'TTFT (s), per request',
                        'Batch size',
                    )
                )
                st.plotly_chart(
                    plot_client_vs_server_barplots(
                        df_req_info,
                        'batch_size_used',
                        ['server_end_to_end_latency_s', 'client_end_to_end_latency_s'],
                        ['Server', 'Client'],
                        'Distribution of end-to-end latency by batch size',
                        'Latency (s), per request',
                        'Batch size',
                    )
                )
                st.plotly_chart(
                    plot_client_vs_server_barplots(
                        df_req_info,
                        'batch_size_used',
                        [
                            'server_output_token_per_s_per_request',
                            'client_output_token_per_s_per_request',
                        ],
                        ['Server', 'Client'],
                        'Distribution of output throughput by batch size',
                        'Tokens per second, per request',
                        'Batch size',
                    )
                )
                # Compute total throughput per batch
                st.plotly_chart(plot_dataframe_summary(df_req_info))
                st.plotly_chart(plot_requests_gantt_chart(df_req_info))

            except Exception as e:
                st.error(f'Error: {e}.')


if __name__ == '__main__':
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )

    _initialize_session_variables()

    if st.session_state.prod_mode:
        if st.session_state.setup_complete:
            main()
        else:
            st.switch_page('./app.py')
    else:
        main()
