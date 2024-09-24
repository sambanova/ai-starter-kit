import sys

sys.path.append('../')
sys.path.append('./src')
sys.path.append('./streamlit')

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from st_pages import Page, show_pages
from streamlit_utils import plot_client_vs_server_barplots, plot_dataframe_summary, plot_requests_gantt_chart

from benchmarking.src.performance_evaluation import SyntheticPerformanceEvaluator

warnings.filterwarnings('ignore')

LLM_API_OPTIONS = ["SambaNova Cloud", "SambaStudio"]
LLM_API_CODENAMES = ["sncloud", "sambastudio"]

@st.cache_data
def _init() -> None:
    load_dotenv('../.env', override=True)


def _run_performance_evaluation() -> pd.DataFrame:
    """Runs the performance evaluation process for different number of workers that will run in parallel.
    Each worker will run num_requests_per_worker requests.

    Returns:
        pd.DataFrame: Dataframe with metrics for each number of workers.
    """

    results_path = './data/results/llmperf'

    # Call benchmarking process
    api_dict = {LLM_API_OPTIONS[i]: LLM_API_CODENAMES[i] for i in range(len(LLM_API_OPTIONS))}
    performance_evaluator = SyntheticPerformanceEvaluator(
        model_name=st.session_state.llm,
        results_dir=results_path,
        num_workers=st.session_state.number_concurrent_workers,
        timeout=st.session_state.timeout,
        llm_api=api_dict[st.session_state.llm_api],
    )

    performance_evaluator.run_benchmark(
        num_input_tokens=st.session_state.input_tokens,
        num_output_tokens=st.session_state.output_tokens,
        num_requests=st.session_state.number_requests,
        sampling_params={},
    )

    # Read generated json and output formatted results
    df_user = pd.read_json(performance_evaluator.individual_responses_file_path)
    df_user['concurrent_user'] = st.session_state.number_concurrent_workers
    valid_df = df_user[(df_user['error_code'] != '')]

    # For non-batching endpoints, batch_size_used will be 1
    if valid_df['batch_size_used'].isnull().all():
        valid_df['batch_size_used'] = 1

    return valid_df


def _initialize_session_variables() -> None:

    if "llm" not in st.session_state:
        st.session_state.llm = None

    # Initialize llm params
    if 'input_tokens' not in st.session_state:
        st.session_state.input_tokens = None
    if 'output_tokens' not in st.session_state:
        st.session_state.output_tokens = None
    if 'number_requests' not in st.session_state:
        st.session_state.number_requests = None
    if 'number_concurrent_workers' not in st.session_state:
        st.session_state.number_concurrent_workers = None
    if 'timeout' not in st.session_state:
        st.session_state.timeout = None
    if 'llm_api' not in st.session_state:
        st.session_state.llm_api = None


def main() -> None:
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )

    show_pages(
        [
            Page('streamlit/app.py', 'Synthetic Performance Evaluation'),
            Page(
                'streamlit/pages/custom_performance_eval_st.py',
                'Custom Performance Evaluation',
            ),
            Page('streamlit/pages/chat_performance_st.py', 'Performance on Chat'),
        ]
    )

    _init()
    _initialize_session_variables()

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

        st.session_state.llm_api = st.selectbox('API type', options=LLM_API_OPTIONS)

        st.session_state.input_tokens = st.number_input(
            'Number of input tokens', min_value=50, max_value=2000, value=1000, step=1
        )

        st.session_state.output_tokens = st.number_input(
            'Number of output tokens', min_value=50, max_value=2000, value=1000, step=1
        )

        st.session_state.number_requests = st.number_input(
            'Number of total requests', min_value=10, max_value=1000, value=10, step=1
        )

        st.session_state.number_concurrent_workers = st.number_input(
            'Number of concurrent workers', min_value=1, max_value=100, value=1, step=1
        )

        st.session_state.timeout = st.number_input('Timeout', min_value=60, max_value=1800, value=600, step=1)

        sidebar_option = st.sidebar.button('Run!')

    if sidebar_option:
        st.toast('Performance evaluation processing now. It should take few minutes.')
        with st.spinner('Processing'):
            try:
                df_req_info = _run_performance_evaluation()

                st.subheader('Performance metrics plots')
                expected_output_tokens = st.session_state.output_tokens
                generated_output_tokens = df_req_info.server_number_output_tokens.unique()[0]
                if not pd.isnull(generated_output_tokens):
                    st.markdown(
                        f"Difference between expected output tokens ({expected_output_tokens}) and generated output tokens ({generated_output_tokens}) is {abs(expected_output_tokens-generated_output_tokens)} token(s)"
                    )

                st.plotly_chart(
                    plot_client_vs_server_barplots(
                        df_req_info,
                        "batch_size_used",
                        ["server_ttft_s", "client_ttft_s"],
                        ["Server", "Client"],
                        "Distribution of Time to First Token (TTFT) by batch size",
                        "TTFT (s), per request",
                        "Batch size",
                    )
                )
                st.plotly_chart(
                    plot_client_vs_server_barplots(
                        df_req_info,
                        "batch_size_used",
                        ["server_end_to_end_latency_s", "client_end_to_end_latency_s"],
                        ["Server", "Client"],
                        "Distribution of end-to-end latency by batch size",
                        "Latency (s), per request",
                        "Batch size",
                    )
                )
                st.plotly_chart(
                    plot_client_vs_server_barplots(
                        df_req_info,
                        "batch_size_used",
                        [
                            "server_output_token_per_s_per_request",
                            "client_output_token_per_s_per_request",
                        ],
                        ["Server", "Client"],
                        "Distribution of throughput by batch size",
                        "Tokens per second, per request",
                        "Batch size",
                    )
                )
                # Compute total throughput per batch
                st.plotly_chart(
                    plot_dataframe_summary(df_req_info)
                )
                st.plotly_chart(
                    plot_requests_gantt_chart(df_req_info)
                )

            except Exception as e:
                st.error(f'Error: {e}.')


if __name__ == '__main__':
    main()
