import warnings

import pandas as pd
import streamlit as st
import yaml
from st_pages import hide_pages

from benchmarking.src.performance_evaluation import CustomPerformanceEvaluator
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


def _initialize_sesion_variables() -> None:
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'perf_metrics_history' not in st.session_state:
        st.session_state.perf_metrics_history = []
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'llm_api' not in st.session_state:
        st.session_state.llm_api = None
    if 'chat_disabled' not in st.session_state:
        st.session_state.chat_disabled = True

    # Initialize llm params
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


def _run_custom_performance_evaluation() -> pd.DataFrame:
    """Runs custom performance evaluation

    Returns:
        pd.DataFrame: valid dataframe containing benchmark results
    """

    api_variables = set_api_variables()

    results_path = './data/results/llmperf'
    custom_performance_evaluator = CustomPerformanceEvaluator(
        model_name=st.session_state.llm,
        results_dir=results_path,
        num_concurrent_requests=st.session_state.number_concurrent_requests,
        timeout=st.session_state.timeout,
        input_file_path=st.session_state.file_path,
        save_response_texts=st.session_state.save_llm_responses,
        llm_api=st.session_state.llm_api,
        api_variables=api_variables,
    )

    # set generic max tokens parameter
    sampling_params = {'max_tokens_to_generate': st.session_state.max_tokens}
    custom_performance_evaluator.run_benchmark(
        sampling_params=sampling_params,
    )

    df_user = pd.read_json(custom_performance_evaluator.individual_responses_file_path)
    df_user['concurrent_user'] = custom_performance_evaluator.num_concurrent_requests
    valid_df = df_user[df_user['error_code'].isnull()]

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

    st.title(':orange[SambaNova] Custom Performance Evaluation')
    st.markdown(
        'Here you can select a custom dataset that you want to benchmark performance with. Note that with models that \
          support dynamic batching, you are limited to the number of cpus available on your machine to send concurrent \
              requests.'
    )

    with st.sidebar:
        ##################
        # File Selection #
        ##################
        st.title('File Selection')
        st.text_input('Full File Path', help='', key='file_path')  # TODO: Fill in help

        #########################
        # Runtime Configuration #
        #########################
        st.title('Configuration')

        st.text_input(
            'Model Name',
            value='llama3-8b',
            key='llm',
            help='Look at your model card in SambaStudio and introduce the same name of the model/expert here.',
        )

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

        st.number_input(
            'Num Concurrent Requests',
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            key='number_concurrent_requests',
        )

        st.number_input('Timeout', min_value=60, max_value=1800, value=600, step=1, key='timeout')

        st.toggle(
            'Save LLM Responses',
            value=False,
            key='save_llm_responses',
            help='Toggle on if you want to save the llm responses to an output JSONL file',
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
        )

        # TODO: Add more tuning params below (temperature, top_k, etc.)

        job_submitted = st.sidebar.button('Run!')

        if st.session_state.prod_mode:
            if st.button('Back to Setup'):
                st.session_state.setup_complete = False
                st.switch_page('app.py')

    if job_submitted:
        st.session_state.mp_events.input_submitted('custom_performance_evaluation ')
        st.toast(
            """Performance evaluation in progress. This could take a while depending on the dataset size and max tokens
              setting."""
        )
        with st.spinner('Processing'):
            try:
                results_df = _run_custom_performance_evaluation()

                st.subheader('Performance metrics plots')
                st.plotly_chart(
                    plot_client_vs_server_barplots(
                        results_df,
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
                        results_df,
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
                        results_df,
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
                st.plotly_chart(plot_dataframe_summary(results_df))
                st.plotly_chart(plot_requests_gantt_chart(results_df))

            except Exception as e:
                st.error(f'Error: {e}.')


if __name__ == '__main__':
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )

    _initialize_sesion_variables()

    if st.session_state.prod_mode:
        if st.session_state.setup_complete:
            main()
        else:
            st.switch_page('./app.py')
    else:
        main()
