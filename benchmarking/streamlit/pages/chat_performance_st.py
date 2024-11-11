import sys

import streamlit as st
import yaml

sys.path.append('../')

import warnings
from typing import Any, Dict

from st_pages import hide_pages

from benchmarking.src.chat_performance_evaluation import ChatPerformanceEvaluator
from benchmarking.src.llmperf import common_metrics
from benchmarking.streamlit.streamlit_utils import APP_PAGES, LLM_API_OPTIONS, find_pages_to_hide

warnings.filterwarnings('ignore')

CONFIG_PATH = './config.yaml'
with open(CONFIG_PATH) as file:
    st.session_state.config = yaml.safe_load(file)
    st.session_state.prod_mode = st.session_state.config['prod_mode']
    st.session_state.pages_to_show = st.session_state.config['pages_to_show']


def _get_params() -> Dict[str, Any]:
    """Get LLM params

    Returns:
        dict: returns dictionary with LLM params
    """
    params = {
        # "do_sample": st.session_state.do_sample,
        'max_tokens_to_generate': st.session_state.max_tokens_to_generate,
        # "repetition_penalty": st.session_state.repetition_penalty,
        # "temperature": st.session_state.temperature,
        # "top_k": st.session_state.top_k,
        # "top_p": st.session_state.top_p,
    }
    return params


def _parse_llm_response(llm: ChatPerformanceEvaluator, prompt: str) -> Dict[str, Any]:
    """Parses LLM output to a dictionary with necessary performance metrics and completion

    Args:
        llm (ChatPerformanceEvaluator): Chat performance evaluation object
        prompt (str): user's prompt text

    Returns:
        dict: dictionary with performance metrics and completion text
    """

    llm_output = llm.generate(prompt=prompt)
    response = {
        'completion': llm_output[1],
        'time_to_first_token': llm_output[0][common_metrics.TTFT],
        'latency': llm_output[0][common_metrics.E2E_LAT],
        'throughput': llm_output[0][common_metrics.REQ_OUTPUT_THROUGHPUT],
    }
    return response


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
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = None

    # Initialize llm params
    # if "do_sample" not in st.session_state:
    #     st.session_state.do_sample = None
    if 'max_tokens_to_generate' not in st.session_state:
        st.session_state.max_tokens_to_generate = None
    # if "repetition_penalty" not in st.session_state:
    #     st.session_state.repetition_penalty = None
    # if "temperature" not in st.session_state:
    #     st.session_state.temperature = None
    # if "top_k" not in st.session_state:
    #     st.session_state.top_k = None
    # if "top_p" not in st.session_state:
    #     st.session_state.top_p = None


def main() -> None:
    if st.session_state.prod_mode:
        pages_to_hide = find_pages_to_hide()
        pages_to_hide.append(APP_PAGES['setup']['page_label'])
        hide_pages(pages_to_hide)
    else:
        hide_pages([APP_PAGES['setup']['page_label']])

    st.title(':orange[SambaNova] Chat Performance Evaluation')
    st.markdown(
        """With this option, users have a way to know performance metrics per response. Set your LLM first on the left
        side bar and then have a nice conversation, also know more about our performance metrics per each response."""
    )

    with st.sidebar:
        st.title('Set up the LLM')
        st.markdown('**Configure your LLM before starting to chat**')

        # Show LLM parameters
        llm_model = st.text_input(
            'Model Name',
            value='llama3-405b',
            help='Look at your model card in SambaStudio and introduce the same name of the model/expert here.',
        )
        llm_selected = f'{llm_model}'
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

        # st.session_state.do_sample = st.toggle("Do Sample")
        st.session_state.max_tokens_to_generate = st.number_input(
            'Max tokens to generate', min_value=50, max_value=2048, value=250, step=1
        )
        # st.session_state.repetition_penalty = st.slider('Repetition penalty', min_value=1.0, max_value=10.0,
        # step=0.01, value=1.0, format="%.2f")
        # st.session_state.temperature = st.slider('Temperature', min_value=0.01, max_value=1.00, value=0.1, step=0.01,
        # format="%.2f")
        # st.session_state.top_k = st.slider('Top K', min_value=1, max_value=1000, value=50)
        # st.session_state.top_p = st.slider('Top P', min_value=0.01, max_value=1.00, value=0.95, step=0.01,
        # format="%.2f")

        # Sets LLM
        sidebar_run_option = st.sidebar.button('Run!')

        # Additional settings
        with st.expander('Additional settings', expanded=True):
            st.markdown('**Reset chat**')
            st.markdown('**Note:** Resetting the chat will clear all conversation history')
            if st.button('Reset conversation'):
                st.session_state.chat_history = []
                st.session_state.perf_metrics_history = []

                st.toast('Conversation reset. The next response will clear the history on the screen')

        if st.session_state.prod_mode:
            if st.button('Back to Setup'):
                st.session_state.setup_complete = False
                st.switch_page('app.py')

    try:
        # Sets LLM based on side bar parameters and COE model selected

        if sidebar_run_option:
            params = _get_params()
            if isinstance(st.session_state.llm_api, str):
                st.session_state.selected_llm = ChatPerformanceEvaluator(
                    model_name=llm_selected, llm_api=st.session_state.llm_api, params=params
                )
                st.toast('LLM setup ready! 🙌 Start asking!')
                st.session_state.chat_disabled = False

        # Chat with user
        user_prompt = st.chat_input('Ask me anything', disabled=st.session_state.chat_disabled)

        # If user's asking something
        if user_prompt:
            st.session_state.mp_events.input_submitted('chat_input')
            with st.spinner('Processing'):
                # Display llm response
                llm_response = _parse_llm_response(st.session_state.selected_llm, user_prompt)

                # Add user message to chat history
                st.session_state.chat_history.append({'role': 'user', 'question': user_prompt})
                st.session_state.chat_history.append({'role': 'system', 'answer': llm_response['completion'].strip()})
                st.session_state.perf_metrics_history.append(
                    {
                        'time_to_first_token': llm_response['time_to_first_token'],
                        'latency': llm_response['latency'],
                        'throughput': llm_response['throughput'],
                    }
                )

                # Display chat messages and performance metrics from history on app
                for user, system, perf_metric in zip(
                    st.session_state.chat_history[::2],
                    st.session_state.chat_history[1::2],
                    st.session_state.perf_metrics_history,
                ):
                    with st.chat_message(user['role']):
                        st.write(f"{user['question']}")
                    with st.chat_message(
                        'ai',
                        avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
                    ):
                        st.write(f"{system['answer']}")
                        with st.expander('Performance metrics'):
                            st.markdown(
                                f"""<font size="2" color="grey">Time to first token:
                                  {round(perf_metric["time_to_first_token"],4)} seconds</font>""",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"""<font size="2" color="grey">Throughput: 
                                {round(perf_metric["throughput"] if perf_metric["throughput"] else 0,4)} 
                                tokens/second</font>""",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"""<font size="2" color="grey">Latency: {round(perf_metric["latency"],4 )}
                                  seconds</font>""",
                                unsafe_allow_html=True,
                            )
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
