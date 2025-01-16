import sys

import streamlit as st

import yaml

from dotenv import load_dotenv

sys.path.append('../')

import warnings
from typing import Any, Dict

from chat_benchmarking.src.chat_performance_evaluation import ChatPerformanceEvaluator
from chat_benchmarking.src.llmperf import common_metrics
from chat_benchmarking.streamlit.streamlit_utils import LLM_API_OPTIONS

warnings.filterwarnings('ignore')

def _init_config() -> None:
    # Load the configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    st.session_state.config = config

def _get_params() -> Dict[str, Any]:
    """Get LLM params

    Returns:
        dict: returns dictionary with LLM params
    """
    params = {
        'max_tokens_to_generate': st.session_state.max_tokens_to_generate,
        "repetition_penalty": st.session_state.repetition_penalty,
        "temperature": st.session_state.temperature,
        "top_k": st.session_state.top_k,
        "top_p": st.session_state.top_p,
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
    # Initialize config
    if 'config' not in st.session_state:
        st.session_state.config = None
    
    # Initialize chat history
    if 'chat_history_1' not in st.session_state:
        st.session_state.chat_history_1 = []
    if 'chat_history_2' not in st.session_state:
        st.session_state.chat_history_2 = []
        
    if 'perf_metrics_history_1' not in st.session_state:
        st.session_state.perf_metrics_history_1 = []
    if 'perf_metrics_history_2' not in st.session_state:
        st.session_state.perf_metrics_history_2 = []
        
    if 'chat_disabled' not in st.session_state:
        st.session_state.chat_disabled = True
        
    # Initialize llm variables    
    if 'llm_nickname_1' not in st.session_state:
        st.session_state.llm_nickname_1 = None
    if 'llm_1' not in st.session_state:
        st.session_state.llm_1 = None
    if 'llm_nickname_2' not in st.session_state:
        st.session_state.llm_nickname_2 = None
    if 'llm_2' not in st.session_state:
        st.session_state.llm_2 = None
    if 'llm_api' not in st.session_state:
        st.session_state.llm_api = None

    # Initialize llm parameters
    if 'max_tokens_to_generate' not in st.session_state:
        st.session_state.max_tokens_to_generate = None
    if "repetition_penalty" not in st.session_state:
        st.session_state.repetition_penalty = None
    if "temperature" not in st.session_state:
        st.session_state.temperature = None
    if "top_k" not in st.session_state:
        st.session_state.top_k = None
    if "top_p" not in st.session_state:
        st.session_state.top_p = None


def main() -> None:

    st.markdown("<h1 style='text-align: center; color: black;'> <span style='color: orange;'>SambaNova</span> Chat Performance Evaluation</h1>", unsafe_allow_html=True)
    st.markdown(
        """Users can know and compare the speed performance metrics per chat response. Set the LLMs first on the left
        side bar, configure the parameters, and then run your query. Once the chat responses are displayed, 
        you will be able to see the performance metrics."""
    )

    with st.sidebar:
        st.title('Set up the LLM')
        
        # Select LLM APIs
        st.markdown('**Choose the providers you want to compare.**')
        llm_options = [provider['nickname'] for provider in st.session_state.config['providers']]
        if len(llm_options) > 2:
            
            st.session_state.llm_nickname_1 = st.selectbox(
                'LLM Provider 1', options=llm_options, index=0
            )
            llm_model_options_1 = [provider for provider in st.session_state.config['providers'] if provider['nickname'] == st.session_state.llm_nickname_1][0]['models']
            st.session_state.llm_model_1 = st.selectbox(
                'LLM models for Provider 1', options=llm_model_options_1, index=0
            )
            
            
            st.session_state.llm_nickname_2 = st.selectbox(
                'LLM Provider 2', options=llm_options, index=1
            )
            llm_model_options_2 = [provider for provider in st.session_state.config['providers'] if provider['nickname'] == st.session_state.llm_nickname_2][0]['models']
            st.session_state.llm_model_2 = st.selectbox(
                'LLM models for Provider 2', options=llm_model_options_2, index=0
            )
        else:
            raise ValueError('Please provide at least 2 LLM providers in the config file')
        
        st.divider()
        
        # Select LLM parameters
        st.markdown('**Configure your LLMs before starting to chat**')
        st.session_state.max_tokens_to_generate = st.number_input(
            'Max tokens to generate', min_value=50, max_value=2048, value=250, step=1
        )
        st.session_state.temperature = st.number_input('Temperature', min_value=0.00, max_value=1.00, value=0.7, step=0.01,
        format="%.2f")
        st.session_state.top_k = st.number_input('Top K', min_value=1, max_value=100, value=50)
        st.session_state.top_p = st.number_input('Top P', min_value=0.0, max_value=1.00, value=0.95, step=0.01,
        format="%.2f")
        st.session_state.repetition_penalty = st.number_input('Repetition penalty', min_value=1.0, max_value=10.0,
        step=0.01, value=1.0, format="%.2f")

        # Sets LLM
        st.markdown("""<style>.stButton > button {width: 100%;background-color: orange;color: white;}</style>""", unsafe_allow_html=True)
        sidebar_set_option = st.sidebar.button('Set!')

        # Additional settings
        with st.expander('Additional settings', expanded=True):
            st.markdown('**Reset chat: Resetting the chat will clear all conversation history**')
            if st.button('Reset conversation'):
                st.session_state.chat_history_1 = []
                st.session_state.chat_history_2 = []
                st.session_state.perf_metrics_history_1 = []
                st.session_state.perf_metrics_history_2 = []

                st.toast('Conversation reset. The next response will clear the history on the screen')
    try:
        # Sets LLM based on side bar parameters and bundle model selected
        if sidebar_set_option:
            params = _get_params()
            
            # Set LLM 1
            api_variables_1 = {
                'API_KEY': [provider for provider in st.session_state.config['providers'] if provider['nickname'] == st.session_state.llm_nickname_1][0]['api_key'],
                'API_URL': [provider for provider in st.session_state.config['providers'] if provider['nickname'] == st.session_state.llm_nickname_1][0]['api_url'],
            }
            st.session_state.llm_1 = ChatPerformanceEvaluator(
                model_name=st.session_state.llm_model_1, llm_api='sncloud', params=params, api_variables=api_variables_1
            )
            
            # Set LLM 2
            api_variables_2 = {
                'API_KEY': [provider for provider in st.session_state.config['providers'] if provider['nickname'] == st.session_state.llm_nickname_2][0]['api_key'],
                'API_URL': [provider for provider in st.session_state.config['providers'] if provider['nickname'] == st.session_state.llm_nickname_2][0]['api_url'],
            }
            st.session_state.llm_2 = ChatPerformanceEvaluator(
                model_name=st.session_state.llm_model_2, llm_api='sncloud', params=params, api_variables=api_variables_2
            )
            
            st.toast('LLM setup ready! 🙌 Start asking!')
            st.session_state.chat_disabled = False

        # Chat with user
        user_prompt = st.chat_input('Ask me anything', disabled=st.session_state.chat_disabled)

        # If user's asking something
        if user_prompt:
            with st.spinner('Processing'):
                # Display llm response
                llm_response_1 = _parse_llm_response(st.session_state.llm_1, user_prompt)
                llm_response_2 = _parse_llm_response(st.session_state.llm_2, user_prompt)

                # Add user message to chat history
                st.session_state.chat_history_1.append({'role': 'user', 'question': user_prompt})
                st.session_state.chat_history_1.append({'role': 'system', 'answer': llm_response_1['completion'].strip()})
                
                st.session_state.chat_history_2.append({'role': 'user', 'question': user_prompt})
                st.session_state.chat_history_2.append({'role': 'system', 'answer': llm_response_2['completion'].strip()})
                
                
                st.session_state.perf_metrics_history_1.append(
                    {
                        'time_to_first_token': llm_response_1['time_to_first_token'],
                        'latency': llm_response_1['latency'],
                        'throughput': llm_response_1['throughput'],
                    }
                )
                
                st.session_state.perf_metrics_history_2.append(
                    {
                        'time_to_first_token': llm_response_2['time_to_first_token'],
                        'latency': llm_response_2['latency'],
                        'throughput': llm_response_2['throughput'],
                    }
                )

                # Display chat messages and performance metrics from history on app        
                col1, col2 = st.columns(2)

                with col1:
                    st.header(st.session_state.llm_nickname_1)
                    for user, system, perf_metric in zip(
                        st.session_state.chat_history_1[::2],
                        st.session_state.chat_history_1[1::2],
                        st.session_state.perf_metrics_history_1,
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

                with col2:
                    st.header(st.session_state.llm_nickname_2)
                    for user, system, perf_metric in zip(
                        st.session_state.chat_history_2[::2],
                        st.session_state.chat_history_2[1::2],
                        st.session_state.perf_metrics_history_2,
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
        layout="wide"
    )
    
    _initialize_sesion_variables()
    _init_config()

    main()
