import sys
sys.path.append('../')

import streamlit as st
from chat_benchmarking.src.llmperf import common_metrics
from chat_benchmarking.src.chat_performance_evaluation import ChatPerformanceEvaluator
from typing import Any, Dict


def _initialize_session_variables() -> None:
    print('Initializing session variables...')
    # Initialize chat history
    if 'chat_history_1' not in st.session_state:
        st.session_state.chat_history_1 = []
    if 'chat_history_2' not in st.session_state:
        st.session_state.chat_history_2 = []
    if 'perf_metrics_history' not in st.session_state:
        st.session_state.perf_metrics_history = []
    if 'max_tokens_to_generate' not in st.session_state:
        st.session_state.max_tokens_to_generate = 100

def _get_params() -> Dict[str, Any]:
    """Get LLM params

    Returns:
        dict: returns dictionary with LLM params
    """
    params = {
        'max_tokens_to_generate': st.session_state.max_tokens_to_generate,
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

# Initialize session variables
_initialize_session_variables()

# Input text box
user_input = st.text_input("Enter your message:")

if user_input:
    params = _get_params()
    llm_1 = llm_2 = ChatPerformanceEvaluator(
        model_name='Meta-Llama-3.3-70B-Instruct', llm_api='sncloud', params=params
    )
    st.toast('LLM setup ready! 🙌 Start asking!')
    st.session_state.chat_disabled = False

    # Simulate LLM response
    response_1 = _parse_llm_response(llm_1, user_input)
    response_2 = _parse_llm_response(llm_2, user_input)

    # Update chat history
    st.session_state.chat_history_1.append({"user": user_input, "bot": response_1['completion']})
    st.session_state.chat_history_2.append({"user": user_input, "bot": response_2['completion']})

# Display chat messages side-by-side
col1, col2 = st.columns(2)

with col1:
    st.header("Chat 1")
    for message in st.session_state.chat_history_1:
        st.write(f"**User:** {message['user']}")
        st.write(f"**Bot:** {message['bot']}")

with col2:
    st.header("Chat 2")
    for message in st.session_state.chat_history_2:
        st.write(f"**User:** {message['user']}")
        st.write(f"**Bot:** {message['bot']}")