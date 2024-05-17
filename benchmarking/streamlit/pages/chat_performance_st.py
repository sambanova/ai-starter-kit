import os
import sys
sys.path.append('../../')

import streamlit as st

from benchmarking.src.chat_performance_evaluation import SambaVerseHandlerTmp


def _get_params():
    params = {
        # "do_sample": {"type": "bool", "value": "false"},
        "max_tokens_to_generate": {"type": "int", "value":str(max_output_tokens)},
        "temperature": {"type": "float", "value": str(temperature)},
        # "repetition_penalty":{"type":"float","value":"1.0"},
        # "top_k":{"type":"int","value":"50"},
        # "top_p":{"type":"float","value":"0.95"},
        
        "process_prompt":{"type":"bool", "value":"false"},
        "select_expert":{"type":"str", "value":f"{llm_selected.split('/')[-1]}"},
    }
    return params

def _parse_llm_response(llm: SambaVerseHandlerTmp, prompt):
    response = llm.generate(prompt=prompt)
    return response['result']['responses'][0]

st.title(":orange[SambaNova]Performance evaluation")    
st.header("Chat")    
st.markdown("With this option, users have a way to know performance metrics per response. Have a nice conversation with the LLM and know more about our performance metrics.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm" not in st.session_state:
    st.session_state.llm = None

with st.sidebar:
    st.title("Evaluation process")
    st.markdown("**Introduce inputs before running the process**")
    
    llm_options = ['Mistral/Mistral-7B-Instruct-v0.2', 'Meta/llama-2-7b-chat-hf']
    llm_selected = st.selectbox('Select LLM model', llm_options, index=0)
    max_output_tokens = st.number_input('Max tokens to generate', min_value=50, max_value=4096, value=250)
    temperature = st.number_input('Temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
    
    sidebar_option = st.sidebar.button("Run!")

if sidebar_option:
    
    params = _get_params()
    st.session_state.llm = SambaVerseHandlerTmp(model_name=llm_selected,params=params)
    st.toast('LLM ready! 🙌 Start asking!')

# React to user input
user_prompt = st.chat_input("Ask me anything")
        
if user_prompt:
    with st.spinner("Processing"):
        
        # Display chat messages from history on app rerun
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Display user message in chat message container
        st.chat_message("user").markdown(user_prompt)
        
        # Display system response
        system_response = _parse_llm_response(st.session_state.llm, user_prompt)
        st.chat_message("system").markdown(system_response['completion'])
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        st.session_state.chat_history.append({"role": "system", "content": system_response['completion']})
        
        