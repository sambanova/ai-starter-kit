import os
import streamlit as st

DEFAULT_FASTAPI_URL = "https://fast-api.snova.ai/v1/chat/completions"

def initialize_env_variables(prod_mode=False, additional_env_vars=None):
    if additional_env_vars is None:
        additional_env_vars = []

    if not prod_mode:
        # In non-prod mode, prioritize environment variables
        st.session_state.FASTAPI_URL = os.environ.get("FASTAPI_URL", st.session_state.get("FASTAPI_URL", DEFAULT_FASTAPI_URL))
        st.session_state.FASTAPI_API_KEY = os.environ.get("FASTAPI_API_KEY", st.session_state.get("FASTAPI_API_KEY", ""))
        for var in additional_env_vars:
            st.session_state[var] = os.environ.get(var, st.session_state.get(var, ""))
    else:
        # In prod mode, only use session state
        if 'FASTAPI_URL' not in st.session_state:
            st.session_state.FASTAPI_URL = DEFAULT_FASTAPI_URL
        if 'FASTAPI_API_KEY' not in st.session_state:
            st.session_state.FASTAPI_API_KEY = ""
        for var in additional_env_vars:
            if var not in st.session_state:
                st.session_state[var] = ""

def set_env_variables(api_key, additional_vars=None, prod_mode=False):
    st.session_state.FASTAPI_API_KEY = api_key
    if additional_vars:
        for key, value in additional_vars.items():
            st.session_state[key] = value
    if not prod_mode:
        # In non-prod mode, also set environment variables
        os.environ["FASTAPI_API_KEY"] = api_key
        if additional_vars:
            for key, value in additional_vars.items():
                os.environ[key] = value

def env_input_fields(additional_env_vars=None):
    if additional_env_vars is None:
        additional_env_vars = []

    api_key = st.text_input("Sambanova API Key", value=st.session_state.FASTAPI_API_KEY, type="password")
    
    additional_vars = {}
    for var in additional_env_vars:
        additional_vars[var] = st.text_input(f"{var}", value=st.session_state.get(var, ""), type="password")

    return api_key, additional_vars

def are_credentials_set(additional_env_vars=None):
    if additional_env_vars is None:
        additional_env_vars = []

    base_creds_set = bool(st.session_state.FASTAPI_API_KEY)
    additional_creds_set = all(bool(st.session_state.get(var, "")) for var in additional_env_vars)
    
    return base_creds_set and additional_creds_set

def save_credentials(api_key, additional_vars=None, prod_mode=False):
    set_env_variables(api_key, additional_vars, prod_mode)
    return "Credentials saved successfully!"