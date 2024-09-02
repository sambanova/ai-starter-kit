import os
import streamlit as st
from typing import Tuple

def initialize_env_variables(prod_mode=False):
    if not prod_mode:
        # In non-prod mode, prioritize environment variables
        st.session_state.FASTAPI_URL = os.environ.get("FASTAPI_URL", st.session_state.get("FASTAPI_URL", ""))
        st.session_state.FASTAPI_API_KEY = os.environ.get("FASTAPI_API_KEY", st.session_state.get("FASTAPI_API_KEY", ""))
    else:
        # In prod mode, only use session state
        if 'FASTAPI_URL' not in st.session_state:
            st.session_state.FASTAPI_URL = ""
        if 'FASTAPI_API_KEY' not in st.session_state:
            st.session_state.FASTAPI_API_KEY = ""

def set_env_variables(url, api_key, prod_mode=False):
    st.session_state.FASTAPI_URL = url
    st.session_state.FASTAPI_API_KEY = api_key
    if not prod_mode:
        # In non-prod mode, also set environment variables
        os.environ["FASTAPI_URL"] = url
        os.environ["FASTAPI_API_KEY"] = api_key

def env_input_fields() -> Tuple[str, str]:
    url = st.text_input("API URL", value=st.session_state.FASTAPI_URL, type="password")
    api_key = st.text_input("API Key", value=st.session_state.FASTAPI_API_KEY, type="password")
    return url, api_key

def are_credentials_set() -> bool:
    return bool(st.session_state.FASTAPI_URL and st.session_state.FASTAPI_API_KEY)

def save_credentials(url, api_key, prod_mode=False) -> str:
    set_env_variables(url, api_key, prod_mode)
    return "Credentials saved successfully!"