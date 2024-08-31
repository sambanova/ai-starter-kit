import os
import streamlit as st

def initialize_env_variables():
    if 'FASTAPI_URL' not in st.session_state:
        st.session_state.FASTAPI_URL = os.environ.get("FASTAPI_URL", "")
    if 'FASTAPI_API_KEY' not in st.session_state:
        st.session_state.FASTAPI_API_KEY = os.environ.get("FASTAPI_API_KEY", "")

    # Set environment variables from session state
    os.environ["FASTAPI_URL"] = st.session_state.FASTAPI_URL
    os.environ["FASTAPI_API_KEY"] = st.session_state.FASTAPI_API_KEY

def set_env_variables(url, api_key):
    st.session_state.FASTAPI_URL = url
    st.session_state.FASTAPI_API_KEY = api_key
    os.environ["FASTAPI_URL"] = url
    os.environ["FASTAPI_API_KEY"] = api_key

def env_input_fields():
    url = st.text_input("API URL", value=st.session_state.FASTAPI_URL, type="password")
    api_key = st.text_input("API Key", value=st.session_state.FASTAPI_API_KEY, type="password")
    
    if st.button("Save Credentials"):
        set_env_variables(url, api_key)
        st.success("Credentials saved successfully!")
    
    return url, api_key

def are_credentials_set():
    return bool(st.session_state.FASTAPI_URL and st.session_state.FASTAPI_API_KEY)