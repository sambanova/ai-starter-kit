import os
import streamlit as st

def initialize_env_variables():
    
    os.environ["FASTAPI_URL"] = ""
    os.environ["FASTAPI_API_KEY"] = ""

def set_env_variables(url, api_key):
    os.environ["FASTAPI_URL"] = url
    os.environ["FASTAPI_API_KEY"] = api_key

def env_input_fields():
    api_key = st.text_input("API Key", value=os.getenv("FASTAPI_API_KEY", ""), type="password")
    url = st.text_input("API URL", value=os.getenv("FASTAPI_URL", ""), type="password")
    
    
    if st.button("Save Credentials"):
        set_env_variables(url, api_key)
        st.success("Credentials saved successfully!")
    
    return url, api_key