import streamlit
import os
from financial_assistant.constants import CONFIG_PATH
from financial_assistant.src.llm import SambaNovaLLM

# Instantiate the LLM
sambanova_llm = SambaNovaLLM(
    config_path=CONFIG_PATH,
    sambanova_api_key=streamlit.session_state.SAMBANOVA_API_KEY
    if 'session_state' in streamlit.session_state
    else os.getenv('SAMBANOVA_API_KEY'),
)
