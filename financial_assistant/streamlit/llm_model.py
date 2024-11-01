import streamlit

from financial_assistant.constants import CONFIG_PATH
from financial_assistant.src.llm import SambaNovaLLM

# Instantiate the LLM
sambanova_llm = SambaNovaLLM(config_path=CONFIG_PATH)

# Set the LLM with the SambaNova API key
sambanova_llm.set_llm(
    sambanova_api_key=streamlit.session_state.SAMBANOVA_API_KEY,
)
