import streamlit

from financial_assistant.constants import CONFIG_PATH
from financial_assistant.src.llm import SambaNovaLLM

# Instantiate the LLM
sambanova_llm = SambaNovaLLM(
    config_path=CONFIG_PATH,
    sambanova_api_key=streamlit.session_state.SAMBANOVA_API_KEY,
)

from langchain_openai import ChatOpenAI

sambanova_chat = ChatOpenAI(model='gpt-4', temperature=0)
