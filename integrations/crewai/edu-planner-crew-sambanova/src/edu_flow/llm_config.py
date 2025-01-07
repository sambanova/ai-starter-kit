from crewai import LLM
from .config import LLM_CONFIG

llm = LLM(
    model=LLM_CONFIG["model"],
    api_key=LLM_CONFIG["api_key"]
)