import os

from dotenv import load_dotenv

from financial_assistant.constants import CONFIG_PATH, repo_dir
from financial_assistant.src.llm import SambaNovaLLM

load_dotenv(os.path.join(repo_dir, '.env'))

# Instantiate the LLM
sambanova_llm = SambaNovaLLM(
    config_path=CONFIG_PATH,
)
