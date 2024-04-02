import os
from dotenv import load_dotenv
load_dotenv('.env')

from prompts.prompts import CUSTOM_PROMPT_TEMPLATE
from utils.sambanova_endpoint import SambaNovaEndpoint

llm = SambaNovaEndpoint(
    base_url=os.getenv('BASE_URL'),
    project_id=os.getenv('PROJECT_ID'),
    endpoint_id=os.getenv('ENDPOINT_ID'),
    api_key=os.getenv('API_KEY'),
    model_kwargs={
        "do_sample": False,
        "temperature": 0.0,
        "max_tokens_to_generate": 250
    },
)

def translate(sentence):
    return llm(CUSTOM_PROMPT_TEMPLATE + sentence)