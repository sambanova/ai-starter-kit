import os

from crewai import LLM

from utils.model_wrappers.api_gateway import APIGateway

llm = LLM(model='sambanova/Meta-Llama-3.1-70B-Instruct', temperature=0)

# Instantiate the LLM
rag_llm = APIGateway.load_llm(
    type='sncloud',
    streaming=False,
    bundle=True,
    do_sample=False,
    max_tokens_to_generate=1024,
    temperature=0.7,
    select_expert='Meta-Llama-3.1-70B-Instruct',
    process_prompt=False,
    sambanova_api_key=os.getenv('SAMBANOVA_API_KEY'),
)
