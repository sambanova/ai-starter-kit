import os
from typing import Any

import weave
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_sambanova import ChatSambaNovaCloud  # type: ignore

load_dotenv()

# Initialize Weave with your project name
weave.init('weave_integration_langchain')
SAMBANOVA_URL = os.getenv('SAMBANOVA_URL')
SAMBANOVA_API_KEY = os.getenv('SAMBANOVA_API_KEY')
model = 'Meta-Llama-3.3-70B-Instruct'


def llm_call() -> Any:
    llm = ChatSambaNovaCloud(
        model=model,
        temperature=0.7,
        top_p=0.95,
    )
    prompt = PromptTemplate.from_template('1 + {number} = ')

    llm_chain = prompt | llm

    output = llm_chain.invoke({'number': 2})
    return output


def llm_call_attribute() -> Any:
    llm = ChatSambaNovaCloud(
        model=model,
        temperature=0.7,
        top_p=0.95,
    )
    prompt = PromptTemplate.from_template('1 + {number} = ')

    llm_chain = prompt | llm

    with weave.attributes({'number_to_increment': 'value'}):
        output = llm_chain.invoke({'number': 2})
    return output


if __name__ == '__main__':
    output = llm_call()
    print(output)

    output = llm_call_attribute()
    print(output)
