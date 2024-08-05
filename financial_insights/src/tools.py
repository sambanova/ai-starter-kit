import json
import os
import sys

import streamlit
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)


CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

TEMP_DIR = 'financial_insights/streamlit/cache/'

load_dotenv(os.path.join(repo_dir, '.env'))


# tool schema
class ConversationalResponse(BaseModel):
    """
    Respond conversationally only if no other tools should be called for a given query,
    or if you have a final answer. The response must be in the same language as the user query.
    """

    user_request: str = Field(..., description='The user query.')
    response_object: str = Field(
        ..., description='The final answer to the query, to be put in conversational response.'
    )


@tool(args_schema=ConversationalResponse)
def get_conversational_response(user_request: str, response_object: str) -> str:
    """
    Respond conversationally only if no other tools should be called for a given query,
    or if you have a final answer. The response must be in the same language as the user query.
    """
    # Convert object to string
    response_string = json.dumps(response_object)

    prompt = (
        f'Here is the user request:\n{user_request}\n'
        + f'Here is the response object:\n{response_string}\n'
        + 'Please rephrase and return the answer in a conversational, but formal style. Just return the answer without any preamble.'
    )

    # Get response from llama3
    response = streamlit.session_state.fc.llm.invoke(prompt)
    return response
