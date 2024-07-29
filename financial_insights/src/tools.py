import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas
import plotly
import plotly.graph_objects as go
import requests  # type: ignore
import streamlit
import yaml
import yfinance
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from pandasai import SmartDataframe
from pandasai.connectors.yahoo_finance import YahooFinanceConnector

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
    or if you have a final answer. response must be in the same language as the user query.
    """

    response: str = Field(
        ..., description='Conversational response to the user. Must be in the same language as the user query.'
    )


@tool(args_schema=ConversationalResponse)
def get_conversational_response(response: str) -> str:
    """
    Respond conversationally only if no other tools should be called for a given query,
    or if you have a final answer. response must be in the same language as the user query.
    """

    return response

