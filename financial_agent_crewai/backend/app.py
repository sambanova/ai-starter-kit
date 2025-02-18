import base64
import os
import sys
import time
from typing import Any


from financial_agent_crewai.main import FinancialFlow
from financial_agent_crewai.src.financial_agent_crewai.config import *
from financial_agent_crewai.utils.utilities import *

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional

import logging
import asyncio
import sys
import queue
import threading
from contextlib import contextmanager
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
from typing import AsyncGenerator


class UserInput(BaseModel):
    user_query: str
    source_generic_search: bool
    source_sec_filings: bool
    source_yfinance_news: bool
    source_yfinance_stocks: bool


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/financial_agent")
def financial_agent(user_input: UserInput):
    financial_flow = FinancialFlow(
        query=user_input.user_query,
        source_generic_search=user_input.source_generic_search,
        source_sec_filings=user_input.source_sec_filings,
        source_yfinance_news=user_input.source_yfinance_news,
        source_yfinance_stocks=user_input.source_yfinance_stocks,
    )
    results = financial_flow.kickoff()

    return results
