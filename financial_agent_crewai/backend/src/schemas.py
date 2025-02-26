from pydantic import BaseModel


class UserInput(BaseModel):
    user_query: str
    source_generic_search: bool = True
    source_sec_filings: bool = False
    source_yfinance_news: bool = False
    source_yfinance_stocks: bool = False


class AgentFinalOutput(BaseModel):
    title: str
    summary: str
