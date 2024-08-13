import datetime
import json
import os
import sys
from typing import Any, Dict, Union

import pandas
import streamlit
import yfinance
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
def get_conversational_response(user_request: str, response_object: str) -> Any:
    """
    Respond conversationally only if no other tools should be called for a given query,
    or if you have a final answer. The response must be in the same language as the user query.
    """
    # Convert object to string
    response_string = json.dumps(response_object)

    prompt = (
        f'Here is the user request:\n{user_request}\n'
        + f'Here is the response object:\n{response_string}\n'
        + 'Please rephrase and return the answer in a conversational, but formal style. '
        'Just return the answer without any preamble.'
    )

    # Get response from llama3
    response = streamlit.session_state.fc.llm.invoke(prompt)
    return response


def extract_yfinance_data(
    symbol: str, start_date: datetime.date, end_date: datetime.date
) -> Dict[str, Union[pandas.DataFrame, Dict[Any, Any]]]:
    company = yfinance.Ticker(ticker=symbol)

    company_dict = dict()

    # get all stock info
    company_dict['info'] = company.info

    # get historical market data
    company_dict['history'] = company.history(start=start_date, end=end_date)

    # show meta information about the history (requires history() to be called first)
    company_dict['history_metadata'] = company.history_metadata

    # show actions (dividends, splits, capital gains)
    company_dict['actions'] = company.actions
    company_dict['dividends'] = company.dividends
    company_dict['splits'] = company.splits
    company_dict['capital_gains'] = company.capital_gains  # only for mutual funds & etfs

    # show share count
    company_dict['shares'] = company.get_shares_full(start=start_date, end=end_date)

    # show financials:
    # - income statement
    company_dict['income_stmt'] = convert_date_index_to_column(company.income_stmt.T)
    company_dict['quarterly_income_stmt'] = convert_date_index_to_column(company.quarterly_income_stmt.T)
    # - balance sheet
    company_dict['balance_sheet'] = convert_date_index_to_column(company.balance_sheet.T)
    company_dict['quarterly_balance_sheet'] = convert_date_index_to_column(company.quarterly_balance_sheet.T)
    # - cash flow statement
    company_dict['cashflow'] = convert_date_index_to_column(company.cashflow.T)
    company_dict['quarterly_cashflow'] = convert_date_index_to_column(company.quarterly_cashflow.T)
    # see `Ticker.get_income_stmt()` for more options

    # show holders
    company_dict['major_holders'] = company.major_holders
    company_dict['institutional_holders'] = company.institutional_holders
    company_dict['mutualfund_holders'] = company.mutualfund_holders
    company_dict['insider_transactions'] = company.insider_transactions
    company_dict['insider_purchases'] = company.insider_purchases
    company_dict['insider_roster_holders'] = company.insider_roster_holders

    company_dict['sustainability'] = company.sustainability

    # show recommendations
    company_dict['recommendations'] = company.recommendations
    company_dict['recommendations_summary'] = company.recommendations_summary
    company_dict['upgrades_downgrades'] = company.upgrades_downgrades

    # Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
    # Note: If more are needed use company.get_earnings_dates(limit=XX) with increased limit argument.
    company_dict['earnings_dates'] = company.earnings_dates

    # show ISIN code - *experimental*
    # ISIN = International Securities Identification Number
    company_dict['isin'] = company.isin

    # show options expirations
    company_dict['options'] = company.options

    # show news
    company_dict['news'] = company.news

    # # get option chain for specific expiration
    # company_dict["option_chain"] = company.option_chain()
    # # data available via: opt.calls, opt.puts

    return company_dict


def convert_data_to_frame(data: Any, df_name: str) -> pandas.DataFrame:
    if isinstance(data, pandas.DataFrame):
        df = data
    elif isinstance(data, dict):
        for key, value in data.items():
            if is_unhashable(value):
                data[key] = json.dumps(value)
        df = pandas.Series(data).to_frame().T
        df.reset_index(inplace=True)
    elif isinstance(data, pandas.Series):
        df = data.to_frame()
    elif isinstance(data, str):
        df = pandas.DataFrame({df_name: [data]})
    elif isinstance(data, (list, tuple)):
        for item in data:
            item = json.dumps(item)
        df = pandas.DataFrame({df_name: data})
    else:
        raise ValueError(f'Unsupported data type for {df_name}: {type(data)}')

    return df


def is_unhashable(obj: Any) -> bool:
    try:
        # Attempt to call hash() on the object
        hash(obj)
    except TypeError:
        # If a TypeError occurs, the object is unhashable
        return True
    return False


def convert_date_index_to_column(df: pandas.DataFrame) -> pandas.DataFrame:
    df_new = df.reset_index()
    new_column_list = ['Date'] + list(df.columns)
    df_new.columns = new_column_list
    return df_new
