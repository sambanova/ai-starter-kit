import datetime
import json
from typing import Any, Dict, Optional, Union

import pandas
import streamlit
import yfinance
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

from financial_insights.streamlit.constants import *


# Tool schema for the conversational response tool
class ConversationalResponse(BaseModel):
    """Elaborate a conversational answer"""

    response: str = Field(description='The conversational answer')


# tool schema for the final conversational response tool
class FinalConversationalResponse(BaseModel):
    """Turn a response in a conversational format of the same language as the user query."""

    user_query: str = Field(description='The user query.')
    response_object: str = Field(description='The response to the query, to be put in conversational format.')


@tool(args_schema=FinalConversationalResponse)
def get_conversational_response(user_query: str, response_object: str) -> Any:
    """Turn a response in a conversational format of the same language as the user query."""

    # The output parser
    conversational_parser = PydanticOutputParser(pydantic_object=ConversationalResponse)  # type: ignore

    # Convert object to string
    response_string = json.dumps(response_object)
    # Clear all special characters
    response_string = response_string.replace('"', '')
    response_string = response_string.replace('\n', '')

    # The prompt template
    conversational_prompt_template = (
        'Here is the user request:\n{user_query}\n'
        + 'Here is the response object:\n{response_string}\n'
        + 'Please rephrase and return the response object in a conversational, but formal style. '
        'Just return the conversational answer without any preamble.'
    )

    # The prompt
    conversational_prompt = PromptTemplate(
        template=conversational_prompt_template,
        input_variables=['user_query', 'response_string'],
        partial_variables={'format_instructions': conversational_parser.get_format_instructions()},
    )

    # The chain
    # | conversational_parser. TODO: The parser does not work
    conversational_chain = conversational_prompt | streamlit.session_state.fc.llm

    # Get response from the LLM
    response = conversational_chain.invoke({'user_query': user_query, 'response_string': response_string})
    response = response.replace(' \n', '')
    return response


def extract_yfinance_data(
    symbol: str, start_date: datetime.date, end_date: datetime.date
) -> Dict[str, Union[pandas.DataFrame, Dict[Any, Any]]]:
    """
    Extracts all the data of a given company using Yahoo Finance for specified dates.

    Args:
        symbol: The ticker symbol of the company to extract data from.
        start_date: The start date of the historical price data to retrieve.
        end_date: The end date of the historical price data to retrieve.

    Returns:
        A dictionary containing the data of the company extracted from Yahoo Finance.

    Raises:
        TypeError: If `symbol` is not a string or `start_date` and `end_date` are not of type datetime.date.
    """
    # Check inputs
    assert isinstance(symbol, str), TypeError('Symbol must be a string.')
    assert isinstance(start_date, datetime.date), TypeError('Start date must be of type datetime.date.')
    assert isinstance(end_date, datetime.date), TypeError('End date must be of type datetime.date.')

    # Extract the data from Yahoo Finance for the given ticker symbol
    company = yfinance.Ticker(ticker=symbol)

    # Initialize the return dictionary
    company_dict = dict()

    # Get all the stock information
    company_dict['info'] = company.info

    # Get historical market data
    company_dict['history'] = company.history(start=start_date, end=end_date)

    # Get meta information about the history (requires history() to be called first)
    company_dict['history_metadata'] = company.history_metadata

    # Get actions (dividends, splits, capital gains)
    company_dict['actions'] = company.actions
    company_dict['dividends'] = company.dividends
    company_dict['splits'] = company.splits
    company_dict['capital_gains'] = company.capital_gains  # only for mutual funds & etfs

    # Get share count
    company_dict['shares'] = company.get_shares_full(start=start_date, end=end_date)

    # Get financials:
    # Income statement
    company_dict['income_stmt'] = convert_index_to_column(company.income_stmt.T, 'Date')
    company_dict['quarterly_income_stmt'] = convert_index_to_column(company.quarterly_income_stmt.T, 'Date')
    # Balance sheet
    company_dict['balance_sheet'] = convert_index_to_column(company.balance_sheet.T, 'Date')
    company_dict['quarterly_balance_sheet'] = convert_index_to_column(company.quarterly_balance_sheet.T, 'Date')
    # Cash flow statement
    company_dict['cashflow'] = convert_index_to_column(company.cashflow.T, 'Date')
    company_dict['quarterly_cashflow'] = convert_index_to_column(company.quarterly_cashflow.T, 'Date')
    # see `Ticker.get_income_stmt()` for more options

    # Show holders
    company_dict['major_holders'] = company.major_holders
    company_dict['institutional_holders'] = company.institutional_holders
    company_dict['mutualfund_holders'] = company.mutualfund_holders
    company_dict['insider_transactions'] = company.insider_transactions
    company_dict['insider_purchases'] = company.insider_purchases
    company_dict['insider_roster_holders'] = company.insider_roster_holders

    # Get sustainability
    company_dict['sustainability'] = company.sustainability

    # Get recommendations
    company_dict['recommendations'] = company.recommendations
    company_dict['recommendations_summary'] = company.recommendations_summary
    company_dict['upgrades_downgrades'] = company.upgrades_downgrades

    # Get future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
    # Note: If more are needed use company.get_earnings_dates(limit=XX) with increased limit argument.
    company_dict['earnings_dates'] = company.earnings_dates

    # Get ISIN code - *experimental*
    # ISIN = International Securities Identification Number
    company_dict['isin'] = company.isin

    # Get options expirations
    company_dict['options'] = company.options

    # Get news
    company_dict['news'] = company.news

    # # Get option chain for specific expiration
    # company_dict["option_chain"] = company.option_chain()
    # # data available via: opt.calls, opt.puts

    return company_dict


def convert_data_to_frame(data: Any, df_name: str) -> pandas.DataFrame:
    """
    Converts data to pandas DataFrame.

    Args:
        data: Data to be converted to DataFrame.
        df_name: Name of the DataFrame to be created.

    Returns:
        `pandas.DataFrame` object from the original data.

    Raises:
        `TypeError`: If the data not of type `pandas.DataFrame`, `pandas.Series`, `dict`, `list`, `tuple`, `str`.
    """
    if isinstance(data, pandas.DataFrame):
        df = data

    elif isinstance(data, pandas.Series):
        df = data.to_frame()

    elif isinstance(data, dict):
        for key, value in data.items():
            if is_unhashable(value):
                data[key] = json.dumps(value)
        df = pandas.Series(data).to_frame().T
        df.reset_index(inplace=True)

    elif isinstance(data, (list, tuple)):
        for item in data:
            item = json.dumps(item)
        df = pandas.DataFrame({df_name: data})

    elif isinstance(data, str):
        df = pandas.DataFrame({df_name: [data]})

    else:
        raise TypeError(f'Data type {type(data)} not supported.')

    return df


def is_unhashable(input_object: Any) -> bool:
    """Determine whether an object is unhashable or not."""

    try:
        # Attempt to call hash() on the object
        hash(input_object)
    except TypeError:
        # If a TypeError occurs, the object is unhashable
        return True
    return False


def convert_index_to_column(dataframe: pandas.DataFrame, column_name: Optional[str] = None) -> pandas.DataFrame:
    """
    Convert the index of a dataframe to a column.

    Args:
        df: The input dataframe.
        column_name: The name of the new column.

    Returns:
        The dataframe with the index converted to a column.

    Raises:
        TypeError: If `dataframe` is not of type `pandas.DataFrame`
            or `column_name` is not of type string.
    """
    # Check inputs
    assert isinstance(dataframe, pandas.DataFrame), TypeError(
        f'Input must be a Pandas DataFrame. Got {type(dataframe)}.'
    )
    assert isinstance(column_name, str), TypeError(f'Column name must be a string. Got {type(column_name)}.')

    # Determine the column name
    if column_name is None:
        column_name = dataframe.index.name

    df_new = dataframe.reset_index()
    new_column_list = [column_name] + list(dataframe.columns)
    df_new.columns = new_column_list
    return df_new
