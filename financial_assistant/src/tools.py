import ast
import datetime
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

import pandas
import streamlit
import yfinance
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

from financial_assistant.prompts.conversational_prompts import CONVERSATIONAL_RESPONSE_PROMPT_TEMPLATE
from financial_assistant.streamlit.constants import *

# Generic function type
F = TypeVar('F', bound=Callable[..., Any])


class HasCall(Protocol):
    """Class to check if a class has a call method."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass


# Configure the logger
def get_logger(logger_name: str = 'logger') -> logging.Logger:
    # Get or create a logger instance
    logger = logging.getLogger(logger_name)

    # Ensure that the logger has not been configured with handlers already
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        # Prevent log messages from being propagated to the root logger
        logger.propagate = False

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        general_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(general_formatter)

        logger.addHandler(console_handler)

    return logger


# Get the loggers
timing_logger = get_logger('timingLogger')
logger = get_logger()


class ConversationalResponse(BaseModel):
    """Model representing a conversational answer."""

    response: str = Field(description='The conversational answer.')


class FinalConversationalResponse(BaseModel):
    """Tool for converting a response into a conversational format in the same language as the user query."""

    user_query: str = Field(..., description='The user query.')
    response_object: str = Field(..., description='The response to the query, formatted in a conversational style.')


@tool(args_schema=FinalConversationalResponse)
def get_conversational_response(user_query: str, response_object: Any) -> Any:
    """
    Tool for converting a response into a conversational format in the same language as the user query.

    Args:
        user_query: The user query.
        response_object: The response to the query, formatted in a conversational style.

    Returns:
        A conversational response.
    """

    # The output parser
    conversational_parser = PydanticOutputParser(pydantic_object=ConversationalResponse)  # type: ignore

    # Convert object to string
    response_string = json.dumps(response_object)

    # The prompt
    conversational_prompt = PromptTemplate(
        template=CONVERSATIONAL_RESPONSE_PROMPT_TEMPLATE,
        input_variables=['user_query', 'response_string'],
        partial_variables={'format_instructions': conversational_parser.get_format_instructions()},
    )

    try:
        # The chain
        conversational_chain = conversational_prompt | streamlit.session_state.llm.llm | conversational_parser
        # Get response from the LLM
        response = conversational_chain.invoke({'user_query': user_query, 'response_string': response_string}).response
    except:
        # The chain without the output parser
        conversational_chain = conversational_prompt | streamlit.session_state.llm.llm
        # Get response from the LLM
        response = conversational_chain.invoke({'user_query': user_query, 'response_string': response_string})

    return response


def extract_yfinance_data(
    symbol: str, start_date: datetime.date, end_date: datetime.date
) -> Dict[str, pandas.DataFrame | Dict[Any, Any]]:
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
    try:
        company_dict['info'] = company.info
    except:
        logger.warning('Could not retrieve the `info` dataframe.')

    # Get historical market data
    try:
        company_dict['history'] = company.history(start=start_date, end=end_date)
    except:
        logger.warning('Could not retrieve the `history` dataframe.')

    # Get meta information about the history (requires history() to be called first)
    company_dict['history_metadata'] = company.history_metadata

    # Get actions
    try:
        company_dict['actions'] = company.actions
    except:
        logger.warning('Could not retrieve the `actions` dataframe.')

    # Get dividends
    try:
        company_dict['dividends'] = company.dividends
    except:
        logger.warning('Could not retrieve the `dividends` dataframe.')

    # Get splits
    try:
        company_dict['splits'] = company.splits
    except:
        logger.warning('Could not retrieve the `splits` dataframe.')

    # Get capital gains
    try:
        company_dict['capital_gains'] = company.capital_gains  # only for mutual funds & etfs
    except:
        logger.warning('Could not retrieve the `capital gains` dataframe.')

    # Get share count
    try:
        company_dict['shares'] = company.get_shares_full(start=start_date, end=end_date)
    except:
        logger.warning('Could not retrieve the `shares` dataframe.')

    # Get financials
    # Get income statement
    try:
        company_dict['income_stmt'] = convert_index_to_column(company.income_stmt.T, 'Date')
    except:
        logger.warning('Could not retrieve the `income_stmt` dataframe.')

    # Get quarterly income statement
    try:
        company_dict['quarterly_income_stmt'] = convert_index_to_column(company.quarterly_income_stmt.T, 'Date')
    except:
        logger.warning('Could not retrieve the `quarterly_income_stmt` dataframe.')

    # Get balance sheet
    try:
        company_dict['balance_sheet'] = convert_index_to_column(company.balance_sheet.T, 'Date')
    except:
        logger.warning('Could not retrieve the `balance_sheet` dataframe.')

    # Get quarterly balance sheet
    try:
        company_dict['quarterly_balance_sheet'] = convert_index_to_column(company.quarterly_balance_sheet.T, 'Date')
    except:
        logger.warning('Could not retrieve the `quarterly_balance_sheet` dataframe.')

    # Get cash flow statement
    try:
        company_dict['cashflow'] = convert_index_to_column(company.cashflow.T, 'Date')
    except:
        logger.warning('Could not retrieve the `cashflow` dataframe.')

    # Get quarterly cash flow
    try:
        company_dict['quarterly_cashflow'] = convert_index_to_column(company.quarterly_cashflow.T, 'Date')
    except:
        logger.warning('Could not retrieve the `quarterly_cashflow` dataframe.')
    # see `Ticker.get_income_stmt()` for more options

    # Get major holders
    try:
        company_dict['major_holders'] = company.major_holders
    except:
        logger.warning('Could not retrieve the `major_holders` dataframe.')

    # Get institutional holders
    try:
        company_dict['institutional_holders'] = company.institutional_holders
    except:
        logger.warning('Could not retrieve the `institutional_holders` dataframe.')

    # Get mutual fund holders
    try:
        company_dict['mutualfund_holders'] = company.mutualfund_holders
    except:
        logger.warning('Could not retrieve the `mutualfund_holders` dataframe.')

    # Get insider transactions
    try:
        company_dict['insider_transactions'] = company.insider_transactions
    except:
        logger.warning('Could not retrieve the `insider_transactions` dataframe.')

    # Get insider purchases
    try:
        company_dict['insider_purchases'] = company.insider_purchases
    except:
        logger.warning('Could not retrieve the `insider_purchases` dataframe.')

    # Get insider sales
    try:
        company_dict['insider_roster_holders'] = company.insider_roster_holders
    except:
        logger.warning('Could not retrieve the `insider_roster_holders` dataframe.')

    # Get sustainability
    try:
        company_dict['sustainability'] = company.sustainability
    except:
        logger.warning('Could not retrieve the `sustainability` dataframe.')

    # Get recommendations
    try:
        company_dict['recommendations'] = company.recommendations
    except:
        logger.warning('Could not retrieve the `recommendations` dataframe.')

    # Get recommendations summary
    try:
        company_dict['recommendations_summary'] = company.recommendations_summary
    except:
        logger.warning('Could not retrieve the `recommendations` dataframe.')

    # Get upgrades downgrades
    try:
        company_dict['upgrades_downgrades'] = company.upgrades_downgrades
    except:
        logger.warning('Could not retrieve the `upgrades_downgrades` dataframe.')

    # Get future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
    # Note: If more are needed use company.get_earnings_dates(limit=XX) with increased limit argument.
    try:
        company_dict['earnings_dates'] = company.earnings_dates
    except:
        logger.warning('Could not retrieve the `earnings_dates` dataframe.')

    # Get ISIN code - *experimental*
    # ISIN = International Securities Identification Number
    try:
        company_dict['isin'] = company.isin
    except:
        logger.warning('Could not retrieve the `isin` dataframe.')

    # Get options expirations
    try:
        company_dict['options'] = company.options
    except:
        logger.warning('Could not retrieve the `options` dataframe.')

    # Get news
    try:
        company_dict['news'] = company.news
    except:
        logger.warning('Could not retrieve the `news` dataframe.')

    # # Get option chain for specific expiration
    try:
        company_dict['option_chain'] = company.option_chain()
    except:
        logger.warning('Could not retrieve the `option chain` dataframe.')
    # data available via: opt.calls, opt.puts

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
        df = pandas.Series(data, dtype=object).to_frame().T
        df.reset_index(inplace=True)

    elif isinstance(data, (list, tuple)):
        for item in data:
            item = json.dumps(item)
        df = pandas.DataFrame({df_name: data})

    elif isinstance(data, str):
        df = pandas.DataFrame({df_name: [data]})

    else:
        raise TypeError(f'Data type {type(data)} not supported.')

    # Sort the dataframe by date
    df = sort_dataframe_by_date(df)

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


def transform_string_to_list(input_string: str) -> List[str] | str:
    """
    Transforms a string representation of a list to an actual list.

    The input string is only changed if the string contains a valid list.

    Args:
        input_string: The input string to transform.

    Returns:
        The transformed list if a valid list is detected, otherwise the original string.
    """
    # Strip any leading/trailing whitespace
    input_string = input_string.strip()

    # Check if the string starts with '[' and ends with ']'
    if input_string.startswith('[') and input_string.endswith(']'):
        try:
            # Attempt to parse the string as a Python literal
            result = ast.literal_eval(input_string)

            # Check if the result is a list
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            # If parsing fails, return the original string
            pass

    # If no valid list is detected, return the original string
    return input_string


def coerce_str_to_list(input_string: str | List[str]) -> List[str]:
    """
    Coerce a string to a list of strings.

    Args:
        input_string: The string to be coerced.

    Returns:
        The coerced string as a list of strings.

    Raises:
        TypeError: If `input_string` is not of type string or list.
    """
    if isinstance(input_string, list):
        return input_string

    elif isinstance(input_string, str):
        # Remove leading and trailing whitespace
        s = input_string.strip()
        # Remove outer brackets if present
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        # Remove quotes and split by comma
        output_list = [item.strip().strip('\'"') for item in s.split(',')]

        return output_list

    else:
        raise TypeError(f'Input must be a string or a list of strings. Got {type(input_string)}.')


def time_llm(func: F) -> Any:
    """
    Decorator to measure the execution time of a function and log the duration.

    Args:
        func: The function to be decorated.

    Returns:
        The wrapped function that returns the original function's return value.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        Wrapper function to execute the timing logic.

        Args:
            *args: Variable length argument list for the decorated function.
            **kwargs: Arbitrary keyword arguments for the decorated function.

        Returns:
            The original function's return value.
        """
        # Create a scratchpad placeholder
        scratchpad = streamlit.empty()

        # Initialize time
        start_time = time.time()

        # Call the function
        result = func(*args, **kwargs)

        # End time
        end_time = time.time()

        # Compute duration
        duration = end_time - start_time

        # Log to file
        timing_logger.info(f'{func.__name__} took {duration:.2f} seconds.\n')

        # Stream duration to scratchpad
        scratchpad.markdown(
            f'- **<span style="color:rgb{SAMBANOVA_ORANGE}">{func.__name__}</span>'
            f'** took **<span style="color:rgb{SAMBANOVA_ORANGE}">{duration:.2f}</span>** seconds.',
            unsafe_allow_html=True,
        )

        # Return only result
        return result

    return wrapper


def sort_dataframe_by_date(df: pandas.DataFrame, column_name: Optional[str] = None) -> pandas.DataFrame:
    """
    Sort a pandas DataFrame by chronological dates.

    This function checks if the `datetime` format is present in any of the DataFrame's columns.
    If more than one column follows the datetime format, it prioritizes the column named 'Date'.
    If 'Date' is not present, it takes the first occurrence among the columns.
    The function then orders the rows by chronological dates from the earliest to the latest.

    Args:
        df: The input dataframe.
        column_name (Optional): The name of the datetime column to sort by. Defaults to None.

    Returns:
        The dataframe ordered by the specified datetime column.
    """
    # Initialize the variable to store the name of the datetime column to sort by
    datetime_col = column_name

    if datetime_col is None:
        # Iterate through the DataFrame columns to find datetime columns
        for col in df.columns:
            # Check if the column is of any datetime dtype
            if (
                pandas.api.types.is_datetime64_any_dtype(df[col])
                or pandas.to_datetime(df[col], errors='coerce').notna().any()
            ):
                # If the 'Date' column is found, select it
                if col == 'Date':
                    datetime_col = col
                    break
                # Otherwise, set the first datetime column found
                elif datetime_col is None:
                    datetime_col = col
                    break

    # If no datetime columns were found, log a message and return the original dataframe
    if datetime_col is None:
        logger.info('No datetime columns found in the DataFrame.')
        return df

    # Convert the selected datetime column to datetime type
    df[datetime_col] = pandas.to_datetime(df[datetime_col], errors='coerce')

    # Order the DataFrame by the datetime column
    df_sorted = df.sort_values(by=datetime_col).reset_index(drop=True)

    return df_sorted
