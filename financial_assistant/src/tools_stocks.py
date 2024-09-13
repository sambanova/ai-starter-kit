import datetime
from datetime import timedelta
from typing import Any, Dict, List, Optional

import pandas
import streamlit
import yfinance
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from matplotlib import dates as mdates
from matplotlib import pyplot
from matplotlib.figure import Figure
from pandasai import SmartDataframe
from pandasai.connectors.yahoo_finance import YahooFinanceConnector

from financial_assistant.prompts.pandasai_prompts import PLOT_INSTRUCTIONS
from financial_assistant.src.tools import (
    coerce_str_to_list,
    convert_data_to_frame,
    extract_yfinance_data,
    time_llm,
)
from financial_assistant.streamlit.constants import *


class StockInfoSchema(BaseModel):
    """Tool for retrieving accurate stock information for a list of companies using the specified dataframe name."""

    user_query: str = Field(..., description='User query to retrieve stock information.')
    company_list: List[str] | str = Field(..., description='List of required companies.')
    dataframe_name: Optional[str] = Field(None, description='Name of the dataframe to be used, if applicable.')


class TickerSymbol(BaseModel):
    """Model for the stock ticker symbol of a specified company."""

    symbol: str = Field(..., description='The ticker symbol of the company.')


@tool(args_schema=StockInfoSchema)
def get_stock_info(
    user_query: str, company_list: List[str] | str, dataframe_name: Optional[str] = None
) -> Dict[str, str]:
    """
    Tool for retrieving accurate stock information for a list of companies using the specified dataframe name.

    Args:
        user_query: User query to retrieve stock information.
        company_list: List of required companies.
        dataframe_name: Name of the dataframe to be used.

    Returns:
        Dictionary containing the correct stock information (value) for a given ticker symbol (key).

    Raises:
        TypeError: If `user_query` is not a string or `symbol_list` is not a list of strings,
            or `dataframe_name` is not a string.
    """

    # Checks the inputs
    assert isinstance(user_query, str), TypeError(f'User query must be of type string. Got {(type(user_query))}.')
    assert isinstance(company_list, (list, str)), TypeError(
        f'`company_list` must be of type list or string. Got {(type(company_list))}.'
    )

    # If `symbol_list` is a string, coerce it to a list of strings
    company_list = coerce_str_to_list(company_list)

    assert all([isinstance(name, str) for name in company_list]), TypeError(
        '`company_names_list` must be a list of strings.'
    )

    # Retrieve the list of ticker symbols
    symbol_list = retrieve_symbol_list(company_list)

    assert isinstance(dataframe_name, str | None), TypeError(
        f'Dataframe name must be a string. Got {(type(dataframe_name))}.'
    )

    # Retrieve the correct stock information
    response_dict: Dict[str, str] = get_stock_info_from_dataframe(user_query, symbol_list, dataframe_name)

    return response_dict


def get_stock_info_from_dataframe(
    user_query: str,
    symbol_list: List[str],
    dataframe_name: str | None = None,
) -> Dict[str, Any]:
    """
    Interrogate the dataframe via `pandasai` to retrieve the correct stock information.

    Args:
        user_query: String containing the user query.
        symbol_list: List of strings containing the ticker symbols.
        dataframe_name: The name of the dataframe that contains the stock information.

    Returns:
        A dictionary containing the stock information for each company symbol.
    """
    response_dict = dict()
    for symbol in symbol_list:
        # If the user has not provided a dataframe name then use the generic yfinance langchain connector
        if dataframe_name is None or dataframe_name == 'None':
            response_dict[symbol] = get_yahoo_connector_answer(user_query, symbol)
        # If the user has provided a dataframe name then use the custom langchain connector
        else:
            response_dict[symbol] = get_pandasai_answer_from_dataframe(user_query, symbol, dataframe_name)

    return response_dict


def get_pandasai_answer_from_dataframe(user_query: str, symbol: str, dataframe_name: str) -> Any:
    """
    Get the relevant stock information by querying the corresponding dataframe via `pandasai`.

    Args:
        user_query: The user query to answer.
        symbol: The ticker symbol for which to retrieve the relevant stock information.
        dataframe_name: The name of the dataframe that contains the stock information.

    Returns:
        The relevant stock information answering the user query for the given company symbol.
    """
    # Extract the relevant yfinance data
    company_data_dict = extract_yfinance_data(
        symbol,
        start_date=datetime.datetime.today().date() - timedelta(days=365),
        end_date=datetime.datetime.today().date(),
    )

    # Extract the relevant dataframe from the yfinance data dictionary
    data = company_data_dict[dataframe_name]

    # Coerce the retrieved data to a `pandas.DataFrame`
    dataframe = convert_data_to_frame(data, dataframe_name)

    try:
        # Answer the user query by symbol
        return interrogate_dataframe_pandasai(dataframe, user_query)
    except:
        # If the answer could not be generated, then use the generic yfinance langchain connector
        return get_yahoo_connector_answer(user_query, symbol)


def get_yahoo_connector_answer(user_query: str, symbol: str) -> Any:
    """
    Answer the user query using the generic `pandasai.connectors.yahoo_finance.YahooFinanceConnector`.

    Args:
        user_query: The user query to answer.
        symbol: The company symbol for which to retrieve data from yfinance.

    Returns:
        The response generated by the connector.

    Raises:
        TypeError: If `user_query` is not a string or symbol is not a string.
    """
    # Checks the inputs
    assert isinstance(user_query, str), TypeError(f'The user query must be of type string. Got {(type(user_query))}.')
    assert isinstance(symbol, str), TypeError(f'The company symbol must be of type string. Got {(type(symbol))}.')

    # Instantiate a `pandasai.connectors.yahoo_finance.YahooFinanceConnector` object with the relevant symbol
    yahoo_connector = YahooFinanceConnector(symbol)

    # Answer the user query by symbol
    return interrogate_dataframe_pandasai(yahoo_connector, user_query)


@time_llm
def interrogate_dataframe_pandasai(df_pandas: pandas.DataFrame, user_query: str) -> Any:
    """
    Interrogate a dataframe via `pandasai` with the user query.

    Args:
        df_pandas: The dataframe to interrogate.
        user_query: The user query to answer with information from the dataframe.

    Returns:
        The response to the user query, generated by the LLM via `pandasai`.
    """

    # Instantiate a `pandasai.SmartDataframe` object with the relevant `pandas.DataFrame` and user query
    df = SmartDataframe(
        df_pandas,
        config={
            'llm': streamlit.session_state.llm.llm,
            'open_charts': False,
            'save_charts': True,
            'save_charts_path': streamlit.session_state.stock_query_figures_dir,
            'enable_cache': False,
        },
    )

    # Add the plot instructions to the user query
    final_query = user_query + '\n' + PLOT_INSTRUCTIONS

    return df.chat(final_query)


@time_llm
def retrieve_symbol_list(company_names_list: List[str] | str = list()) -> List[str]:
    """
    Retrieve a list of ticker symbols.

    Args:
        company_names_list: List of company names to search for in the JSON file.

    Returns:
        Tuple of a list of ticker symbols and the user query.

    Raises:
        TypeError: If `company_names_list` is not of type list or string.
    """
    # Check inputs
    assert isinstance(company_names_list, (list, str)), TypeError(
        f'company_names_list` names must be a list of strings. Got {type(company_names_list)}.'
    )

    # If `symbol_list` is a string, coerce it to a list of strings
    company_names_list = coerce_str_to_list(company_names_list)

    assert len(company_names_list) > 0, ValueError('`company_names_list` cannot be empty.')

    assert all([isinstance(name, str) for name in company_names_list]), TypeError(
        '`company_names_list` must be a list of strings.'
    )

    symbol_list = list()

    # Iterate over the JSON data to extract ticker symbols
    for company in company_names_list:
        # The prompt template
        prompt_template_symbol = (
            'What is the ticker symbol for {company}?\n' 'Format instructions: {format_instructions}'
        )

        # The parser
        parser_symbol = PydanticOutputParser(pydantic_object=TickerSymbol)  # type: ignore

        # The prompt
        prompt_symbol = PromptTemplate(
            template=prompt_template_symbol,
            input_variables=['company'],
            partial_variables={'format_instructions': parser_symbol.get_format_instructions()},
        )

        # The chain
        chain_symbol = prompt_symbol | streamlit.session_state.llm.llm | parser_symbol

        # Invoke the chain to derive the ticker symbol of the company
        symbol = chain_symbol.invoke(company).symbol
        symbol_list.append(symbol)

    return list(set(symbol_list))


class HistoricalPriceSchema(BaseModel):
    """Tool for fetching historical stock prices for a given list of companies from `start_date` to `end_date`."""

    company_list: List[str] | str = Field(
        ..., description='List of required companies.', examples=['Google', 'Microsoft']
    )
    start_date: datetime.date = Field(
        description='The start date for retrieving historical prices.'
        + 'Default to "2000-01-01" if the date is vaguely requested. Must be before the end date.',
    )
    end_date: datetime.date = Field(
        description='The end date for retrieving historical prices.'
        + 'Typically today unless a specific end date is provided. Must be greater than the start date.',
    )
    quantity: str = Field(
        'Close',
        description='The specific quantity to analyze. Defaults to "Close" if not specified.',
        examples=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'],
    )


@tool(args_schema=HistoricalPriceSchema)
def get_historical_price(
    company_list: List[str] | str, start_date: datetime.date, end_date: datetime.date, quantity: str = 'Close'
) -> pandas.DataFrame:
    """
    Tool for fetching historical stock prices for a given list of companies from `start_date` to `end_date`.

    Args:
        company_list: List of required companies.
        end_date: Typically today unless a specific end date is provided. End date MUST be greater than start date
        start_date: Set explicitly, or calculated as 'end_date - date interval'
            (for example, if prompted 'over the past 6 months',
            date interval = 6 months so start_date would be 6 months earlier than today's date).
            Default to '2000-01-01' if vaguely asked for historical price.
            Start date must always be before the current date.
        quantity: Quantity to analize. Default is `Close`.

    Returns:
        A tuple with the following elements:
            - The figure with historical price data.
            - A dataframe with historical price data.
            - The list of company ticker symbols.
    """
    assert isinstance(company_list, (list, str)), TypeError(
        f'`company_list` must be of type list or string. Got {(type(company_list))}.'
    )
    assert start_date <= end_date or (end_date - datetime.timedelta(days=365)) >= start_date, ValueError(
        'Start date must be before the end date.'
    )

    # If `symbol_list` is a string, coerce it to a list of strings
    company_list = coerce_str_to_list(company_list)

    assert all([isinstance(name, str) for name in company_list]), TypeError(
        '`company_names_list` must be a list of strings.'
    )

    # Retrieve the list of ticker symbols
    symbol_list = retrieve_symbol_list(company_list)

    # Retrive the historical price data
    data_price = download_data_history(symbol_list, quantity, start_date, end_date)

    # Plot the historical data
    fig = plot_price_over_time(data_price)
    return fig, data_price, symbol_list


def download_data_history(
    symbol_list: List[str],
    quantity: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> pandas.DataFrame:
    """
    Download historical price data from Yahoo Finance.

    Args:
        symbol_list: A list of company ticker symbols.
        quantity: Quantity to analize.
        start_date: The start of the time period.
        end_date: The end of the time period.

    Returns:
        A `pandas.DataFrame` object containing the historical price data for each symbol.
    """
    # Initialise a pandas DataFrame with symbols as columns and dates as index
    data_price = pandas.DataFrame(columns=symbol_list)

    # Fetch historical price data from Yahoo Finance
    for symbol in symbol_list:
        data = yfinance.Ticker(symbol)
        data_history = data.history(start=start_date, end=end_date)
        data_price[symbol] = data_history[quantity]

    return data_price


def plot_price_over_time(data_close: pandas.DataFrame) -> Figure:
    """Plot the historical data over time."""

    full_df = pandas.DataFrame(columns=['Date'])
    for column in data_close:
        full_df = full_df.merge(data_close[column], on='Date', how='outer')

    # Create a matplotlib figure
    fig, ax = pyplot.subplots(figsize=(12, 6))

    # Dynamically plot each stock symbol in the DataFrame
    for column in full_df.columns[1:]:  # Skip the first column since it's the date
        ax.plot(full_df['Date'], full_df[column], label=column, marker='o', markersize=2, linestyle='-', linewidth=1.5)

    # Set title and labels
    ax.set_title('Stock Price Over Time: ' + ', '.join(full_df.columns.tolist()[1:]))
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price (USD)')

    from matplotlib.ticker import FuncFormatter

    # Format y-axis ticks
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # type: ignore
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # type: ignore

    # Rotate and align the tick labels so they look better
    pyplot.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.75)
    ax.grid(True, axis='y', linestyle='--', alpha=0.75)

    # Customize colors
    ax.set_facecolor('lightgray')
    fig.patch.set_facecolor('lightgray')

    # Add legend
    ax.legend(title='Stock Symbol', loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to prevent cutting off labels
    pyplot.tight_layout()

    return fig
