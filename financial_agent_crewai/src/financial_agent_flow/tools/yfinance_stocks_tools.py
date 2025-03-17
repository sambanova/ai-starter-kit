import datetime
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas
import pandas as pd
import yfinance
from crewai.tools import BaseTool
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from pandas.api.types import is_datetime64_any_dtype
from pandasai import Agent
from pydantic import BaseModel, Field

from financial_agent_crewai.src.financial_agent_flow.config import *
from financial_agent_crewai.src.financial_agent_flow.tools.general_tools import FilenameOutput, FilenameOutputList
from financial_agent_crewai.utils.utilities import create_yfinance_stock_dir

logger = logging.getLogger(__name__)
load_dotenv()

DATE_COLUMN = 'Date'
FILENAME = 'report_yfinance_stocks'
# How many days we extend the start date and end date for data retrieval
DATE_OFFSET = 3

MAX_DATA_SOURCES = 5

# Prompt template
SOURCES_PROMPT_TEMPLATE = """
Please review the following data sources and determine the minimal set of the most relevant sources
(including their exact column names) that best address the user query.

Query: {query}

• Return a minimum of 1 data source and a maximum of {max_data_sources} data sources.
• Sort the selected data sources from the most relevant to the least relevant.
• Do not invent or modify any data source names or column names.
• Only provide the data sources and the columns that exactly match the query requirements.
• Preserve the original data source and column names exactly as listed.

Data sources:\n\n
"""

PANDASAI_FORMAT_INSTRUCTIONS_CONVERSATIONAL = """
Please respond to the user's query in a clear, conversational manner.
Provide any necessary explanations and details, but do not include any plots or visualizations.
Query: {query}
"""


class YFinanceSource(BaseModel):
    """
    Represents a relevant data source from Yahoo Finance (YFinance)
    and the columns related to the user's query.
    """

    name: str = Field(..., description='The name or identifier of the data source.')
    columns: List[str] = Field(
        ...,
        description='The list of column names that might be relevant for the query, in their original spelling/casing.',
    )


class YFinanceSourceList(BaseModel):
    """
    A collection of YFinanceSource objects, each specifying a data source
    and its relevant columns, pertinent to the user's query.
    """

    sources: List[YFinanceSource] = Field(
        ...,
        min_length=1,
        max_length=MAX_DATA_SOURCES,
        description='A list of YFinanceSource objects representing potentially relevant data sources and columns.',
    )


class YFinanceStocksTool(BaseTool):  # type: ignore
    name: str = 'Yahoo Finance Stocks Tool'
    description: str = (
        'A tool that leverages the Yahoo Finance API '
        "to collect and analyze a company's financial and market data, "
        'enabling insight-driven decision-making.'
    )
    llm: BaseChatModel
    query: str
    ticker_symbol: str
    start_date: datetime.date
    end_date: datetime.date
    cache_dir: Path

    def _run(self) -> FilenameOutputList:
        """Execute the search query and return results."""

        # Extract data from yfinance
        data = extract_yfinance_data(
            self.ticker_symbol,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        # Create a prompt template
        PROMPT_TEMPLATE = SOURCES_PROMPT_TEMPLATE.format(query=self.query, max_data_sources=MAX_DATA_SOURCES)
        # Append the data sources and their corresponding columns
        for data_source in list(data):
            PROMPT_TEMPLATE += f'Data source: {data_source}.'
            PROMPT_TEMPLATE += data[data_source][1] + '\n\n'

        # Bind the schema to the model
        model_with_structure = self.llm.with_structured_output(YFinanceSourceList)

        # Invoke the model
        structured_output = model_with_structure.invoke(PROMPT_TEMPLATE)

        # Convert each DataFrame into a JSON string and store under its name
        answer_data_dict = dict()
        dataframe_dict = dict()
        meta_data_dict = dict()
        for count, source in enumerate(list(structured_output.sources)):  # type: ignore
            # Coerce the retrieved data to a `pandas.DataFrame`
            dataframe = convert_data_to_frame(data[source.name][0], source.name)
            # Retrieve the columns and exclude the columns that are not present in the data
            columns = [column for column in source.columns if column in dataframe.columns]
            # Retrieve the dataframe description, without the list of columns
            meta_data_dict[source.name] = data[source.name][1].strip('columns')
            # Store only the relevant columns
            dataframe_dict[source.name] = dataframe[columns]
            # Convert the selected dataframe columns to JSON
            answer_data_dict[source.name] = dataframe[columns].to_json(orient='split')
            # Break if we exceed the maximum number of data sources
            if count >= MAX_DATA_SOURCES:
                break

        # Dump all the dataframe JSON strings into one file
        filename_json = self.cache_dir / f'{FILENAME}_{self.ticker_symbol}_{self.start_date}_{self.end_date}.json'
        answer_data_dict_json = dict()
        for key, value in dataframe_dict.items():
            answer_data_dict_json[key] = apply_short_notation(value).to_json(orient='split')
        with open(filename_json, 'w') as f:
            json.dump(answer_data_dict_json, f, indent=2)

        # Answer the user query for the given ticker symbol
        answer = interrogate_dataframe_pandasai(
            dataframe_dict, self.query, self.llm, self.ticker_symbol, self.cache_dir
        )

        # Save the answer to a text file
        filename_txt = self.cache_dir / f'{FILENAME}_{self.ticker_symbol}_{self.start_date}_{self.end_date}.txt'
        with open(filename_txt, 'w') as f:
            f.write(answer)

        # Return both the PandasAI answer and the JSON file with the dataframe
        return FilenameOutputList(
            file_output_list=[FilenameOutput(filename=str(filename_txt)), FilenameOutput(filename=str(filename_json))]
        )


def interrogate_dataframe_pandasai(
    dataframe_dict: Dict[str, pandas.DataFrame],
    query: str,
    llm: BaseChatModel,
    ticker_symbol: str,
    cache_dir: Path,
) -> Any:
    """
    Interrogate multiple dataframes via `pandasai` with the user query.

    Args:
        dataframe_dict: A dictionary containing the dataframes to interrogate.
        query: The user query to answer with information from the dataframe.
        llm: The LLM to use when answering the user query.
        ticker_symbol: The ticker symbol to use to generate the output folder.

    Returns:
        The response to the user query, generated by the LLM via `pandasai`.
    """
    # Create the output folder name
    output_folder = create_yfinance_stock_dir(cache_dir) / f'{ticker_symbol}'

    # Extract the list of dataframes
    pandasai_dataframes_list = [dataframe.dropna() for dataframe in dataframe_dict.values()]

    # Create a new pandasai agent
    pandasai_agent = Agent(
        pandasai_dataframes_list,
        config={
            'llm': llm,
            'open_charts': False,
            'save_charts': True,
            'save_charts_path': str(output_folder / 'dataframes'),
            'enable_cache': False,
        },
    )

    # Generate the response to the user query in a conversational style
    query_conversational = query + '\nResponse: '
    answer_conversational = str(pandasai_agent.chat(PANDASAI_FORMAT_INSTRUCTIONS_CONVERSATIONAL.format(query=query)))
    if 'error' not in answer_conversational.lower():
        answer_conversational = query_conversational + answer_conversational
    else:
        answer_conversational = query_conversational

    # Calculate the maximum number of rows in the data lake
    max_data_length = 0
    for dataframe in dataframe_dict.values():
        max_data_length = max(max_data_length, dataframe.shape[0])
    # If there are no dataframes with more than 2 lines, we do no need plots
    if max_data_length <= 1:
        return answer_conversational
    else:
        # Creat the dataframe folder
        dataframe_folder = output_folder / 'dataframes'
        os.makedirs(dataframe_folder, exist_ok=True)

        # Plot the dataframes
        for data_source, dataframe in dataframe_dict.items():
            auto_plot(dataframe, data_source, dataframe_folder / data_source)

        # Return the concatenation of the two answers
        return answer_conversational


def extract_yfinance_data(
    ticker_symbol: str, start_date: datetime.date, end_date: datetime.date
) -> Dict[str, Tuple[Union[pandas.DataFrame, Dict[Any, Any], str, List[Any]], str]]:
    """
    Extracts all the data of a given company using Yahoo Finance for specified dates.

    Each entry of the returned dictionary is a tuple consisting of:
        - The extracted object (DataFrame, dict, string, list, etc.)
        - A brief description of what information that object contains,
            ending with the corresponding list of columns/key-names.

     Args:
        ticker_symbol: The ticker symbol of the company to extract data from.
        start_date: The start date of the historical price data to retrieve.
        end_date: The end date of the historical price data to retrieve.

     Returns:
        A dictionary where each key is a data-type label (e.g. "info", "history")
        and each value is a tuple (data_object, description_with_columns).

     Raises:
        TypeError: If `symbol` is not a string or `start_date` and `end_date` are not of type `datetime.date`.
    """
    # Check inputs
    if not isinstance(ticker_symbol, str):
        raise TypeError('Symbol must be a string.')
    if not isinstance(start_date, datetime.date):
        raise TypeError('Start date must be of type datetime.date.')
    if not isinstance(end_date, datetime.date):
        raise TypeError('End date must be of type datetime.date.')

    # To be on the safe side, we expand to business days
    # We have no trading data for weekends
    start_date = (pandas.Timestamp(start_date) - pandas.offsets.BDay(DATE_OFFSET)).date()
    end_date = (pandas.Timestamp(end_date) + pandas.offsets.BDay(DATE_OFFSET)).date()

    # Extract the data from Yahoo Finance for the given ticker symbol
    company = yfinance.Ticker(ticker=ticker_symbol)

    # Initialize the return dictionary
    company_dict: Dict[str, Tuple[Union[pandas.DataFrame, Dict[Any, Any], str, List[Any]], str]] = dict()

    def get_columns(data: Any) -> List[str]:
        """Get the names of the keys of a dictionary or of the columns of a dataframe."""

        if isinstance(data, pandas.DataFrame):
            return list(data.columns)
        elif isinstance(data, pandas.Series):
            return [data.name]
        elif isinstance(data, dict):
            return list(data.keys())
        else:
            raise TypeError('Data must be of type `pandas.DataFrame`, or `pandas.Series`, or `dict`.')

    # 1) Get all the stock information
    try:
        # Dictionary
        data = company.info
        description = (
            'This data source contains general metadata about the company, '
            'including name, sector, industry, and contact details. '
            "It also has information about the current stock prices (i.e. today's date or the latest date only)."
            f'Columns: {get_columns(data)}.'
        )
        company_dict['info'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `info` dictionary.')

    # 2) Get historical market data
    try:
        # DataFrame
        data = company.history(start=start_date, end=end_date)
        description = (
            'This data source contains all the historical stock prices for the requested date range. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['history'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `history` dataframe.')

    # 3) Get metadata about the history
    # Dictionary
    data = company.history_metadata
    description = (
        'This data source contains metadata about the retrieved historical data (e.g., refresh times, extents). '
        f'Columns: {get_columns(data)}.'
    )
    company_dict['history_metadata'] = (data, description)

    # 4) Get actions
    try:
        # DataFrame
        data = company.actions
        description = (
            'This data source shows the timeline of corporate actions (dividends and splits) if available, '
            'indexed by date. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['actions'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `actions` dataframe.')

    # 5) Get dividends
    try:
        # Series
        data = company.dividends
        description = (
            'This data source shows the dividend distribution history, indexed by date. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['dividends'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `dividends` dataframe.')

    # 6) Get splits
    try:
        # Series
        data = company.splits
        description = (
            'This data source shows stock split events for the requested date range, indexed by date. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['splits'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `splits` dataframe.')

    # 7) Get capital gains
    try:
        # Series
        data = company.capital_gains  # only for mutual funds & etfs
        description = (
            'This data source shows the capital gains distributions for mutual funds or ETFs if available, '
            'indexed by date. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['capital_gains'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `capital gains` dataframe.')

    # 8) Get share count
    try:
        # DataFrame
        data = company.get_shares_full(start=start_date, end=end_date)
        description = (
            'This data source shows the number of shares outstanding over time (if provided). '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['shares'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `shares` dataframe.')

    # 9) Get financials - income statement
    try:
        # DataFrame
        data = company.income_stmt.T
        description = (
            'This data source contains the annual income statement data, transposed per year/period. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['income_stmt'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `income_stmt` dataframe.')

    # 10) Get quarterly income statement
    try:
        # DataFrame
        data = company.quarterly_income_stmt.T
        description = (
            'This data source contains the quarterly income statement data, transposed per year/period. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['quarterly_income_stmt'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `quarterly_income_stmt` dataframe.')

    # 11) Get balance sheet
    try:
        # DataFrame
        data = company.balance_sheet.T
        description = (
            'This data source shows the annual balance sheet data, transposed per year/period. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['balance_sheet'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `balance_sheet` dataframe.')

    # 12) Get quarterly balance sheet
    try:
        # DataFrame
        data = company.quarterly_balance_sheet.T
        description = (
            'This data source shows the quarterly balance sheet data, transposed per year/period. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['quarterly_balance_sheet'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `quarterly_balance_sheet` dataframe.')

    # 13) Get cash flow statement
    try:
        # DataFrame
        data = company.cashflow.T
        description = (
            'This data source shows the annual cash flow statement, transposed per year/period. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['cashflow'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `cashflow` dataframe.')

    # 14) Get quarterly cash flow
    try:
        # DataFrame
        data = company.quarterly_cashflow.T
        description = (
            'This data source shows the quarterly cash flow statement, transposed per year/period. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['quarterly_cashflow'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `quarterly_cashflow` dataframe.')

    # 15) Get major holders
    try:
        # DataFrame
        data = company.major_holders
        description = (
            "This data source shows the major direct holders of the company's stocks if available. "
            f'Columns: {get_columns(data)}.'
        )
        company_dict['major_holders'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `major_holders` dataframe.')

    # 16) Get institutional holders
    try:
        # DataFrame
        data = company.institutional_holders
        description = (
            "This data source lists institutional holders of the company's stocks if available. "
            f'Columns: {get_columns(data)}.'
        )
        company_dict['institutional_holders'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `institutional_holders` dataframe.')

    # 17) Get mutual fund holders
    try:
        # DataFrame
        data = company.mutualfund_holders
        description = (
            "This data source lists mutual fund holders of the company's stocks if available. "
            f'Columns: {get_columns(data)}.'
        )
        company_dict['mutualfund_holders'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `mutualfund_holders` dataframe.')

    # 18) Get insider transactions
    try:
        # DataFrame
        data = company.insider_transactions
        description = (
            "This data source shows the insider transactions for the company's stocks. "
            f'Columns: {get_columns(data)}.'
        )
        company_dict['insider_transactions'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `insider_transactions` dataframe.')

    # 19) Get insider purchases
    try:
        # DataFrame
        data = company.insider_purchases
        description = f'This data source shows insider purchase activities. Columns: {get_columns(data)}.'
        company_dict['insider_purchases'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `insider_purchases` dataframe.')

    # 20) Get insider sales
    try:
        # DataFrame
        data = company.insider_roster_holders
        description = (
            'This data source shows the roster of insiders or insider holders if available. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['insider_roster_holders'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `insider_roster_holders` dataframe.')

    # 21) Get sustainability
    try:
        # DataFrame
        data = company.sustainability
        description = (
            'This data source contains environmental, social, and governance (ESG) metrics for the company. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['sustainability'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `sustainability` dataframe.')

    # 22) Get recommendations
    try:
        # DataFrame
        data = company.recommendations
        description = 'This data source shows broker recommendations for the company. ' f'Columns: {get_columns(data)}.'
        company_dict['recommendations'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `recommendations` dataframe.')

    # 23) Get recommendations summary
    try:
        # DataFrame
        data = company.recommendations_summary
        description = (
            'This data source provides a summary of broker recommendations for the company. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['recommendations_summary'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `recommendations_summary` dataframe.')

    # 24) Get upgrades/downgrades
    try:
        # DataFrame
        data = company.upgrades_downgrades
        description = f'This data source tracks analyst upgrades and downgrades. Columns: {get_columns(data)}.'
        company_dict['upgrades_downgrades'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `upgrades_downgrades` dataframe.')

    # 25) Get future and historic earnings dates
    try:
        # DataFrame
        data = company.earnings_dates
        description = (
            'This data source shows the upcoming and past earnings dates, including reported EPS and estimates. '
            f'Columns: {get_columns(data)}.'
        )
        company_dict['earnings_dates'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `earnings_dates` dataframe.')

    # 26) Get ISIN code
    try:
        # String
        data = company.isin
        description = "This string represents the company's International Securities Identification Number (ISIN). "
        company_dict['isin'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `isin` code.')

    # 27) Get options expirations
    try:
        # List
        data = list(company.options)
        description = (
            "This list shows the available option expiration dates for the company's stocks, indexed by date. "
        )
        company_dict['options'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `options` list.')

    # 28) Get news
    try:
        # List
        data = company.news
        description = 'This list shows recent news stories related to the company. '
        company_dict['news'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `news` dataframe.')

    # 29) Get option chain for specific expiration
    try:
        # Object
        data = company.option_chain()
        description = (
            'This object contains the calls and puts DataFrames for the specified (default/latest) option expiration. '
        )
        company_dict['option_chain'] = (data, description)
    except Exception:
        logger.warning('Could not retrieve the `option chain` dataframe.')

    return company_dict


def convert_data_to_frame(data: Any, df_name: str) -> pandas.DataFrame:
    """
    Converts data to `pandas.DataFrame`.

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

    # Drom NA
    df.dropna(axis=1, how='all', inplace=True)

    # Sort values by date
    df = sort_dataframe_by_date(df)

    return df


def sort_dataframe_by_date(df: pandas.DataFrame, column_name: Optional[str] = None) -> pandas.DataFrame:
    """
    Sort a pandas DataFrame by chronological dates.

    This function checks if the provided `column_name` is a datetime column.
    If no column_name is provided, it will:
      1. Check if the index is a datetime index and sort by the index if so.
      2. If the index is not a datetime index, it looks through the columns
        to find a suitable datetime column to sort by. It will prioritize
        a column named 'Date', if present, otherwise it uses the first
        column that can be parsed as datetime.

    Args:
        df: The input DataFrame to sort.
        column_name: The name of a specific datetime column to sort by. Defaults to None.

    Returns:
        A new DataFrame sorted chronologically either by the specified column,
        the datetime index, or a detected datetime column.
    """
    # If a specific column was provided, try sorting by that
    if column_name is not None:
        if column_name not in df.columns:
            logger.info(f"The specified column '{column_name}' does not exist in the DataFrame.")
            return df
        # Attempt to convert to datetime
        df[column_name] = pandas.to_datetime(df[column_name], errors='coerce')
        if df[column_name].notna().any():
            return df.sort_values(by=column_name).reset_index(drop=True)
        else:
            logger.info(f"The specified column '{column_name}' cannot be converted to datetime.")
            return df

    # If no column_name was provided, check if the index is a datetime index
    if isinstance(df.index, pandas.DatetimeIndex):
        return df.sort_index()

    # Otherwise, look for a suitable datetime column (prioritizing "Date")
    datetime_col = None
    for col in df.columns:
        # Check if the column is of any datetime dtype or can be coerced to datetime
        if pandas.api.types.is_datetime64_any_dtype(df[col]):
            if col == 'Date':
                datetime_col = col
                break
            elif datetime_col is None:
                datetime_col = col

    # If no datetime columns were found, log a message and return the original DataFrame
    if datetime_col is None:
        logger.info('No datetime columns found in the DataFrame, and index is not a datetime index.')
        return df

    # Convert the selected column to datetime and sort
    df[datetime_col] = pandas.to_datetime(df[datetime_col], errors='coerce')
    df_sorted = df.sort_values(by=datetime_col).reset_index(drop=True)
    return df_sorted


def is_unhashable(input_object: Any) -> bool:
    """Determine whether an object is unhashable or not."""

    try:
        # Attempt to call hash() on the object
        hash(input_object)
    except TypeError:
        # If a TypeError occurs, the object is unhashable
        return True
    return False


def auto_plot(df: pandas.DataFrame, df_name: str, output_path: str | Path) -> None:
    """
    Automatically choose and create a Matplotlib plot based on the contents of a DataFrame.

    Logic Overview:
    1. Identify all columns that contain numeric data. This is done by attempting to convert
       their non-null values to numeric (do not rely solely on the data type).
       Exclude any columns detected as datetimes.
    2. Check if the DataFrame's index is datetime-like, or if there is a column with
       datetime values:
       a) If yes, plot the numeric columns against that datetime data (line plot).
          The x-axis is formatted for dates in a human-readable way.
    3. When there is no date in the index or columns:
        a) If only one numeric column is found:
            - If its sum is approximately 1 or 100, make a pie chart (indicating proportions).
            - Otherwise, make a bar plot.
        b) If multiple numeric columns are found:
            - Make a line plot if multiple rows are detected.
            - Make a barplot if a single row is detected.
    4. The figure is titled using 'df_name' and saved to 'output_path'.

    Args:
        df: The DataFrame to plot.
        df_name: A descriptive name or title for the DataFrame; used as the plot's title.
        output_path: File path (including file extension) where the plot is saved.
    """
    # Check if the index is datetime-like
    is_index_datetime = pandas.api.types.is_datetime64_any_dtype(df.index)

    # Identify a datetime column if the index is not datetime
    date_col = None
    if not is_index_datetime:
        for col in df.columns:
            if pandas.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break

    # Identify numeric columns (by attempting conversion), excluding datetime columns
    numeric_cols = [
        col
        for col in df.columns
        if is_numeric_series(df[col]) and not pandas.api.types.is_datetime64_any_dtype(df[col])
    ]

    # If there are no truly numeric columns, do nothing
    if not numeric_cols:
        return

    df_filtered = df.dropna(how='all')

    fig, ax = plt.subplots()

    # If there's a date index or a date column, make a line plot using dates on x-axis
    if is_index_datetime or date_col is not None:
        if is_index_datetime:
            # Plot numeric columns against the datetime index
            df_filtered[numeric_cols].plot(ax=ax, title=df_name)
            ax.set_xlabel('Date')
            # Enforce that the x-axis starts at the minimum x-value (and ends at max)
            ax.set_xlim(left=min(df_filtered.index), right=max(df_filtered.index))
        else:
            # Plot numeric columns using the identified date column as x-axis
            df_filtered[[*numeric_cols, date_col]].plot(x=date_col, ax=ax, title=df_name)
            # Enforce that the x-axis starts at the minimum x-value (and ends at max)
            ax.set_xlim(left=min(df_filtered[date_col]), right=max(df_filtered[date_col].index))
            ax.set_xlabel('Date')

    else:
        # No date in index or columns
        if len(numeric_cols) == 1:
            # Single numeric column
            col = numeric_cols[0]
            total = df_filtered[col].sum()
            # If only one element
            if len(df_filtered[col]) == 1:
                return
            # If sum is near 1 or 100, make a pie chart. Otherwise, bar chart
            if abs(total - 1) < 1e-7 or abs(total - 100) < 1e-7:
                df_filtered[col].plot(kind='pie', title=df_name, autopct='%1.1f%%', ax=ax, ylabel='')
            else:
                df_filtered[col].plot(kind='bar', title=df_name, ax=ax)

        else:
            if df_filtered.shape[0] > 1:
                # Multiple rows => line plot
                df_filtered[numeric_cols].plot(kind='line', title=df_name, ax=ax)
            else:
                # Single row => bar plot
                df_filtered[numeric_cols].plot(kind='bar', title=df_name, ax=ax)

    # Save the figure and close
    plt.savefig(output_path)
    plt.close(fig)


def is_numeric_series(series: pandas.Series) -> bool:
    """
    Determine if a pandas Series can be interpreted as numeric (float or int).
    ignoring missing values by attempting to convert its non-null entries to numbers.

    Args:
        series: The input Series to check.

    Returns:
        True if the non-null entries can all be converted to numeric, False otherwise.
    """
    try:
        pandas.to_numeric(series.dropna(), errors='raise')
        return True
    except (ValueError, TypeError):
        return False


def apply_short_notation(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert numeric values in specified columns of a `pandas.DataFrame` to:
      • Short notation (e.g. “1.50K”, “3.20M”) if abs(value) ≥ 1000,
      • Two-decimal standard form if 0.01 ≤ abs(value) < 1000,
      • Scientific notation if abs(value) < 0.01 (e.g. “0.002” -> “2e-3”).

    Datetime columns are left unchanged.

    Returns a copy of the DataFrame with updated string representations.

    Args:
        df: The DataFrame whose numeric values will be converted.
        columns: The columns to convert. If None, all numeric columns are used.

    Returns:
        A copy of the DataFrame with converted string values.
    """

    SUFFIXES = [
        ('T', 1e12),
        ('B', 1e9),
        ('M', 1e6),
        ('K', 1e3),
    ]

    def format_scientific(value: float, decimals: int = 2) -> str:
        """
        Convert a number to scientific notation (e.g., 0.002 → “2e-3”)
        with up to 'decimals' digits in the mantissa (trailing zeros removed).
        """
        if value == 0:
            return '0.00'
        negative = value < 0
        value = abs(value)

        exp = math.floor(math.log10(value))
        mantissa = value / 10**exp

        # Round mantissa, then strip trailing zeros
        mantissa_rounded = round(mantissa, decimals)
        mantissa_str = f'{mantissa_rounded}'.rstrip('0').rstrip('.')

        sign_str = '-' if negative else ''
        return f'{sign_str}{mantissa_str}e{exp}'

    def human_format(x: float) -> str:
        """
        Convert a single numeric value to:
          • Short notation if abs(x) ≥ 1000,
          • Two-decimal “regular” representation if 0.01 ≤ abs(x) < 1000,
          • Scientific notation if abs(x) < 0.01,
          • Empty string if NaN or non-numeric.
        """
        if pd.isnull(x):
            return ''
        try:
            num = float(x)
        except (ValueError, TypeError):
            # Non-numeric -> return original as string
            return str(x)

        if num == 0:
            return '0.00'

        abs_num = abs(num)
        negative = num < 0

        # Very small -> scientific notation
        if abs_num < 0.01:
            return format_scientific(num, 2)

        # Large -> short notation
        for suffix, threshold in SUFFIXES:
            if abs_num >= threshold:
                scaled = abs_num / threshold
                scaled_str = f'{scaled:.2f}'
                out = f'{scaled_str}{suffix}'
                return f'-{out}' if negative else out

        # Otherwise, two-decimal format
        rounded_str = f'{abs_num:.2f}'
        return f'-{rounded_str}' if negative else rounded_str

    # Make a copy so the original DataFrame is unmodified
    df_copy = df.copy()

    # If no columns are specified, pick all numeric (and non-datetime) columns
    if columns is None:
        columns = [
            col
            for col in df_copy.columns
            if is_numeric_series(df_copy[col]) and not is_datetime64_any_dtype(df_copy[col])
        ]

    # Apply the formatting function to the specified columns
    for col in columns:
        df_copy[col] = df_copy[col].apply(human_format)

    return df_copy
