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


class StockInfoSchema(BaseModel):
    """Return the correct stock info value given the appropriate symbol and key.
    Infer valid ticker symbol and key from the user prompt.
    The key must be one of the following:
    address1, city, state, zip, country, phone, website, industry, industryKey, industryDisp, sector, sectorKey,
    sectorDisp, longBusinessSummary, fullTimeEmployees, companyOfficers, auditRisk, boardRisk, compensationRisk,
    shareHolderRightsRisk, overallRisk, governanceEpochDate, compensationAsOfEpochDate, maxAge, priceHint,
    previousClose, open, dayLow, dayHigh, regularMarketPreviousClose, regularMarketOpen, regularMarketDayLow,
    regularMarketDayHigh, dividendRate, dividendYield, exDividendDate, beta, trailingPE, forwardPE, volume,
    regularMarketVolume, averageVolume, averageVolume10days, averageDailyVolume10Day, bid, ask, bidSize, askSize,
    marketCap, fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months, fiftyDayAverage, twoHundredDayAverage,
    currency, enterpriseValue, profitMargins, floatShares, sharesOutstanding, sharesShort, sharesShortPriorMonth,
    sharesShortPreviousMonthDate, dateShortInterest, sharesPercentSharesOut, heldPercentInsiders,
    heldPercentInstitutions,shortRatio, shortPercentOfFloat, impliedSharesOutstanding, bookValue, priceToBook,
    lastFiscalYearEnd, nextFiscalYearEnd, mostRecentQuarter, earningsQuarterlyGrowth, netIncomeToCommon, trailingEps,
    forwardEps, pegRatio, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange, SandP52WeekChange, lastDividendValue,
    lastDividendDate, exchange, quoteType, symbol, underlyingSymbol, shortName, longName, firstTradeDateEpochUtc,
    timeZoneFullName, timeZoneShortName, uuid, messageBoardId, gmtOffSetMilliseconds, currentPrice, targetHighPrice,
    targetLowPrice, targetMeanPrice, targetMedianPrice, recommendationMean, recommendationKey, numberOfAnalystOpinions,
    totalCash, totalCashPerShare, ebitda, totalDebt, quickRatio, currentRatio, totalRevenue, debtToEquity,
    revenuePerShare, returnOnAssets, returnOnEquity, freeCashflow, operatingCashflow, earningsGrowth, revenueGrowth,
    grossMargins, ebitdaMargins, operatingMargins, financialCurrency, trailingPegRatio.
    If asked generically for 'stock price', use currentPrice.
    """

    user_query: str = Field('User query to retrieve stock information.')
    symbol_list: List[str] = Field('List of stock ticker symbols.')
    keys: List[str] = Field('List of keys to be retrieved or inferred from the user prompt.')


@tool(args_schema=StockInfoSchema)
def get_stock_info(user_query: str, symbol_list: List[str] = list(), keys: List[str] = list()) -> Dict[str, str]:
    """Return the correct stock info value given the appropriate symbol and key.
    Infer valid ticker symbol and key from the user prompt.
    The key must be one of the following:
    address1, city, state, zip, country, phone, website, industry, industryKey, industryDisp, sector, sectorKey,
    sectorDisp, longBusinessSummary, fullTimeEmployees, companyOfficers, auditRisk, boardRisk, compensationRisk,
    shareHolderRightsRisk, overallRisk, governanceEpochDate, compensationAsOfEpochDate, maxAge, priceHint,
    previousClose, open, dayLow, dayHigh, regularMarketPreviousClose, regularMarketOpen, regularMarketDayLow,
    regularMarketDayHigh, dividendRate, dividendYield, exDividendDate, beta, trailingPE, forwardPE, volume,
    regularMarketVolume, averageVolume, averageVolume10days, averageDailyVolume10Day, bid, ask, bidSize, askSize,
    marketCap, fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months, fiftyDayAverage, twoHundredDayAverage,
    currency, enterpriseValue, profitMargins, floatShares, sharesOutstanding, sharesShort, sharesShortPriorMonth,
    sharesShortPreviousMonthDate, dateShortInterest, sharesPercentSharesOut, heldPercentInsiders,
    heldPercentInstitutions,shortRatio, shortPercentOfFloat, impliedSharesOutstanding, bookValue, priceToBook,
    lastFiscalYearEnd, nextFiscalYearEnd, mostRecentQuarter, earningsQuarterlyGrowth, netIncomeToCommon, trailingEps,
    forwardEps, pegRatio, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange, SandP52WeekChange, lastDividendValue,
    lastDividendDate, exchange, quoteType, symbol, underlyingSymbol, shortName, longName, firstTradeDateEpochUtc,
    timeZoneFullName, timeZoneShortName, uuid, messageBoardId, gmtOffSetMilliseconds, currentPrice, targetHighPrice,
    targetLowPrice, targetMeanPrice, targetMedianPrice, recommendationMean, recommendationKey, numberOfAnalystOpinions,
    totalCash, totalCashPerShare, ebitda, totalDebt, quickRatio, currentRatio, totalRevenue, debtToEquity,
    revenuePerShare, returnOnAssets, returnOnEquity, freeCashflow, operatingCashflow, earningsGrowth, revenueGrowth,
    grossMargins, ebitdaMargins, operatingMargins, financialCurrency, trailingPegRatio.
    If asked generically for 'stock price', use currentPrice.
    """
    stock_key: Dict[str, str] = dict()

    for symbol in symbol_list:
        try:
            for key in keys:
                data = yfinance.Ticker(symbol)
                stock_info = data.info
                stock_key[symbol] = stock_key[symbol] + key + ': ' + str(stock_info[key]) + '. '
        except Exception as e:
            yahoo_connector = YahooFinanceConnector(symbol)

            df = SmartDataframe(yahoo_connector, config={'llm': streamlit.session_state.fc.llm})
            stock_key = df.chat(user_query)

    return stock_key


class RetrievalCompanyNameSchema(BaseModel):
    """
    Retrieve a list of company names.
    """

    company_names_list: List[str] = Field(
        'List of company names to retrieve from the query.', example=['Apple', 'Microsoft']
    )
    user_query: str = Field(
        'User query or quantities to search for company names. If you cannot retrieve it, return an empty string.',
    )


def is_similar_string(sub: Optional[str], string: Optional[str], threshold: int = 80) -> bool:
    """Checks if two strings are similar.

    Returns the similarity between two strings.
    :param sub: Substring to be compared with.
    :param string: String to be compared
    :param threshold: Threshold to compare the similarity between two strings.
    :return: The similarity of two strings.
    """
    if sub is not None and string is not None and len(string) > 0 and len(sub) > 0:
        ratio = fuzz.partial_ratio(sub.lower(), string.lower())
        if isinstance(ratio, (int, float)):
            return ratio >= threshold
    return False


def get_ticker_by_title(json_data: Dict[str, Dict[str, str]], company_title: str) -> Optional[str]:
    """
    Retrieve a ticker by company title.
    :param json_data: JSON data to search through.
    :param company_title: Company name to search for in the JSON file.
    :return: Ticker of company if found, else returns none.
    """
    for key, value in json_data.items():
        if is_similar_string(company_title, value.get('title')) or is_similar_string(value.get('title'), company_title):
            return value.get('ticker')
    return None


@tool(args_schema=RetrievalCompanyNameSchema)
def retrieve_symbol_list(company_names_list: List[str], user_query: str) -> Tuple[List[Optional[str]], str]:
    """Retrieve a list of ticker symbols."""

    # URL to the JSON file
    url = 'https://www.sec.gov/files/company_tickers.json'

    # Fetch the JSON data from the URL
    response = requests.get(url)

    symbol_list = list()

    # Specify the file path where you want to save the JSON data
    file_path = kit_dir + '/company_tickers.json'

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Save the DataFrame to a json file
        if not os.path.exists(kit_dir):
            os.makedirs(kit_dir)

        # Write the JSON data to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    else:
        # Load JSON data from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Iterate over the JSON data to extract ticker symbols
    for company in company_names_list:
        ticker = get_ticker_by_title(data, company)
        symbol_list.append(ticker)

    return symbol_list, user_query


class RetrievalSymbolQuantitySchema(BaseModel):
    """
    Retrieve a list of ticker symbols and the quantity that the user wants to analyze.

    The quantity must be one of the following:

    Open, High, Low, Close, Volume, Dividends, Stock Splits.
    If you can't retrieve the quantity, use 'Close'.
    """

    symbol_list: List[str] = Field('List of stock ticker symbols', example=['AAPL', 'MSFT'])
    quantity: str = Field('Quantity to analize', example=['Open', 'Close'])


@tool(args_schema=RetrievalSymbolQuantitySchema)
def retrieve_symbol_quantity_list(symbol_list: List[str], quantity: str) -> Tuple[List[str], str]:
    """Retrieve a list of ticker symbols and the quantity that the user wants to analyze."""

    return symbol_list, quantity


class RetrievalSymbolSchema(BaseModel):
    """
    Get the finanical summary for a given list of ticker symbols.
    The balance sheet has the following fields:
    'Ordinary Shares Number', 'Share Issued', 'Net Debt', 'Total Debt',
    'Tangible Book Value', 'Invested Capital', 'Working Capital',
    'Net Tangible Assets', 'Capital Lease Obligations',
    'Common Stock Equity', 'Total Capitalization',
    'Total Equity Gross Minority Interest', 'Stockholders Equity',
    'Gains Losses Not Affecting Retained Earnings',
    'Other Equity Adjustments', 'Retained Earnings', 'Capital Stock',
    'Common Stock', 'Total Liabilities Net Minority Interest',
    'Total Non Current Liabilities Net Minority Interest',
    'Other Non Current Liabilities', 'Tradeand Other Payables Non Current',
    'Non Current Deferred Liabilities', 'Non Current Deferred Revenue',
    'Non Current Deferred Taxes Liabilities',
    'Long Term Debt And Capital Lease Obligation',
    'Long Term Capital Lease Obligation', 'Long Term Debt',
    'Current Liabilities', 'Other Current Liabilities',
    'Current Deferred Liabilities', 'Current Deferred Revenue',
    'Current Debt And Capital Lease Obligation', 'Current Debt',
    'Pensionand Other Post Retirement Benefit Plans Current',
    'Payables And Accrued Expenses', 'Payables', 'Total Tax Payable',
    'Income Tax Payable', 'Accounts Payable', 'Total Assets',
    'Total Non Current Assets', 'Other Non Current Assets',
    'Investments And Advances', 'Long Term Equity Investment',
    'Goodwill And Other Intangible Assets', 'Other Intangible Assets',
    'Goodwill', 'Net PPE', 'Accumulated Depreciation', 'Gross PPE',
    'Leases', 'Other Properties', 'Machinery Furniture Equipment',
    'Buildings And Improvements', 'Land And Improvements', 'Properties',
    'Current Assets', 'Other Current Assets', 'Hedging Assets Current',
    'Inventory', 'Finished Goods', 'Work In Process', 'Raw Materials',
    'Receivables', 'Accounts Receivable',
    'Allowance For Doubtful Accounts Receivable',
    'Gross Accounts Receivable',
    'Cash Cash Equivalents And Short Term Investments',
    'Other Short Term Investments', 'Cash And Cash Equivalents',
    'Cash Equivalents', 'Cash Financial'.
    """

    symbol_list: List[str] = Field(
        description='A list of ticker symbols.',
    )
    user_query: str = Field(
        description='The extracted user query for given companies. '
        'If you cannot retrieve any specific query, '
        'use the default "Plot the key financial indicators over time."',
        default='Plot the key financial indicators over time.',
    )


@tool(args_schema=RetrievalSymbolSchema)
def get_financial_summary(symbol_list: List[str], user_query: str) -> Any:
    """Get the finanical summary for a given stock."""

    question = 'You are an expert in the stock market.\n' f'{user_query}\n'

    company = yfinance.Tickers(tickers=symbol_list)

    balance_sheet = company.balance_sheet.T

    df = SmartDataframe(
        balance_sheet,
        config={
            'llm': streamlit.session_state.fc.llm,
            'save_charts': True,
            'open_charts': False,
            'save_charts_path': TEMP_DIR,
            'enable_cache': False,
        },
    )
    response = df.chat(question)

    return response


@tool(args_schema=RetrievalSymbolSchema)
def analyse_stock_data_analysis(symbol_list: List[str], question: str) -> str:
    """Get the data analysis for a given stock"""

    response_dict = dict()

    for symbol in symbol_list:
        yahoo_connector = YahooFinanceConnector(symbol)
        df = SmartDataframe(yahoo_connector, config={'llm': streamlit.session_state.fc.llm})
        response_dict[symbol] = df.chat(question)

    # Convert dictionary to JSON string
    json_string = json.dumps(response_dict, indent=4)
    return json_string


# tool schema
class HistoricalPriceSchema(BaseModel):
    """Fetches historical stock prices for a given list of ticker symbols from 'start_date' to 'end_date'."""

    symbol_list: List[str] = Field('List of stock ticker symbol list.')
    quantity: str = Field('Quantity to analize', example=['Open', 'Close'])
    end_date: datetime.date = Field(
        'Typically today unless a specific end date is provided. End date MUST be greater than start date.'
    )
    start_date: datetime.date = Field(
        "Set explicitly, or calculated as 'end_date - date interval' "
        "(for example, if prompted 'over the past 6 months', "
        "date interval = 6 months so start_date would be 6 months earlier than today's date). "
        "Default to '1900-01-01' if vaguely asked for historical price. "
        'Start date must always be before the current date'
    )


@tool(args_schema=HistoricalPriceSchema)
def get_historical_price(
    symbol_list: List[str], quantity: str, start_date: datetime.date, end_date: datetime.date
) -> pandas.DataFrame:
    """
    Fetches historical stock prices for a given list of ticker symbols from 'start_date' to 'end_date'.
    - symbol (str): Stock ticker symbol.
    - quantity (str): Quantity to analize.
    - end_date (date): Typically today unless a specific end date is provided. End date MUST be greater than start date
    - start_date (date): Set explicitly, or calculated as 'end_date - date interval'
        (for example, if prompted 'over the past 6 months',
        date interval = 6 months so start_date would be 6 months earlier than today's date).
        Default to '1900-01-01' if vaguely asked for historical price. Start date must always be before the current date
    """

    # Initialise a pandas DataFrame with symbols as columns and dates as index
    data_close = pandas.DataFrame(columns=symbol_list)

    for symbol in symbol_list:
        data = yfinance.Ticker(symbol)
        data_history = data.history(start=start_date, end=end_date)
        data_close[symbol] = data_history[quantity]

    fig = plot_price_over_time(data_close)
    return data_close, fig


def plot_price_over_time(data_close: pandas.DataFrame) -> plotly.graph_objs.Figure:
    full_df = pandas.DataFrame(columns=['Date'])
    for column in data_close:
        full_df = full_df.merge(data_close[column], on='Date', how='outer')

    # Create a Plotly figure
    fig = go.Figure()

    # Dynamically add a trace for each stock symbol in the DataFrame
    for column in full_df.columns[1:]:  # Skip the first column since it's the date
        fig.add_trace(go.Scatter(x=full_df['Date'], y=full_df[column], mode='lines+markers', name=column))

    # Update the layout to add titles and format axis labels
    fig.update_layout(
        title='Stock Price Over Time: ' + ', '.join(full_df.columns.tolist()[1:]),
        xaxis_title='Date',
        yaxis_title='Stock Price (USD)',
        yaxis_tickprefix='$',
        yaxis_tickformat=',.2f',
        xaxis=dict(
            tickangle=-45,
            nticks=20,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True,  # Enable y-axis grid lines
            gridcolor='lightgrey',  # Set grid line color
        ),
        legend_title_text='Stock Symbol',
        plot_bgcolor='gray',  # Set plot background to white
        paper_bgcolor='gray',  # Set overall figure background to white
        legend=dict(
            bgcolor='gray',  # Optional: Set legend background to white
            bordercolor='black',
        ),
    )

    # Show the figure
    streamlit.plotly_chart(fig, use_container_width=True)

    # Return plot
    return fig