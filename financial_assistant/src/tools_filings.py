import datetime
import os
from typing import Dict, List, Optional, Tuple

import pandas
import requests
import streamlit
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings

from financial_assistant.constants import *
from financial_assistant.src.retrieval import get_qa_response
from financial_assistant.src.tools import coerce_str_to_list
from financial_assistant.src.tools_stocks import retrieve_symbol_list
from financial_assistant.src.utilities import get_logger

load_dotenv(os.path.join(repo_dir, '.env'))

logger = get_logger()


class SecEdgarFilingsInput(BaseModel):
    """Tool for retrieving a financial filing from SEC Edgar and then answering the original user question."""

    user_question: str = Field(..., description='The original user question.')
    company_list: List[str] | str = Field(..., description='The required companies.')
    filing_type: str = Field(..., description='The type of filing (either "10-K" or "10-Q").')
    filing_quarter: Optional[int] = Field(
        None, description='The quarter of the filing (1, 2, 3, or 4). Defaults to None for no quarter.'
    )
    year: int = Field(..., description='The year of the filing.')


@tool(args_schema=SecEdgarFilingsInput)
def retrieve_filings(
    user_question: str,
    company_list: List[str] | str,
    filing_type: str,
    filing_quarter: int,
    year: int,
) -> Dict[str, str]:
    """
    Tool for retrieving a financial filing from SEC Edgar and then answering the original user question.

    Args:
        user_question: The original user question.
        company_list: The required companies.
        filing_type: The type of filing (either `10-K` or `10-Q`).
        filing_quarter: The quarter of the filing (among 1, 2, 3, 4). Defaults to None for no quarter.
        year: The year of the filing.

    Returns:
        A dictionary of the following elements:
            - The keys are the company ticker symbols.
            - The values are the answer to the user question by company,
                preceded by the information about the retrieval:
                `filing_type`, `filing_quarter`, `ticker_symbol`, and `report_date`.

    Raises:
        TypeErrror: If `user_question`, `ticker_symbol`, or `filing_type` are not strings.
        TypeError: If `filing_quarter` is not an integer for a `filing_type` of `10-Q`.
        ValueError: If `filing_quarter` is not in [1, 2, 3, 4] for a `filing_type` of `10-Q`.
        ValueError: If `filing_type` is not one of `10-K` or `10-Q`
        ValueError: If `filing_quarter` is not provided for `10-Q` filing.
        Exception: If a matching document for the given `filing_type`, `ticker_symbol`, and `year` is not available.
        Exception: If a matching document for the given `filing_type`, `ticker_symbol`, `filing_quarter`, and `year`
            is not available.
        Exception: If the LLM response is not a dictionary.
        TypeError: is the extracted report date is not of type `datetime.datetime`.
    """
    # Checks the inputs
    if not isinstance(user_question, str):
        raise TypeError(f'User question must be a string. Got {type(user_question)}.')
    if not isinstance(filing_type, str):
        raise TypeError(f'Filing type must be a string. Got {type(filing_type)}.')

    if not isinstance(company_list, (list, str)):
        raise TypeError(f'`company_list` must be of type list or string. Got {(type(company_list))}.')

    # If `symbol_list` is a string, coerce it to a list of strings
    company_list = coerce_str_to_list(company_list)

    if not all([isinstance(name, str) for name in company_list]):
        raise TypeError('`company_names_list` must be a list of strings.')

    # Retrieve the list of ticker symbols
    symbol_list = retrieve_symbol_list(company_list)

    # Retrieve the filing text from SEC Edgar
    try:
        downloader = Downloader(os.getenv('SEC_API_ORGANIZATION'), os.getenv('SEC_API_EMAIL'))
    except requests.exceptions.HTTPError:
        raise Exception('Please submit your SEC EDGAR details (organization and email) in the sidebar first.')

    # Extract today's year
    current_year = datetime.datetime.now().date().year

    # Extract the delta time, i.e. the number of years between the current year and the year of the filing
    delta = current_year - year

    # Quarterly filing retrieval
    if filing_type == '10-Q':
        if filing_quarter is not None:
            if not isinstance(filing_quarter, int):
                raise TypeError('The quarter must be an integer.')
            if filing_quarter not in [
                1,
                2,
                3,
                4,
            ]:
                raise ValueError('The quarter must be between 1 and 4.')
            delta = (current_year - year + 1) * 3
        else:
            raise ValueError('The quarter must be provided for 10-Q filing.')

    # Yearly filings
    elif filing_type == '10-K':
        delta = current_year - year + 1
    else:
        raise ValueError('The filing type must be either "10-K" or "10-Q".')

    response_dict: Dict[str, str] = dict()

    for symbol in symbol_list:
        # Parse filings
        filename, report_date = parse_filings(downloader, symbol, filing_type, filing_quarter, year, delta)

        # Load the dataframe from the text file
        try:
            df = pandas.read_csv(os.path.join(streamlit.session_state.sources_dir, f'{filename}.csv'))
        except FileNotFoundError:
            logger.error('No scraped data found.')

        # Convert DataFrame rows into Document objects
        documents = list()
        for _, row in df.iterrows():
            document = Document(page_content=row['text'])
            documents.append(document)

        # Return the QA response
        response = get_qa_response(user_question, documents)  # TODO pass sambanova_api_key

        # Ensure that response is indexable
        if not isinstance(response, dict):
            raise Exception('QA response is not a dictionary.')

        # Return the filing type, filing quarter, the ticker symbol, and the year of the filing
        if not isinstance(report_date, datetime.datetime):
            raise TypeError(f'The report date is not a of type `datetime.date`. Got type {type(report_date)}')
        query_dict = {
            'filing_type': filing_type,
            'filing_quarter': filing_quarter,
            'symbol': symbol,
            'report_date': report_date.date().year,
        }

        answer = (
            'Filing type: {filing_type},\nFiling quarter: {filing_quarter},\n'
            'Ticker symbol: {symbol},\nYear of filing: {report_date}'.format(**query_dict)
        )
        answer += '\n\n' + response['answer']
        response_dict[symbol] = answer

    return response_dict


def parse_filings(
    downloader: Downloader, ticker_symbol: str, filing_type: str, filing_quarter: int, year: int, delta: int = 10
) -> Tuple[str, str]:
    """
    Search the filing, parse it, and save it.

    The relevant filing refers to a company by ticker symbol, filing type, for a specific quarter and year.

    Args:
        downloader: Downloader object to download the filings from SEC website.
        ticker_symbol: Ticker symbol of the company to be parsed.
        filing_type: Filing type of the company to be parsed.
        filing_quarter: Filing quarter of the company to be parsed.
        year: Year of the company to be parsed.
        delta: Maximum number of years to be searched.

    Returns:
        A tuple of the followimg pair:
            1. The filename of the relevant parsed filing.
            2. The report date of the relevant parsed filing.
    """

    # Extract the metadata of the filings
    metadatas = downloader.get_filing_metadatas(
        RequestedFilings(ticker_or_cik=ticker_symbol, form_type=filing_type, limit=delta)
    )

    # Extract the filing text
    filename = None
    for metadata in metadatas:
        # Extract the filing date
        report_date = metadata.report_date

        # Convert string to datetime
        report_date = datetime.datetime.strptime(report_date, '%Y-%m-%d')

        # Check the matching year in the time delta
        if report_date.year == year:
            # Check the matching quarter
            if filing_quarter == 0 or (filing_quarter is not None and (report_date.month - filing_quarter * 3) <= 1):
                # Logging
                logger.info(f'Found filing: {metadata}')
                # Download the matching filing
                html_text = downloader.download_filing(url=metadata.primary_doc_url)

                # Convert html to text
                soup = BeautifulSoup(html_text, 'html.parser')
                # Extract text from the parsed HTML
                text = soup.get_text(separator=' ', strip=True)

                # Instantiate the text splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=MAX_CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                    separators=[
                        r'\n\n',  # Split on double newlines (paragraphs)
                        r'(?<=[.!?])\s+(?=[A-Z])',  # Split on sentence boundaries
                        r'\n',  # Split on single newlines
                        r'\s+',  # Split on whitespace
                        r'',  # Split on characters as a last resort
                    ],
                    is_separator_regex=True,
                )

                # Split the text into chunks
                chunks = splitter.split_text(text)

                # Save chunks to csv
                df = pandas.DataFrame(chunks, columns=['text'])
                filename = (
                    f'filing_id_{filing_type.replace("-", "")}_{filing_quarter}_'
                    + f'{ticker_symbol}_{report_date.date().year}'
                )
                df.to_csv(os.path.join(streamlit.session_state.sources_dir, f'{filename}.csv'), index=False)
                break

    # If neither the year nor the quarter match, raise an error
    if filename is None or filename == '':
        if filing_type == '10-K':
            raise Exception(f'Filing document {filing_type} for {ticker_symbol} for the year {year} is not available')
        else:
            raise Exception(
                f'Filing document {filing_type} for {ticker_symbol} for the quarter {filing_quarter} '
                'of the year {date} is not available'
            )

    return filename, report_date
