import datetime
import os
from typing import Optional, Tuple

import pandas
import streamlit
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings

from financial_insights.src.tools import get_logger
from financial_insights.src.utilities_retrieval import get_qa_response
from financial_insights.streamlit.constants import *

load_dotenv(os.path.join(repo_dir, '.env'))

logger = get_logger()


class SecEdgarFilingsInput(BaseModel):
    """Model to retrieve the text of a financial filing from SEC Edgar and then answer the original user question."""

    user_question: str = Field(..., description='The user question.')
    user_request: str = Field(..., description='The retrieval request.')
    ticker_symbol: str = Field(..., description='The company ticker symbol.')
    filing_type: str = Field(..., description='The type of filing (either "10-K" or "10-Q").')
    filing_quarter: Optional[int] = Field(
        None, description='The quarter of the filing (1, 2, 3, or 4). Defaults to 0 for no quarter.'
    )
    year: int = Field(..., description='The year of the filing.')


@tool(args_schema=SecEdgarFilingsInput)
def retrieve_filings(
    user_question: str,
    user_request: str,
    ticker_symbol: str,
    filing_type: str,
    filing_quarter: int,
    year: int,
) -> str:
    """
    Retrieve the text of a financial filing from SEC Edgar and then answer the original user question.

    Args:
        user_question: The user question.
        user_request: The retrieval request.
        ticker_symbol: The company ticker symbol.
        filing_type: The type of filing (either `10-K` or `10-Q`).
        filing_quarter: The quarter of the filing (among 1, 2, 3, 4). Defaults to 0 for no quarter.
        year: The year of the filing.

    Returns:
        A tuple of the following elements:
            - The answer to the user question, preceded by the information about the retrieval:
                `filing_type`, `filing_quarter`, `ticker_symbol`, and `report_date`.

    Raises:
        TypeErrror: If `user_request`, `user_question`, `ticker_symbol`, or `filing_type` are not strings.
        TypeError: If `filing_quarter` is not an integer for a `filing_type` of `10-Q`.
        ValueError: If `filing_quarter` is not in [1, 2, 3, 4] for a `filing_type` of `10-Q`.
        ValueError: If `filing_type` is not one of `10-K` or `10-Q`
        ValueError: If `filing_quarter` is not provided for `10-Q` filing.
        Exception: If a matching document for the given `filing_type`, `ticker_symbol`, and `year` is not available.
        Exception: If a matching document for the given `filing_type`, `ticker_symbol`, `filing_quarter`, and `year`
            is not available.
    """
    # Checks the inputs
    assert isinstance(user_question, str), TypeError(f'User question must be a string. Got {type(user_question)}.')
    assert isinstance(user_request, str), TypeError(f'User question must be a string. Got {type(user_request)}.')
    assert isinstance(ticker_symbol, str), TypeError(f'Filing type must be a string. Got {type(ticker_symbol)}.')
    assert isinstance(filing_type, str), TypeError(f'Filing type must be a string. Got {type(filing_type)}.')

    # Retrieve the filing text from SEC Edgar
    downloader = Downloader(streamlit.session_state.SEC_API_ORGANIZATION, streamlit.session_state.SEC_API_EMAIL)

    # Extract today's year
    current_year = datetime.datetime.now().date().year

    # Extract the delta time, i.e. the number of years between the current year and the year of the filing
    delta = current_year - year

    # Quarterly filing retrieval
    if filing_type == '10-Q':
        if filing_quarter is not None:
            assert isinstance(filing_quarter, int), TypeError('The quarter must be an integer.')
            assert filing_quarter in [
                1,
                2,
                3,
                4,
            ], 'The quarter must be between 1 and 4.'
            delta = (current_year - year + 1) * 3
        else:
            raise ValueError('The quarter must be provided for 10-Q filing.')

    # Yearly filings
    elif filing_type == '10-K':
        delta = current_year - year + 1
    else:
        raise ValueError('The filing type must be either "10-K" or "10-Q".')

    # Parse filings
    filename, report_date = parse_filings(downloader, ticker_symbol, filing_type, filing_quarter, year, delta)

    # Load the dataframe from the text file
    try:
        df = pandas.read_csv(streamlit.session_state.source_dir + f'{filename}' + '.csv')
    except FileNotFoundError:
        logger.error('No scraped data found.')

    # Convert DataFrame rows into Document objects
    documents = []
    for _, row in df.iterrows():
        document = Document(page_content=row['text'])
        documents.append(document)

    # Return the QA response
    response = get_qa_response(user_question, documents)

    # Assert that response is indexable
    assert isinstance(response, dict), 'QA response is not a dictionary.'

    # Return the filing type, filing quarter, the ticker symbol, and the year of the filing
    assert isinstance(report_date, datetime.datetime), 'The report date is not a of type `datetime.date`.'
    query_dict = {
        'filing_type': filing_type,
        'filing_quarter': filing_quarter,
        'ticker_symbol': ticker_symbol,
        'report_date': report_date.date().year,
    }

    answer = (
        'Filing type: {filing_type},\nFiling quarter: {filing_quarter},\n'
        'Ticker symbol: {ticker_symbol},\nYear of filing: {report_date}'.format(**query_dict)
    )
    answer += '\n\n' + response['answer']
    return answer


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
                    f"filing_id_{filing_type.replace('-', '')}_{filing_quarter}_"
                    + f'{ticker_symbol}_{report_date.date().year}'
                )
                df.to_csv(streamlit.session_state.source_dir + filename + '.csv', index=False)
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
