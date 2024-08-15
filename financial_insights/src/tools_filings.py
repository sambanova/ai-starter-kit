import datetime
import logging
import os
from typing import Any, Dict, Optional, Tuple

import pandas
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings

from financial_insights.src.utilities_retrieval import get_qa_response
from financial_insights.streamlit.constants import *

logging.basicConfig(level=logging.INFO)

load_dotenv(os.path.join(repo_dir, '.env'))

MAX_CHUNK_SIZE = 256
RETRIEVE_HEADLINES = False


class SecEdgarFilingsInput(BaseModel):
    """Retrieve the text of a financial filing from SEC Edgar and then answer the original user question."""

    user_question: str = Field('The user question.')
    user_request: str = Field('The retrieval request.')
    ticker_symbol: str = Field('The company ticker symbol.')
    filing_type: str = Field('The type of filing (either "10-K" or "10-Q").')
    filing_quarter: Optional[int] = Field('The quarter of the filing (among 1, 2, 3, 4). Defaults to 0')
    date: int = Field('The year of the filing.')


@tool(args_schema=SecEdgarFilingsInput)
def retrieve_filings(
    user_question: str,
    user_request: str,
    ticker_symbol: str,
    filing_type: str,
    filing_quarter: int,
    date: int,
) -> Tuple[Any, Dict[str, str]]:
    """Retrieve the text of a financial filing from SEC Edgar and then answer the full user request."""

    dl = Downloader(os.environ.get('SEC_API_ORGANIZATION'), os.environ.get('SEC_API_EMAIL'))

    # Extract today's year
    current_year = datetime.datetime.now().date().year

    limit = current_year - date

    if filing_type == '10-Q':
        if filing_quarter is not None:
            assert isinstance(filing_quarter, int), 'The quarter must be an integer.'
            assert filing_quarter in [
                1,
                2,
                3,
                4,
            ], 'The quarter must be between 1 and 4.'
            limit = (current_year - date + 1) * 3
        else:
            raise ValueError('The quarter must be provided for 10-Q filing.')
    elif filing_type == '10-K':
        limit = current_year - date + 1
    else:
        raise ValueError('The filing type must be either "10-K" or "10-Q".')

    metadatas = dl.get_filing_metadatas(
        RequestedFilings(ticker_or_cik=ticker_symbol, form_type=filing_type, limit=limit)
    )

    filename = None
    for metadata in metadatas:
        report_date = metadata.report_date

        # Convert string to datetime
        report_date = datetime.datetime.strptime(report_date, '%Y-%m-%d')

        if report_date.year == date:
            if filing_quarter == 0 or (filing_quarter is not None and (report_date.month - filing_quarter * 3) <= 1):
                # Logging
                logging.info(f'Found filing: {metadata}')
                html_text = dl.download_filing(url=metadata.primary_doc_url)

                # Convert html to text
                soup = BeautifulSoup(html_text, 'html.parser')
                # Extract text from the parsed HTML
                text = soup.get_text(separator=' ', strip=True)

                splitter = CharacterTextSplitter(
                    chunk_size=MAX_CHUNK_SIZE,
                    chunk_overlap=64,
                    separator=r'[.!?]',
                    is_separator_regex=True,
                )
                chunks = splitter.split_text(text)

                # Save chunks to csv
                df = pandas.DataFrame(chunks, columns=['text'])
                filename = (
                    f'filing_id_{filing_type.replace('-', '')}_{filing_quarter}_'
                    + f'{ticker_symbol}_{report_date.date().year}'
                )
                df.to_csv(CACHE_DIR + filename + '.csv', index=False)
                break

    if filename is None or filename == '':
        if filing_type == '10-K':
            raise Exception(f'Filing document {filing_type} for {ticker_symbol} for the year {date} is not available')
        else:
            raise Exception(
                f'Filing document {filing_type} for {ticker_symbol} for the quarter {filing_quarter} '
                'of the year {date} is not available'
            )

    # Load the dataframe from the text file
    try:
        df = pandas.read_csv(CACHE_DIR + f'{filename}' + '.csv')
    except FileNotFoundError:
        logging.error('No scraped data found.')

    # Convert DataFrame rows into Document objects
    documents = []
    for _, row in df.iterrows():
        document = Document(page_content=row['text'])
        documents.append(document)

    response = get_qa_response(documents, user_question)

    query_dict = {
        'filing_type': filing_type,
        'filing_quarter': filing_quarter,
        'ticker_symbol': ticker_symbol,
        'report_date': report_date.date().year,
    }
    return response['answer'], query_dict
