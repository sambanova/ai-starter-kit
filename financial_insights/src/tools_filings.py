import datetime
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

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

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)


logging.basicConfig(level=logging.INFO)

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

load_dotenv(os.path.join(repo_dir, '.env'))

TEMP_DIR = 'financial_insights/streamlit/cache/sources/'

MAX_CHUNK_SIZE = 128
RETRIEVE_HEADLINES = False


class YahooFinanceNewsInput(BaseModel):
    """Input for the YahooFinanceNews tool."""

    ticker_list: List[str] = Field(
        description='A list of ticker symbols to search.',
    )


# def split_text_into_chunks(text: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
#     # Split text into sentences based on punctuation
#     sentences = re.split(r'(?<=[.!?]) +', text)

#     chunks: List[str] = []
#     current_chunk: List[str] = []
#     current_length = 0

#     for sentence in sentences:
#         sentence_length = len(sentence.split())
#         # If adding the sentence exceeds the max_chunk_size, start a new chunk
#         if current_length + sentence_length > max_chunk_size:
#             chunks.append(' '.join(current_chunk))
#             current_chunk = [sentence]
#             current_length = sentence_length
#         else:
#             current_chunk.append(sentence)
#             current_length += sentence_length

#     # Add the last chunk if it has content
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))

#     return chunks


# def filter_texts(texts: Set[str]) -> List[str]:
#     """Filter out texts with fewer than 3 words."""
#     filtered_texts = set()
#     for text in texts:
#         if len(text.split()) >= 4 and len(text.split()) <= MAX_CHUNK_SIZE:
#             filtered_texts.add(text)
#         elif len(text.split()) > MAX_CHUNK_SIZE:
#             # Split the long text into smaller chunks
#             chunks = split_text_into_chunks(text)
#             for chunk in chunks:
#                 filtered_texts.add(chunk)
#         else:
#             pass
#     return list(filtered_texts)


# def filter_text(text: str) -> List[str]:
#     """Filter out texts with fewer than 3 words."""
#     filtered_texts: List[str] = list()
#     if len(text.split()) >= 4 and len(text.split()) <= MAX_CHUNK_SIZE:
#         filtered_texts.append(text)
#     elif len(text.split()) > MAX_CHUNK_SIZE:
#         # Split the long text into smaller chunks
#         chunks = split_text_into_chunks(text)
#         for chunk in chunks:
#             filtered_texts.append(chunk)
#     else:
#         pass
#     return filtered_texts


def clean_text(text: str) -> str:
    """Clean the text by removing extra spaces, newlines, and special characters."""
    return ' '.join(text.split())


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
                text = soup.get_text(separator=' ')

                splitter = CharacterTextSplitter(
                    chunk_size=512,
                    chunk_overlap=64,
                    separator=r'[.!?]',
                    is_separator_regex=True,
                )
                chunks = splitter.split_text(text)

                # Save chunks to csv
                df = pandas.DataFrame(chunks, columns=['text'])
                filename = (
                    f'filing_id_{filing_type.replace('-', '')}_{filing_quarter}_'
                    f'{ticker_symbol}_{report_date.date().year}'
                )
                df.to_csv(TEMP_DIR + filename + '.csv', index=False)
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
        df = pandas.read_csv(TEMP_DIR + f'{filename}' + '.csv')
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
