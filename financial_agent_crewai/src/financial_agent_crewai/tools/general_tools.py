import logging
from pathlib import Path
from typing import List

import pandas
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Main text processing, RAG, and web scraping constants
MIN_CHUNK_SIZE = 4
MAX_CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256

logger = logging.getLogger(__name__)


class FilenameOutput(BaseModel):
    """Model representing a saved file path."""

    filename: str = Field(..., description='The exact file path of the saved file.')


class FilenameOutputList(BaseModel):
    """Model representing a list of saved file paths."""

    file_output_list: List[FilenameOutput] = Field(
        ..., description='The list of the exact file paths of the saved files.'
    )


def get_html_text(html_text: bytes, filename: str) -> None:
    """
    Get text from HTML file.

    Args:
       html_text: The HTML file.
       filename: The filename of the output path to save the text file.
    """
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

    # Save chunks as csv
    df.to_csv(filename, mode='a', index=False, header=False)

    return


class SubQueriesList(BaseModel):
    """Model representing a minimal list of subsequent sub-queries, derived from the original user query."""

    queries_list: List[str] = Field(
        ...,
        min_length=1,
        description=(
            'A list of sub-queries derived from the original user query. '
            'Each sub-query relates to a single company. '
            'If the user explicitly mentions comparing two or more different periods, '
            'then each sub-query focuses on one company-period pair.'
        ),
    )
    is_comparison: bool = Field(
        ...,
        description=(
            'Indicates whether the original user query involves an explicit comparison '
            '(e.g., between multiple companies or multiple years).'
        ),
    )


def convert_csv_source_to_txt_report_filename(source_filename: str) -> str:
    """Convert source to report filename."""

    if source_filename.endswith('.csv'):
        return str(Path(Path(source_filename).parent) / ('report_' + Path(source_filename).name.strip('.csv') + '.txt'))
    elif source_filename.endswith('.txt'):
        return str(Path(Path(source_filename).parent) / ('report_' + Path(source_filename).name))
    else:
        raise ValueError('Source filename must end with either ".csv" or ".txt".')
