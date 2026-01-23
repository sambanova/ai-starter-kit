import re
from typing import Any, List, Optional, Set, Tuple

import pandas
import requests
import streamlit
import yfinance
from bs4 import BeautifulSoup
from langchain_classic.schema import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from financial_assistant.constants import *
from financial_assistant.src.retrieval import get_qa_response
from financial_assistant.src.tools import coerce_str_to_list
from financial_assistant.src.tools_stocks import retrieve_symbol_list
from financial_assistant.src.utilities import get_logger

RETRIEVE_HEADLINES = False

logger = get_logger()


class YahooFinanceNewsInput(BaseModel):
    """Tool for searching financial news on Yahoo Finance through web scraping."""

    company_list: Optional[List[str] | str] = Field(
        None, description='A list of companies to search, if applicable.', examples=['Google', 'Microsoft']
    )
    user_query: str = Field(..., description='The original user query.')


@tool(args_schema=YahooFinanceNewsInput)
def scrape_yahoo_finance_news(company_list: List[str] | str, user_query: str) -> Tuple[str, List[str]]:
    """
    Tool for searching financial news on Yahoo Finance through web scraping.

    Useful for when you need to find financial news about a public company.

    Args:
        company_list: List of companies about which to search for financial news.
        user_query: The search query to be used in the search bar on Yahoo Finance.

    Returns:
        A tuple containing the following pair:
            1. The answer to the user query.
            2. A list of links to articles that have been used for retrieval to answer the user query.

    Raises:
        TypeError: If `company_list` is not a string or a list of strings or `user_query` is not a string.
    """
    if company_list is not None:
        # Check inputs
        if not isinstance(company_list, (list, str)):
            raise TypeError(f'Input must be of type `list` or `str`. Got {type(company_list)}.')

        # If `company_list` is a string, coerce it to a list of strings
        company_list = coerce_str_to_list(company_list)

        if not all(isinstance(company, str) for company in company_list):
            raise TypeError('All elements in `company_list` must be of type str.')
        if not isinstance(user_query, str):
            raise TypeError(f'Input must be of type str. Got {type(user_query)}.')

    # Retrieve the list of ticker symbols
    try:
        symbol_list = retrieve_symbol_list(company_list)
    except:
        symbol_list = None

    # Get the list of relevant urls for the list of companies
    link_urls = get_url_list(symbol_list)

    # Scrape the news articles
    retrieve_text_yahoo_finance_news(link_urls)
    logger.info('News from Yahoo Finance successfully extracted and saved.')

    return get_qa_response_from_news(streamlit.session_state.web_scraping_path, user_query)


def retrieve_text_yahoo_finance_news(link_urls: List[str]) -> None:
    """
    Scrapes news articles from Yahoo Finance for a given list of ticker symbols.

    Args:
        link_urls: A list of urls that point to news articles.
    """
    # Initialize lists to store the extracted data
    headlines = set()
    h2_headings = set()
    h3_headings = set()
    paragraphs = set()
    data = list()
    # Loop through the links to visit each page and extract headlines
    for link_url in link_urls:
        # Send an HTTP GET request to the link
        link_response = requests.get(link_url)
        if link_response.status_code == 200:
            # Parse the content of the link's page
            soup = BeautifulSoup(link_response.content, 'html.parser')

            if RETRIEVE_HEADLINES:
                # Find elements with 'data-test-locator="headline"'
                headline_elements = soup.find_all(attrs={'data-test-locator': 'headline'})
                # Extract and append the headline text to the headlines list
                for headline_element in headline_elements:
                    text = clean_text(headline_element.text.strip())
                    headlines.add(text)
                    filtered_texts = filter_text(text)
                    for filtered_text in filtered_texts:
                        data.append({'type': 'h1', 'url': link_url, 'text': filtered_text})

                # Find and extract all <h2> headings
                for h2_tag in soup.find_all('h2'):
                    text = clean_text(h2_tag.get_text(strip=True))
                    h2_headings.add(text)
                    filtered_texts = filter_text(text)
                    for filtered_text in filtered_texts:
                        data.append({'type': 'h2', 'url': link_url, 'text': filtered_text})

                # Find and extract all <h3> headings
                for h3_tag in soup.find_all('h3'):
                    text = clean_text(h3_tag.get_text(strip=True))
                    h3_headings.add(text)
                    filtered_texts = filter_text(text)
                    for filtered_text in filtered_texts:
                        data.append({'type': 'h3', 'url': link_url, 'text': filtered_text})

            # Find and extract all <p> paragraphs
            for p_tag in soup.find_all('p'):
                text = clean_text(p_tag.get_text(strip=True))
                paragraphs.add(text)
                filtered_texts = filter_text(text)
                for filtered_text in filtered_texts:
                    data.append({'type': 'p', 'url': link_url, 'text': filtered_text})

            logger.info('News articles have been successfully scraped')
        else:
            logger.warning(f'Failed to retrieve the page. Status code: {link_response.status_code}')

    # Filter all texts
    headlines_list = filter_texts_set(headlines)
    h2_headings_list = filter_texts_set(h2_headings)
    h3_headings_list = filter_texts_set(h3_headings)
    paragraphs_list = filter_texts_set(paragraphs)

    # Create a DataFrame from the lists
    df = pandas.DataFrame({'News': headlines_list + h2_headings_list + h3_headings_list + paragraphs_list})
    df.drop_duplicates().reset_index(drop=True, inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv(streamlit.session_state.yfinance_news_csv_path, index=False)

    # Save the data to a text file in the specified order
    with open(streamlit.session_state.yfinance_news_txt_path, 'w') as file:
        if RETRIEVE_HEADLINES:
            file.write('=== Headlines ===\n')
            for item in headlines_list:
                file.write(f'{item}\n')

            file.write('\n=== H2 Tags ===\n')
            for item in h2_headings_list:
                file.write(f'{item}\n')

            file.write('\n=== H3 Tags ===\n')
            for item in h3_headings_list:
                file.write(f'{item}\n')

        # Always include the paragraphs
        file.write('\n=== Paragraphs ===\n')
        for item in paragraphs_list:
            file.write(f'{item}\n')

    # Convert the list to a DataFrame
    df_text_url = pandas.DataFrame(data)
    # Save the DataFrame to a CSV file
    df_text_url.to_csv(streamlit.session_state.web_scraping_path, index=False)


def get_url_list(symbol_list: Optional[List[str]] = None) -> List[Any]:
    """
    Get the most relevant urls from Yahoo Finance News for a given list of company ticker symbols.

    Args:
        symbol_list: A list of strings that represent company ticker symbols.

    Returns:
        A list of relevant urls.
    """
    # Define the URL of the Yahoo Finance news page
    main_url = 'https://finance.yahoo.com/news'

    general_urls = []
    singular_urls = []

    # For each symbol determine the list of URLs to scrape
    if symbol_list is not None and len(symbol_list) > 0:
        for symbol in symbol_list:
            try:
                general_urls.append(f'https://finance.yahoo.com/quote/{symbol}/')

                # Get the YFinance ticker object
                company = yfinance.Ticker(symbol)

                # Get the news articles from Yahoo Finance
                yfinance_url_list = [
                    item['content']['canonicalUrl']['url']
                    for item in company.get_news(count=MAX_URLS)
                    if item['content']['contentType'] == 'STORY'
                ]

                # Extend the list of singular URLs
                singular_urls.extend(yfinance_url_list)
            except:
                pass
    else:
        general_urls.append(main_url)

    link_urls = list()

    # Webscraping by url
    for url in general_urls + singular_urls:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all links on the page
            if url in general_urls:
                # Find all the links mentioned in the webpage
                links = soup.find_all('a')
                link_urls.extend([link['href'] for link in links])
            else:
                link_urls.append(url)
        else:
            logger.warning(f'Failed to retrieve the page. Status code: {response.status_code}')

    # Remove duplicate URLs from the list of links
    link_urls = list(set(link_urls))

    return link_urls[0:MAX_URLS]


def get_qa_response_from_news(web_scraping_path: str, user_query: str) -> Tuple[str, List[str]]:
    """
    Answer questions after retrieving the relevant Yahoo Finance News scraping data.

    Args:
        web_scraping_path: Path to the Yahoo Finance News scraping data file.
        user_query: The user query to be answered after retrieving the Yahoo Finance News scraping data.
        documents: List of Yahoo Finance News scraping data.

    Returns:
        A tuple containing the following pair:
        1. The answer to the user query.
        2. A list of links to articles that have been used for retrieval to answer the user query.

    Raises:
        Exception: If the LLM response is not a dictionary.
    """
    # Load the dataframe from the text file
    try:
        df = pandas.read_csv(web_scraping_path)
    except FileNotFoundError:
        logger.error('No scraped data found.')

    # Convert DataFrame rows into Document objects
    documents = []
    for _, row in df.iterrows():
        document = Document(
            page_content=row['text'],
            metadata={'url': row['url'], 'type': row['type']},
        )
        documents.append(document)

    # Get the QA response
    response = get_qa_response(user_query, documents)  # TODO pass sambanova_api_key

    # Ensure that response is indexable
    if not isinstance(response, dict):
        raise Exception('QA response is not a dictionary.')

    # Extract the answer from  the QA response
    answer = response['answer'] if isinstance(response['answer'], str) else ''

    # Extract the urls from the QA response
    url_list: List[str] = list()
    for doc in response['context']:
        if isinstance(doc, Document):
            if doc.metadata.get('url') is not None:
                url_list.append(doc.metadata['url'])

    return answer, url_list


def filter_texts_set(texts: Set[str]) -> List[str]:
    """
    Filter a set of texts and combine them into a list of chunks.

    Args:
        texts: A set of texts to filter.

    Returns:
        A list of chunks.

    Raises:
        TypeError: If the input `texts` is not a set of strings.
    """
    # Check inputs
    if not isinstance(texts, set):
        raise TypeError(f'Input must be of type set. Got {type(texts)}.')
    if not all(isinstance(text, str) for text in texts):
        raise TypeError(f'Input must be of type str.')

    filtered_texts = set()
    for text in texts:
        chunks = filter_text(text)
        for chunk in chunks:
            filtered_texts.add(chunk)
    return list(filtered_texts)


def filter_text(text: str) -> List[str]:
    """
    Split a text into chunks of MAX_CHUNK_SIZE.

    Args:
        The text to split into chunks.

    Returns:
        A list of chunks.

    Raises:
        TypeError: If the input `text` is not a string.
    """
    # Check inputs
    if not isinstance(text, str):
        raise TypeError(f'Input must be of type str. Got {type(text)}.')

    filtered_texts: List[str] = list()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

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

    # Split the long text into smaller chunks
    chunks = splitter.split_text(text)

    return chunks


def clean_text(text: str) -> str:
    """Clean the text by removing extra spaces, newlines, and special characters."""

    # Check inputs
    if not isinstance(text, str):
        raise TypeError(f'Input must be of type str. Got {type(text)}.')

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return ' '.join(text.split())
