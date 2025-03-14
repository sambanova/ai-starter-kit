import logging
from pathlib import Path
from typing import Optional

import requests
import yfinance
from bs4 import BeautifulSoup
from crewai.tools import BaseTool
from requests.exceptions import HTTPError, ReadTimeout
from urllib3.exceptions import ConnectionError

from financial_agent_crewai.src.financial_agent_flow.config import *
from financial_agent_crewai.src.financial_agent_flow.tools.general_tools import FilenameOutput, get_html_text

logger = logging.getLogger(__name__)


class YahooFinanceNewsTool(BaseTool):  # type: ignore
    """Tool that searches financial news on Yahoo Finance."""

    name: str = 'Yahoo Finance News Tool'
    description: str = (
        'Useful for when you need to find financial news about a public company. '
        'Input should be a company ticker symbol. '
    )
    cache_dir: Path

    def _run(
        self,
        ticker_symbol: Optional[str] = None,
    ) -> FilenameOutput:
        """Use the Yahoo Finance News tool."""
        url_list = list()
        if ticker_symbol is not None:
            max_news_per_ticker = MAX_NEWS_PER_TICKER
            company = yfinance.Ticker(ticker_symbol)
            try:
                if company.isin is None:
                    raise Exception(f'Company ticker {ticker_symbol} not found.')
            except (HTTPError, ReadTimeout, ConnectionError):
                raise Exception(f'Company ticker {ticker_symbol} not found.')
            try:
                url_list = [
                    item['content']['canonicalUrl']['url']
                    for item in company.news
                    if item['content']['contentType'] == 'STORY'
                ]
            except (HTTPError, ReadTimeout, ConnectionError):
                if url_list is None:
                    return f'No news found for company that searched with {ticker_symbol} ticker.'
            if url_list is None:
                return f'No news found for company that searched with {ticker_symbol} ticker.'
        else:
            main_url = 'https://finance.yahoo.com/news'
            response = requests.get(main_url)

            # Check if the request was successful
            if response.status_code == 200:
                max_news_per_ticker = MAX_NEWS_PER_TICKER

                # Parse the HTML content
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find all the links mentioned in the webpage
                links = soup.find_all('a')
                url_list.extend([link['href'] for link in links])

        # Remove duplicate URLs from the list of links
        url_list = list(set(url_list))
        filename_list = list()

        # Create the filename
        filename = str(self.cache_dir / f'yfinance_news_{ticker_symbol}.csv')

        # Webscraping by url
        report_count = 0
        for url in url_list:
            if report_count >= MAX_NEWS_PER_TICKER:
                break

            # Send a GET request to the URL
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                filename_list.append(str(filename))

                # Parse the HTML content
                get_html_text(response.content, str(filename))

                report_count += 1

            else:
                logger.warning(f'Failed to retrieve the page. Status code: {response.status_code}')

        return FilenameOutput(filename=filename)
