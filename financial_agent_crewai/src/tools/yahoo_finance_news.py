import logging

import requests
import yfinance
from crewai.tools import BaseTool
from requests.exceptions import HTTPError, ReadTimeout
from urllib3.exceptions import ConnectionError

from financial_agent_crewai.src.tools.general_tools import FilenameOutputList, get_html_text
from financial_agent_crewai.src.utils.constants import CACHE_DIR

logger = logging.getLogger(__name__)


class YahooFinanceNewsTool(BaseTool):  # type: ignore
    """Tool that searches financial news on Yahoo Finance."""

    name: str = 'yahoo_finance_news'
    description: str = (
        'Useful for when you need to find financial news about a public company. '
        'Input should be a company ticker. '
        'For example, AAPL for Apple, MSFT for Microsoft.'
    )
    ticker_symbol: str

    def _run(
        self,
        query: str,
    ) -> FilenameOutputList:
        """Use the Yahoo Finance News tool."""

        company = yfinance.Ticker(self.ticker_symbol)
        try:
            if company.isin is None:
                raise Exception(f'Company ticker {self.ticker_symbol} not found.')
        except (HTTPError, ReadTimeout, ConnectionError):
            raise Exception(f'Company ticker {self.ticker_symbol} not found.')

        url_list = list()
        try:
            url_list = [n['link'] for n in company.news if n['type'] == 'STORY']
        except (HTTPError, ReadTimeout, ConnectionError):
            if url_list is None:
                return f'No news found for company that searched with {query} ticker.'
        if url_list is None:
            return f'No news found for company that searched with {query} ticker.'

        # Remove duplicate URLs from the list of links
        url_list = list(set(url_list))
        filename_list = list()

        # Webscraping by url
        for count, url in enumerate(url_list):
            if count >= 5:
                break

            # Send a GET request to the URL
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Create the filename
                filename = CACHE_DIR / f'yahoo_news_{self.ticker_symbol}_{count}'
                filename_list.append(str(filename))

                # Parse the HTML content
                get_html_text(response.content, str(filename))

            else:
                logger.warning(f'Failed to retrieve the page. Status code: {response.status_code}')

        return FilenameOutputList(filenames=filename_list)
