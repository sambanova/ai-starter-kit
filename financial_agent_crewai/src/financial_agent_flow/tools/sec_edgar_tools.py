import datetime
import logging
import os
from pathlib import Path
from typing import Dict

import requests
from crewai.tools import BaseTool
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings

from financial_agent_crewai.src.financial_agent_flow.tools.general_tools import FilenameOutput, get_html_text
from financial_agent_crewai.src.financial_agent_flow.tools.sorting_hat_tools import FilingsInput

logger = logging.getLogger(__name__)


class SecEdgarFilingRetriever(BaseTool):  # type: ignore
    """Tool for retrieving a financial filing from SEC Edgar and then answering the original user question."""

    name: str = 'SEC Edgar Filing Retriever'
    description: str = 'Retrieve a financial filing from SEC Edgar and then answer the original user question.'
    filing_metadata: FilingsInput
    cache_dir: Path

    def _run(self) -> FilenameOutput:
        # Check the filing type
        if self.filing_metadata.filing_type not in ['10-K', '10-Q']:
            self.filing_metadata.filing_type = '10-K'
        if self.filing_metadata.filing_type == '10-K':
            self.filing_metadata.year += 1

        # Retrieve the filing text from SEC Edgar
        try:
            downloader = Downloader(os.getenv('SEC_API_ORGANIZATION'), os.getenv('SEC_API_EMAIL'))
        except requests.exceptions.HTTPError:
            raise Exception('Please submit your SEC EDGAR details (organization and email) in the sidebar first.')

        # Extract today's year
        current_year = datetime.datetime.now().date().year

        # Extract the delta time, i.e. the number of years between the current year and the year of the filing
        delta = current_year - self.filing_metadata.year

        # Quarterly filing retrieval
        if self.filing_metadata.filing_type == '10-Q':
            if self.filing_metadata.filing_quarter is not None:
                if not isinstance(self.filing_metadata.filing_quarter, int):
                    raise TypeError('The quarter must be an integer.')
                if self.filing_metadata.filing_quarter not in [
                    1,
                    2,
                    3,
                    4,
                ]:
                    raise ValueError('The quarter must be between 1 and 4.')
                delta = (current_year - self.filing_metadata.year + 1) * 3
            else:
                raise ValueError('The quarter must be provided for 10-Q filing.')

        # Yearly filings
        elif self.filing_metadata.filing_type == '10-K':
            delta = current_year - self.filing_metadata.year + 1
            if delta == 0:
                delta += 1
        else:
            raise ValueError('The filing type must be either "10-K" or "10-Q".')

        response_dict: Dict[str, str] = dict()

        # Extract the metadata of the filings
        metadata = downloader.get_filing_metadatas(
            RequestedFilings(
                ticker_or_cik=self.filing_metadata.ticker_symbol,
                form_type=self.filing_metadata.filing_type,
                limit=delta,
            )
        )[0]

        # Extract HTML text
        html_text = downloader.download_filing(url=metadata.primary_doc_url)

        # Create the filename
        filename = self.cache_dir / (
            f"filing_id_{self.filing_metadata.filing_type.replace('-', '')}_{self.filing_metadata.filing_quarter}_"
            + f'{self.filing_metadata.ticker_symbol}_{self.filing_metadata.year}.csv'
        )

        # Parse filings
        get_html_text(html_text, str(filename))

        return FilenameOutput(filename=str(filename))
