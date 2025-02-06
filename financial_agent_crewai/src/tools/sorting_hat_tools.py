import datetime
import logging
from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FilingType(str, Enum):
    TEN_K = '10-K'
    TEN_Q = '10-Q'


class FilingsInput(BaseModel):
    """
    Model for retrieving a single financial filing from SEC Edgar
    about a given company and a given year.
    """

    ticker_symbol: str = Field(..., min_length=1, description='The company ticker symbol.')
    company: str = Field(..., min_length=1, description='The company name.')
    filing_type: str = Field(
        default=FilingType.TEN_K,
        description='The SEC EDGAR filing type. Either "10-K" for yearly filings or "10-Q" for quarterly filings. '
        'If not specified, always choose "10-K".',
    )
    filing_quarter: Optional[int] = Field(
        None,
        ge=1,
        le=4,
        description='The quarter of the SEC EDGAR filing, if any (1, 2, 3, or 4). '
        'Defaults to None for annual filings (10-K).',
    )
    year: int = Field(
        default_factory=lambda: date.today().year,
        ge=2000,
        description=('Year of the filing.'),
    )
    query: str = Field(
        ...,
        description='A reformulation of the user query, tailored to correspond '
        'to the given company and (if mentioned) year.',
    )
    start_date: datetime.date = Field(
        description=(
            'The start date for retrieving historical data. '
            "Defaults to '2000-01-01' if the date is not specified or is ambiguous. "
            'Must be earlier than the end_date.'
        ),
    )

    end_date: datetime.date = Field(
        description=(
            'The end date for retrieving historical data. '
            "Defaults to today's date unless a specific date is provided. "
            'Must be later than the start_date.'
        ),
    )


class FilingsInputsList(BaseModel):
    """Model representing a list of filing metadata."""

    inputs_list: List[FilingsInput] = Field(..., description='The list of the filing metadata.')
