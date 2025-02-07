import datetime
import logging
from enum import Enum
from typing import List

from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Today's date
TODAY = datetime.date.today()

# Current year
CURRENT_YEAR = TODAY.year

# Current month
CURRENT_MONTH = TODAY.month

# Current quarter of the current year
CURRENT_QUARTER = (CURRENT_MONTH - 1) // 3 + 1

# 3 months ago
RECENT_DATE = TODAY - relativedelta(months=3)


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
        description='The relevant SEC EDGAR filing type. '
        'Either "10-K" for yearly/annual filings or "10-Q" for quarterly filings. '
        'If not specified, always choose "10-K".',
    )
    filing_quarter: int = Field(
        CURRENT_QUARTER,
        ge=1,
        le=4,
        description='The desired quarter of the year, if specified (1, 2, 3, or 4). '
        f'Defaults to the current quarter, i.e. {CURRENT_QUARTER}, if not specified.',
    )
    year: int = Field(
        CURRENT_YEAR,
        ge=2000,
        description=(
            f'Year relevant to the user query. Defaults to the current year, i.e. {CURRENT_YEAR}, if not specified.'
        ),
    )
    query: str = Field(
        ...,
        description='A reformulation of the user query, tailored to correspond '
        'to the given company and (if mentioned) year.',
    )
    start_date: datetime.date = Field(
        RECENT_DATE,
        description=(
            'The start date for retrieving any historical data relevant to the user query.. '
            f'Defaults to {RECENT_DATE}, if not specified. '
            'Must be earlier than the end_date.'
        ),
    )

    end_date: datetime.date = Field(
        TODAY,
        description=(
            'The end date for retrieving any historical data relevant to the user query. '
            f"Defaults to today's date, {TODAY}, if not specified. "
            'Must be later than the start_date.'
        ),
    )


class FilingsInputsList(BaseModel):
    """Model representing a list of filing metadata."""

    inputs_list: List[FilingsInput] = Field(..., description='The list of the filing metadata.')
