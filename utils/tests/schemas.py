from typing import List, Annotated

from pydantic import BaseModel, Field


class DataExtraction(BaseModel):
    section: str
    products: List[str]


class ContactForm(BaseModel):
    name: str
    phone_number: float


class Step(BaseModel):
    explanation: str
    output: str


class Solution(BaseModel):
    final_answer: str
    steps: List[Step]


class YFinanceSource(BaseModel):
    """
    Represents a relevant data source from Yahoo Finance (YFinance)
    and the columns related to the user's query.
    """
    name: str = Field(
        ...,
        description="The name or identifier of the data source.",
        title="Name"
    )
    columns: List[str] = Field(
        ...,
        description="The list of column names that might be relevant for the query, in their original spelling/casing.",
        title="Columns"
    )


class YFinanceSourceList(BaseModel):
    """
    A collection of YFinanceSource objects, each specifying a data source
    and its relevant columns, pertinent to the user's query.
    """
    sources: Annotated[
        List[YFinanceSource],
        Field(min_length=1, max_length=3,
        description="A list of YFinanceSource objects representing potentially relevant data sources and columns.",
        title="Sources")
    ]