from pydantic import BaseModel, Field


class ReportSectionSummary(BaseModel):
    """Class to write a fully detailed report section."""

    title: str = Field(..., description='The section title.')
    summary: str = Field(..., description='The section summary')
    content: str = Field(..., description='The fully detailed section, formatted in markdown.')
