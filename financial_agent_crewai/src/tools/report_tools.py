from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    """Model representing a fully detailed report section."""

    content: str = Field(..., description='The fully detailed and expanded section content in Markdown format.')


class ReportSummary(BaseModel):
    """Model representing the summary of a report section."""

    title: str = Field(..., description='The short and descriptive title of the section.')
    summary: str = Field(..., description='A concise summary of the section, limited to three sentences.')
