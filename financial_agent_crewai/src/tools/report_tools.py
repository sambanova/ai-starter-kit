from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    """Class to write a fully detailed report section."""

    title: str = Field(..., description='The section title, preceded by ##.')
    summary: str = Field(..., description='The section summary')
    content: str = Field(
        ..., description='The fully detailed and expanded section, formatted in markdown, no more than 1000 words.'
    )
