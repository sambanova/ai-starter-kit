from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    """
    Model representing a fully detailed report section
    with title, summary, and content.
    """

    title: str = Field(..., description="The section title. Must start with '##'.")
    summary: str = Field(..., description='A concise summary of the section.')
    content: str = Field(..., description='A fully detailed section in Markdown format, limited to 1000 words.')
