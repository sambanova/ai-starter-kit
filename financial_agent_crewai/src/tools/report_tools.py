from pydantic import BaseModel, Field

from financial_agent_crewai.src.config import MAX_SECTION_WORDS


class ReportSection(BaseModel):
    """
    Model representing a fully detailed report section
    with title, summary, and content.
    """

    title: str = Field(..., description="The section title. Must start with '##'.")
    summary: str = Field(..., description='A concise summary of the section.')
    content: str = Field(
        ...,
        description=f'The fully detailed section content in Markdown format, limited to {MAX_SECTION_WORDS} words. ',
    )
