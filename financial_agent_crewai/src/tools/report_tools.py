from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    """Class to write report sections."""

    title: str = Field(..., description='Section title')
    content: str = Field(..., description='Section content')
    summary: str = Field(..., description='Section summary')
