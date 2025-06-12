from typing import List, Annotated, Literal

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


class Person(BaseModel):
    first_name: str = Field(..., title="First Name")
    last_name: str = Field(..., title="Last Name")
    age: int = Field(..., title="Age")

    class Config:
        extra = "forbid"


class ChickenEgg(BaseModel):
    result: Literal["Chicken", "Egg"]

    class Config:
        extra = "allow"  # Since "strict": false


class CountriesExtraction(BaseModel):
    Countries: List[str]

    class Config:
        extra = "forbid"


class UserIntent(BaseModel):
    userIntent: str
    userIntentCategory: Literal[
        "Unspecified",
        "Asking_A_Question",
        "Searching",
        "Performing_An_Action",
        "Providing_Feedback",
        "Answering_Question",
        "Making_Statement",
        "Summarize",
        "Generate_Text",
        "Write_An_Email",
        "Create_A_Hubshare_Email",
        "Create_A_Digital_Sales_Room",
        "Create_A_Story_Or_Page",
        "Create_A_Brainshark_Presentation",
        "Create_A_Quiz",
        "Create_A_Roleplay_Scenario",
        "List_Content",
        "List_Popular_Content",
        "List_Recommended_Content",
        "List_Favourite_Content",
        "List_Shares",
        "Create_A_Hubshare_Link",
        "Create_A_Podcast",
        "Create_Scoring_Criteria",
        "Generate_A_Diagram"
    ]
    searchQuery: str
    searchKeywords: str
    language: str

    class Config:
        extra = "forbid"


class AnswerSchema(BaseModel):
    answer: str
    chatTitle: str
    subject: str
    message: str
    suggestions: List[str]
    ableToAnswer: bool

    class Config:
        extra = "forbid"


class PodcastTurn(BaseModel):
    speakerName: str
    gender: Literal["Male", "Female"]
    content: str

    class Config:
        extra = "forbid"


class Podcast(BaseModel):
    topic: str
    turns: List[PodcastTurn]

    class Config:
        extra = "forbid"


class PodcastGeneration(BaseModel):
    answer: str
    podcast: Podcast
    suggestions: List[str]
    ableToAnswer: bool

    class Config:
        extra = "forbid"


class ExtractName(BaseModel):
    name: str

    class Config:
        extra = "allow"