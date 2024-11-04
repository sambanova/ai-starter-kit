import datetime
import operator
import os
from typing import Annotated, Any, Dict, List

from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage, get_buffer_string
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from financial_assistant.constants import *
from financial_assistant.streamlit.app_financial_filings import handle_financial_filings
from financial_assistant.streamlit.app_stock_database import handle_database_creation, handle_database_query
from financial_assistant.streamlit.app_yfinance_news import handle_yfinance_news

llm = ChatOpenAI(model='gpt-4o', temperature=0)


class Analyst(BaseModel):
    role: str = Field(
        description='Role of the analyst in the context of the topic.',
    )
    description: str = Field(
        description='Description of the analyst focus, concerns, and motives.',
    )

    @property
    def persona(self) -> str:
        return f'Role: {self.role}\nDescription: {self.description}\n'


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description='Comprehensive list of analysts with their roles and descriptions.',
    )


class GenerateAnalystsState(TypedDict):
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]


class AnalysisState(MessagesState):
    question: str
    max_num_turns: int
    analyst: Analyst
    analysis: str
    sections: List[str]


class SearchQuery(BaseModel):
    requested_companies: List[str] = Field(None, description='List of companies to research.')
    user_query: str = Field(None, description='Search query for retrieval.')


class SingularQuery(BaseModel):
    company_name: str = Field(None, description='Company to compare.')
    company_query: str = Field(None, description='Search query for the given company.')
    start_date: datetime.date = Field(DEFAULT_START_DATE, description='Start date for search.')
    end_date: datetime.date = Field(DEFAULT_END_DATE, description='End date for search.')
    filing_type: str = Field('10-K', description='Filing type to search.')
    filing_quarter: int = Field(0, description='Quarter to search. 0 for no quarters.')
    selected_year: int = Field(2023, description='Year to search.')


class ReportGraphState(TypedDict):
    human_analyst_feedback: str
    analysts: List[Analyst]
    user_query: str
    companies: List[str]
    sections: Annotated[List[str], operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


def generate_question(analysis_state: AnalysisState) -> Dict[str, Any]:
    """Node to generate a question"""

    # Get state
    analyst = analysis_state['analyst']
    messages = analysis_state['messages']

    # Generate question
    system_message = ''
    question = llm.invoke([SystemMessage(content=system_message)] + messages)  # type: ignore

    # Write messages to state
    return {'messages': [question]}


def generate_answer(analysis_state: AnalysisState) -> Dict[str, Any]:
    """Node to answer a question"""

    # Get state
    analyst = analysis_state['analyst']
    messages = analysis_state['messages']

    # Answer question
    system_message = ''
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)  # type: ignore

    # Append it to state
    return {'messages': [answer]}


def save_analysis(analysis_state: AnalysisState) -> Dict[str, Any]:
    """Save analysis."""

    # Get messages
    messages = analysis_state['messages']

    # Convert interview to a string
    analysis = get_buffer_string(messages)

    # Save to interviews key
    return {'analysis': analysis}


def stock_database_query(singular_query: SingularQuery) -> Dict[str, Any]:
    handle_database_creation(
        requested_companies=singular_query.company_name,
        start_date=singular_query.start_date,
        end_date=singular_query.end_date,
    )

    response_text_to_sql = handle_database_query(
        user_question=singular_query.company_query,
        query_method='text-to-SQL',
    )

    response_pandasai = handle_database_query(
        user_question=singular_query.company_query,
        query_method='PandasAI-SqliteConnector',
    )

    return {
        'text-to-SQL': response_text_to_sql,
        'PandasAI-SqliteConnector': response_pandasai,
    }


def yfinance_news_query(search_query: SearchQuery) -> Any:
    response = handle_yfinance_news(
        user_question=search_query.user_query,
    )

    return response


def financial_filings_query(singular_query: SingularQuery) -> Any:
    response = handle_financial_filings(
        user_question=singular_query.company_query,
        company_name=singular_query.company_name,
        filing_type=singular_query.filing_type,
        filing_quarter=singular_query.filing_quarter,
        selected_year=singular_query.selected_year,
    )

    return response


def write_section(state: AnalysisState) -> Dict[str, Any]:
    """Node to write a section"""

    section = ''

    # Append it to state
    return {'sections': section}


def route_messages() -> None:
    pass


# Add nodes and edges
analysis_builder = StateGraph(AnalysisState)
analysis_builder.add_node('ask_question', generate_question)
analysis_builder.add_node('stock_database_query', stock_database_query)
analysis_builder.add_node('yfinance_news_query', yfinance_news_query)
analysis_builder.add_node('financial_filings_query', financial_filings_query)
analysis_builder.add_node('answer_question', generate_answer)
analysis_builder.add_node('save_analysis', save_analysis)
analysis_builder.add_node('write_section', write_section)

# Flow
analysis_builder.add_edge(START, 'ask_question')
analysis_builder.add_edge('ask_question', 'stock_database_query')
analysis_builder.add_edge('ask_question', 'yfinance_news_query')
analysis_builder.add_edge('ask_question', 'financial_filings_query')
analysis_builder.add_edge('stock_database_query', 'answer_question')
analysis_builder.add_edge('yfinance_news_query', 'answer_question')
analysis_builder.add_edge('financial_filings_query', 'answer_question')
analysis_builder.add_conditional_edges('answer_question', route_messages, ['ask_question', 'save_analysis'])
analysis_builder.add_edge('save_analysis', 'write_section')
analysis_builder.add_edge('write_section', END)

# View
display(  # type: ignore
    Image(  # type: ignore
        analysis_builder.compile()
        .get_graph()
        .draw_mermaid_png(output_file_path=os.path.join(kit_dir, 'analysis_builder.png'))
    )
)


def supervisor(report_state: ReportGraphState) -> None:
    pass


def supervisor(report_state: ReportGraphState) -> Dict[str, Any]:
    """Decompose the analysis into a list of analysis by company."""
    user_query = report_state['user_query']
    system_message = f'The following companies match your query: {user_query}'

    companies = llm.invoke([SystemMessage(system_message)] + [HumanMessage(user_query)])
    return {'companies': companies}


def human_feedback(report_state: ReportGraphState) -> None:
    pass


def write_report(report_state: ReportGraphState) -> None:
    pass


def start_analysis(report_state: ReportGraphState) -> None:
    pass


def end_analysis(report_state: ReportGraphState) -> None:
    pass


# Add nodes and edges
builder = StateGraph(ReportGraphState)
builder.add_node('supervisor', supervisor)
builder.add_node('human_feedback', human_feedback)
builder.add_node('conduct_analysis', analysis_builder.compile())
builder.add_node('write_report', write_report)

# Logic
builder.add_edge(START, 'supervisor')
builder.add_edge('supervisor', 'human_feedback')
builder.add_conditional_edges('human_feedback', start_analysis, ['supervisor', 'conduct_analysis'])
builder.add_edge('conduct_analysis', 'write_report')
builder.add_conditional_edges('write_report', end_analysis, ['supervisor', END])

# Compile
graph = builder.compile(interrupt_before=['human_feedback'])

# View
display(Image(graph.get_graph().draw_mermaid_png(output_file_path=os.path.join(kit_dir, 'graph.png'))))  # type: ignore
