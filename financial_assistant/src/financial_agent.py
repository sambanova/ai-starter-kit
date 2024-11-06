import datetime
import functools
import operator
import os
from typing import Annotated, Any, Dict, List, Sequence
import streamlit

streamlit.session_state.SAMBANOVA_API_KEY = os.getenv('SAMBANOVA_API_KEY')
from IPython.display import Image, display
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, get_buffer_string
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from financial_assistant.constants import *
from financial_assistant.src.tools import get_conversational_response
from financial_assistant.src.tools_database import create_stock_database, query_stock_database
from financial_assistant.src.tools_filings import retrieve_filings
from financial_assistant.src.tools_pdf_generation import pdf_rag
from financial_assistant.src.tools_stocks import (
    get_historical_price,
    get_stock_info,
)
from financial_assistant.src.tools_yahoo_news import scrape_yahoo_finance_news
from financial_assistant.streamlit.llm_model import sambanova_chat

# tool mapping of available tools
TOOLS = {
    'get_stock_info': get_stock_info,
    'get_historical_price': get_historical_price,
    'scrape_yahoo_finance_news': scrape_yahoo_finance_news,
    'get_conversational_response': get_conversational_response,
    'retrieve_filings': retrieve_filings,
    'create_stock_database': create_stock_database,
    'query_stock_database': query_stock_database,
    'pdf_rag': pdf_rag,
}

# llm = ChatOpenAI(model='gpt-4o', temperature=0)
llm = sambanova_chat


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


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
    user_query: str
    companies: List[str]
    sections: Annotated[List[str], operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


def generate_question(state: AgentState) -> Dict[str, Any]:
    """Node to generate a question"""

    system_message = 'You are a financial analyst.'
    question = llm.invoke([SystemMessage(content=system_message)] + state['messages'])
    return {'messages': [question]}


def generate_answer(state: AgentState) -> Dict[str, Any]:
    """Node to answer a question"""

    system_message = 'Compose an answer from the received messages.'
    answer = llm.invoke([SystemMessage(content=system_message)] + state.messages)
    return {'messages': [answer]}


def save_analysis(state: AgentState) -> Dict[str, Any]:
    """Save analysis."""

    analysis = get_buffer_string(state.messages)
    return {'analysis': analysis}


def agent_node(state: StateGraph, agent: CompiledGraph, name: str) -> Dict[str, List[Any]]:
    """Helper function to create a node for a given agent."""

    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={'type', 'name'}), name=name)
    return {
        'messages': [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        'sender': name,
    }


database_tools = [
    'create_stock_database',
    'query_stock_database',
]

database_agent = create_react_agent(
    llm,
    tools=[TOOLS[tool_name] for tool_name in database_tools],
)
stock_database_node = functools.partial(agent_node, agent=database_agent, name='Database')

yfinance_news_tools = [
    'scrape_yahoo_finance_news',
]

yahoo_news_agent = create_react_agent(
    llm,
    tools=[TOOLS[tool_name] for tool_name in yfinance_news_tools],
)
yfinance_news_node = functools.partial(agent_node, agent=yahoo_news_agent, name='YahooFinance')

financial_filings_tools = [
    'retrieve_filings',
]

financial_filings_agent = create_react_agent(
    llm,
    tools=[TOOLS[tool_name] for tool_name in financial_filings_tools],
)
financial_filings_node = functools.partial(agent_node, agent=financial_filings_agent, name='FinancialFilings')

tool_node = ToolNode(
    tools=[TOOLS[tool_name] for tool_name in database_tools + yfinance_news_tools + financial_filings_tools]
)


def write_section(state: AgentState) -> Dict[str, Any]:
    """Node to write a section"""

    section = ''

    # Append it to state
    return {'sections': section}


def router(state: AgentState) -> str:
    """Router node."""

    last_message = state.messages[-1]
    match last_message:
        case BaseMessage(tool_calls=[*_]):
            return 'call_tool'
        case BaseMessage(content=content) if 'FINAL ANSWER' in content:
            return END
        case _:
            return 'continue'


analysis_builder = StateGraph(AgentState)
nodes = {
    'ask_question': generate_question,
    'call_tool': tool_node,
    'stock_database_node': stock_database_node,
    'yfinance_news_node': yfinance_news_node,
    'financial_filings_node': financial_filings_node,
    'answer_question': generate_answer,
    'save_analysis': save_analysis,
    'write_section': write_section,
}

tool_nodes = ['stock_database_node', 'yfinance_news_node', 'financial_filings_node']

for name, func in nodes.items():
    analysis_builder.add_node(name, func)

analysis_builder.add_edge(START, 'ask_question')
for node in tool_nodes:
    analysis_builder.add_edge('ask_question', node)
    analysis_builder.add_conditional_edges(node, router, {'continue': 'answer_question', 'call_tool': 'call_tool'})

analysis_builder.add_conditional_edges(
    'call_tool',
    lambda x: x['sender'],
    {node: node for node in tool_nodes},
)

analysis_builder.add_conditional_edges('answer_question', router, ['ask_question', 'save_analysis'])
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

analyis_graph = analysis_builder.compile()
events = analyis_graph.stream(
    {
        'messages': [HumanMessage(content='What is the research and development trend of Meta?')],
    },
    # Maximum number of steps to take in the graph
    {'recursion_limit': 150},
)
for s in events:
    print(s)
    print('----')
