import os
import sys
from typing import Any, Optional

import streamlit
from streamlit.elements.widgets.time_widgets import DateWidgetReturn

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from financial_insights.streamlit.utilities_app import save_output_callback
from financial_insights.streamlit.utilities_methods import handle_userinput, set_fc_llm


def get_stock_database() -> None:
    streamlit.markdown('<h2> Stock Database </h2>', unsafe_allow_html=True)
    streamlit.markdown(
        '<a href="https://pypi.org/project/yfinance/" target="_blank" '
        'style="color:cornflowerblue;text-decoration:underline;"><h3>via Yahoo! Finance API</h3></a>',
        unsafe_allow_html=True,
    )
    streamlit.markdown('<h3> Create database </h3>', unsafe_allow_html=True)

    output = streamlit.empty()

    requested_companies = streamlit.text_input(
        'Enter the named of the companies you want to retrieve.',
        key='create-database',
    )
    start_date = streamlit.date_input('Start Date')
    end_date = streamlit.date_input('End Date')

    if streamlit.button('Create database'):
        with streamlit.expander('**Execution scratchpad**', expanded=True):
            response_string = handle_database_creation(requested_companies, start_date, end_date)

    streamlit.markdown('<br><br>', unsafe_allow_html=True)
    streamlit.markdown('<h3> Query database </h3>', unsafe_allow_html=True)
    query_method = streamlit.selectbox('Select method:', ['text-to-SQL', 'PandasAI-SqliteConnector'], index=0)
    user_request = streamlit.text_input(
        'Enter your query.',
        key='query-database',
    )
    if streamlit.button(label='Query database'):
        with streamlit.expander('**Execution scratchpad**', expanded=True):
            response_dict = handle_database_query(user_request, query_method)
            content = user_request + '\n\n' + response_dict['message']
            save_path = 'db_query.txt'
            if streamlit.button(
                'Save Query',
                on_click=save_output_callback,
                args=(content, save_path),
            ):
                pass


def handle_database_creation(
    requested_companies: Optional[str],
    start_date: DateWidgetReturn,
    end_date: DateWidgetReturn,
) -> Any:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question (str): The user's question or input.
    """
    if requested_companies is None:
        return None

    streamlit.session_state.tools = ['retrieve_symbol_list', 'create_stock_database']
    set_fc_llm(streamlit.session_state.tools)

    user_request = (
        'Please create a SQL database for the following companies '
        '(expressed via their ticker symbols): '
        'and within the following dates.\n' + requested_companies
    )
    user_request += f'\nThe requested dates are from {start_date} to {end_date}'

    return handle_userinput(requested_companies, user_request)


def handle_database_query(
    user_question: Optional[str] = None,
    query_method: Optional[str] = 'text-to-SQL',
) -> Any:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question (str): The user's question or input.
    """
    if user_question is None:
        return None

    assert query_method in ['text-to-SQL', 'PandasAI-SqliteConnector'], f'Invalid query method {query_method}'

    streamlit.session_state.tools = ['retrieve_symbol_list', 'query_stock_database']
    set_fc_llm(streamlit.session_state.tools)

    user_request = (
        'Please query the SQL database for the following companies '
        '(expressed via their ticker symbols): '
        + user_question
        + 'Use the method: "'
        + query_method
        + '" to generate the response.'
    )

    return handle_userinput(user_question, user_request)
