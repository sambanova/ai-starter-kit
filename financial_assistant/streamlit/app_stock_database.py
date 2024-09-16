import datetime
from typing import Any, Dict, List, Optional

import streamlit
from streamlit.elements.widgets.time_widgets import DateWidgetReturn

from financial_assistant.streamlit.constants import *
from financial_assistant.streamlit.utilities_app import save_output_callback
from financial_assistant.streamlit.utilities_methods import attach_tools, handle_userinput


def get_stock_database() -> None:
    streamlit.markdown('<h2> Stock Database </h2>', unsafe_allow_html=True)
    streamlit.markdown(
        '<a href="https://pypi.org/project/yfinance/" target="_blank" '
        'style="color:cornflowerblue;text-decoration:underline;"><h3>via Yahoo! Finance API</h3></a>',
        unsafe_allow_html=True,
    )
    streamlit.markdown('<h3> Create database </h3>', unsafe_allow_html=True)

    requested_companies = streamlit.text_input(
        label=f'Enter the names of the companies you want to retrieve. :sparkles: :violet[{DEFAULT_COMPANY_NAME}]',
        key='create-database',
        placeholder='E.g. ' + DEFAULT_COMPANY_NAME,
    )
    start_date = streamlit.date_input(
        'Start Date', value=datetime.datetime.now() - datetime.timedelta(days=365), key='start-date'
    )
    end_date = streamlit.date_input('End Date', value=datetime.datetime.now(), key='end-date')

    if streamlit.button('Create database'):
        if len(requested_companies) == 0:
            streamlit.error('Please enter at least one company.')
        else:
            with streamlit.expander('**Execution scratchpad**', expanded=True):
                response_string = handle_database_creation(requested_companies, start_date, end_date)

    streamlit.markdown('<br><br>', unsafe_allow_html=True)
    streamlit.markdown('<h3> Query database </h3>', unsafe_allow_html=True)
    streamlit.write(r':red[NB: Before querying the database for one company, you need to create it!]')
    help_query_method = (
        'text-to-SQL will generate SQL queries,'
        '\nwhereas PandasAI-SqliteConnector will use pandasai to query the database.'
    )
    query_method = streamlit.selectbox(
        'Select method (for best results, try both):',
        ['text-to-SQL', 'PandasAI-SqliteConnector'],
        index=0,
        help=help_query_method,
    )
    user_request = streamlit.text_input(
        label=f'Enter your query. :sparkles:  :violet[{DEFAULT_STOCK_QUERY}]',
        key='query-database',
        placeholder='E.g. ' + DEFAULT_STOCK_QUERY,
    )
    if streamlit.button(label='Query database'):
        if len(user_request) == 0:
            streamlit.error('Please enter your query.')
        else:
            with streamlit.expander('**Execution scratchpad**', expanded=True):
                response_dict = handle_database_query(user_request, query_method)

                # Save the query and answer to the history text file
                save_output_callback(response_dict, streamlit.session_state.history_path, user_request)

                # Save the query and answer to the database query text file
                if streamlit.button(
                    'Save Query',
                    on_click=save_output_callback,
                    args=(response_dict, streamlit.session_state.db_query_path, user_request),
                ):
                    pass


def handle_database_creation(
    requested_companies: str,
    start_date: DateWidgetReturn,
    end_date: DateWidgetReturn,
) -> Dict[str, List[str]]:
    """
    Handle the database creation for given companies and dates.

    Args:
        requested_companies: List of company names for which to create the database.
        start_date: Start date for the database creation.
        end_date: End date for the database creation.

    Returns:
        A dictionary with company symbols as keys and a list of SQL table names as values.

    Raises:
        TypeError: If the LLM response does not conform to the return type.
    """
    assert isinstance(
        requested_companies, str
    ), f'`requested_companies` should be a string. Got type {type(requested_companies)}.'
    assert len(requested_companies) > 0, 'No companies selected.'

    # Declare the permitted tools for function calling
    streamlit.session_state.tools = ['create_stock_database']

    # Attach the tools for the LLM to use
    attach_tools(streamlit.session_state.tools)

    # Compose the user request
    user_request = f"""
        Please create a SQL database for the following list of companies:
        {requested_companies}

        The requested dates for data retrieval are from {start_date} to {end_date}.
    """

    # Call the LLM on the user request with the attached tools
    response = handle_userinput(requested_companies, user_request)

    # Check the final answer of the LLM
    assert isinstance(response, dict), f'`response` should be of type `dict`. Got {type(response)}.'
    assert all(isinstance(key, str) for key in response.keys()), f'`response.keys()` should be of type `str`.'
    assert all(
        isinstance(value, str) for value in list(response.values())[0]
    ), f'`response.values()` should be of type `str`.'

    return response


def handle_database_query(
    user_question: Optional[str] = None,
    query_method: Optional[str] = 'text-to-SQL',
) -> Any:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question (str): The user's question or input.

    Raises:
        AssertionError: If `query_method` is not one of the permitted methods.
    """
    if user_question is None:
        return None

    assert query_method in ['text-to-SQL', 'PandasAI-SqliteConnector'], f'Invalid query method {query_method}'

    # Declare the permitted tools for function calling
    streamlit.session_state.tools = ['query_stock_database']

    # Attach the tools for the LLM to use
    attach_tools(streamlit.session_state.tools)

    # Compose the user request
    user_request = f"""
        Please answer the following query for the given list of companies:
        {user_question}

        First, extract the company (or companies) from the user query.
        Then, convert the query from natural language to SQL using the method:
        "{query_method}" to generate the response.
    """

    # Call the LLM on the user request with the attached tools
    response = handle_userinput(user_question, user_request)

    return response
