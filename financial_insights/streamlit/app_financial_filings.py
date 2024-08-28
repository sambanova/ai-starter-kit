from typing import Optional

import streamlit

from financial_insights.streamlit.constants import *
from financial_insights.streamlit.utilities_app import save_output_callback
from financial_insights.streamlit.utilities_methods import attach_tools, handle_userinput


def include_financial_filings() -> None:
    """Include the app for financial filings analysis."""

    streamlit.markdown('<h2> Financial Filings Analysis </h2>', unsafe_allow_html=True)
    streamlit.markdown(
        '<a href="https://www.sec.gov/edgar/search/" target="_blank" '
        'style="color:cornflowerblue;text-decoration:underline;"><h3>via SEC EDGAR</h3></a>',
        unsafe_allow_html=True,
    )

    # User request
    user_request = streamlit.text_input(r'$\textsf{\normalsize Enter your query:}$', key='financial-filings')
    company_name = streamlit.text_input(r'$\textsf{\normalsize Company name (optional if in the query already)}$')
    # Define the range of years
    start_year = 2020
    end_year = 2024
    years = list(range(start_year, end_year + 1))
    # Set the default year (e.g., 2023)
    default_year = 2023
    default_index = years.index(default_year)
    # Create the selectbox with the default year
    selected_year = streamlit.selectbox('Select a year:', years, index=default_index)

    # Filing type, between `10-K` (yearly) and `10-Q` (quarterly)
    filing_type = streamlit.selectbox('Select Filing Type:', ['10-K', '10-Q'], index=0)
    if filing_type == '10-Q':
        # Filing quarter among 1, 2, 3, and 4
        filing_quarter = streamlit.selectbox('Select Quarter:', [1, 2, 3, 4])
    else:
        filing_quarter = 0
    if streamlit.button('Analyze Filing'):
        with streamlit.expander('**Execution scratchpad**', expanded=True):
            # Call the function to analyze the financial filing
            answer = handle_financial_filings(
                user_request,
                company_name,
                filing_type,
                filing_quarter,
                selected_year,
            )

            # Save the query and answer to the history text file
            save_output_callback(answer, HISTORY_PATH, user_request)

            # Save the query and answer to the filing text file
            if streamlit.button(
                'Save Answer',
                on_click=save_output_callback,
                args=(answer, FILINGS_PATH, user_request),
            ):
                pass


def handle_financial_filings(
    user_question: str,
    company_name: str,
    filing_type: Optional[str] = '10-K',
    filing_quarter: Optional[int] = 0,
    selected_year: Optional[int] = 2023,
) -> str:
    """
    Handle the user request for financial filing data.

    Args:
        user_question: The user's question about financial filings.
        company_name: The company name to search for.
        filing_type: The type of financial filing to search for, between `10-K` and `10-Q`.
            Default is `10-K`.
        selected_year: The year of the financial filing to search for.

    Returns:
        A tuple of the following elements:
            - The final LLM answer to the user's question.
            - A dictionary of metadata about the retrieval, with the following keys:
                `filing_type`, `filing_quarter`, `ticker_symbol`, and `report_date`.

    Raises:
        Exception: If `response` (the final return from `handle_userinput`) does not conform to the return type.
    """
    # Declare the permitted tools for function calling
    streamlit.session_state.tools = ['retrieve_symbol_list', 'retrieve_filings']

    # Attach the tools for the LLM to use
    attach_tools(streamlit.session_state.tools)

    # Compose the user request
    user_request = (
        'Please answer the following query for a given list of companies. ' + user_question + '\n'
        'First retrieve the list of ticker symbols from the list of company names within the query.\n'
        'Then provide an answer after retrieving the provided context using RAG.\n'
        f'In order to provide context for the question, please retrieve the given SEC EDGAR '
        f'financial filing type: {filing_type} '
        f'and filing quarter: {filing_quarter} '
        f'for the company {company_name} for the year {selected_year}.\n'
        f'The original user question is the following: {user_question}'
    )

    # Call the LLM on the user request with the attached tools
    response = handle_userinput(user_question, user_request)

    # Check the final answer of the LLM
    assert isinstance(response, str), f'Invalid response: {response}'

    return response
