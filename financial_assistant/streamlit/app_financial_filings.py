from typing import Dict, Optional

import streamlit

from financial_assistant.streamlit.constants import *
from financial_assistant.streamlit.utilities_app import save_output_callback
from financial_assistant.streamlit.utilities_methods import attach_tools, handle_userinput


def include_financial_filings() -> None:
    """Include the app for the financial filings analysis."""

    streamlit.markdown('<h2> Financial Filings Analysis </h2>', unsafe_allow_html=True)
    streamlit.markdown(
        '<a href="https://www.sec.gov/edgar/search/" target="_blank" '
        'style="color:cornflowerblue;text-decoration:underline;"><h3>via SEC EDGAR</h3></a>',
        unsafe_allow_html=True,
    )

    # User request
    user_request = streamlit.text_input(
        label=f'Enter your query. :sparkles: :violet[{DEFAULT_RAG_QUERY}]',
        key='financial-filings',
        placeholder='E.g. ' + DEFAULT_RAG_QUERY,
    )

    # Company name
    company_name = streamlit.text_input(
        label=f'Company name (optional if in the query already). :sparkles: :violet[{DEFAULT_COMPANY_NAME}]',
        key='company_name',
        placeholder='E.g. ' + DEFAULT_COMPANY_NAME,
    )

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
        if len(user_request) == 0:
            streamlit.error('Please enter your query.')
        else:
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
                save_output_callback(answer, streamlit.session_state.history_path, user_request)

                # Save the query and answer to the filing text file
                if streamlit.button(
                    'Save Answer',
                    on_click=save_output_callback,
                    args=(answer, streamlit.session_state.filings_path, user_request),
                ):
                    pass


def handle_financial_filings(
    user_question: str,
    company_name: str,
    filing_type: Optional[str] = '10-K',
    filing_quarter: Optional[int] = 0,
    selected_year: Optional[int] = 2023,
) -> Dict[str, str]:
    """
    Handle the user request for the financial filing data.

    Args:
        user_question: The user's question about financial filings.
        company_name: The company name to search for.
        filing_type: The type of financial filing to search for, between `10-K` and `10-Q`.
            Default is `10-K`.
        filing_quarter: The quarter of the financial filing to search for,
            between 1 and 4. Default is `0`.
        selected_year: The year of the financial filing to search for.

    Returns:
        A tuple of the following elements:
            - The final LLM answer to the user's question,
                preceded by the metadata about the retrieval:
                `filing_type`, `filing_quarter`, `ticker_symbol`, and `report_date`.

    Raises:
        TypeError: If the LLM response does not conform to the return type.
    """
    # Declare the permitted tools for function calling
    streamlit.session_state.tools = ['retrieve_filings']

    # Attach the tools for the LLM to use
    attach_tools(streamlit.session_state.tools)

    # Compose the user request
    user_request = f"""
        Please answer the following query based on a list of companies using RAG (Retrieval-Augmented Generation).

        To provide context, retrieve the specified SEC EDGAR financial filing:
        - Filing type: {filing_type}
        - Filing quarter: {filing_quarter}
        - Company: {company_name}
        - Year: {selected_year}

        The user question is: {user_question}
    """

    # Call the LLM on the user request with the attached tools
    response = handle_userinput(user_question, user_request)

    # Check the final answer of the LLM
    assert isinstance(response, dict), TypeError(f'Invalid LLM response: {response}.')

    return response
