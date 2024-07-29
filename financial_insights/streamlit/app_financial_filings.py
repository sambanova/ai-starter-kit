import os
import sys
from typing import Optional, Tuple

import streamlit

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from financial_insights.streamlit.utilities_app import \
    save_string_answer_callback
from financial_insights.streamlit.utilities_methods import (handle_userinput,
                                                            set_fc_llm)


def get_financial_filings() -> None:
    streamlit.markdown(
        "<h2> Financial Filings Analysis </h2>", unsafe_allow_html=True
    )
    streamlit.markdown(
        '<a href="https://www.sec.gov/edgar/search/" target="_blank" '
        'style="color:cornflowerblue;text-decoration:underline;"><h3>via SEC EDGAR</h3></a>',
        unsafe_allow_html=True,
    )
    user_request = streamlit.text_input("Enter your query:", key="financial-filings")
    company_name = streamlit.text_input(
        "Company name (optional if in the query already)"
    )
    # Define the range of years
    start_year = 2020
    end_year = 2024
    years = list(range(start_year, end_year + 1))
    # Set the default year (e.g., 2023)
    default_year = 2023
    default_index = years.index(default_year)
    # Create the selectbox with the default year
    selected_year = streamlit.selectbox(
        "Select a year:", years, index=default_index
    )

    filing_type = streamlit.selectbox(
        "Select Filing Type:", ["10-K", "10-Q"], index=0
    )
    if filing_type == "10-Q":
        filing_quarter = streamlit.selectbox("Select Quarter:", [1, 2, 3, 4])
    else:
        filing_quarter = None
    if streamlit.button("Analyze Filing"):
        with streamlit.expander("**Execution scratchpad**", expanded=True):
            answer, filename = handle_financial_filings(
                user_request,
                company_name,
                filing_type,
                filing_quarter,
                selected_year,
            )

            content = user_request + "\n\n" + answer + "\n\n\n"
            if streamlit.button(
                "Save Answer",
                on_click=save_string_answer_callback,
                args=(content, filename + ".txt"),
            ):
                pass


def handle_financial_filings(
    user_question: str,
    company_name: str,
    filing_type: Optional[str] = "10-K",
    filing_quarter: Optional[int] = 0,
    selected_year: Optional[int] = 2023,
) -> Tuple[str, str]:
    streamlit.session_state.tools = ["retrieve_symbol_list", "retrieve_filings"]
    set_fc_llm(streamlit.session_state.tools)

    user_request = (
        "You are an expert in the stock market.\n"
        + f"In order to provide context for the question, please retrieve the given SEC EDGAR "
        + f"financial filing type: {filing_type} "
        + f"and filing quarter: {filing_quarter} "
        + f"for the company {company_name} for the year {selected_year}.\n"
        + f'The original user question if the following: {user_question}'
    )

    response = handle_userinput(user_question, user_request)

    assert (
        isinstance(response, tuple)
        and len(response) == 2
        and all(isinstance(item, str) for item in response)
    ), f"Invalid response: {response}"

    return response  # type: ignore