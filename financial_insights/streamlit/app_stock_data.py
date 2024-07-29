import datetime
import os
import sys
from typing import Any, Optional, Tuple

import pandas
import plotly
import streamlit

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from financial_insights.streamlit.utilities_app import (
    save_dataframe_figure_callback, save_dict_answer_callback,
    save_string_answer_callback)
from financial_insights.streamlit.utilities_methods import (handle_userinput,
                                                            set_fc_llm)


def get_stock_data_analysis() -> None:
    streamlit.markdown("<h2> Stock Data Analysis </h2>", unsafe_allow_html=True)
    streamlit.markdown(
        '<a href="https://pypi.org/project/yfinance/" target="_blank" '
        'style="color:cornflowerblue;text-decoration:underline;"><h3>via Yahoo! Finance API</h3></a>',
        unsafe_allow_html=True,
    )
    streamlit.markdown("<h3> Info retrieval </h3>", unsafe_allow_html=True)

    output = streamlit.empty()  # type: ignore

    user_request = streamlit.text_input(
        "Enter the info that you want to retrieve for given companies",
        key="ticker_symbol",
    )
    if streamlit.button("Retrieve stock info"):
        with streamlit.expander("**Execution scratchpad**", expanded=True):
            response_string = handle_stock_info(user_request)

            save_path = "stock_info.txt"
            content = user_request + "\n\n" + response_string + "\n\n\n"
            if streamlit.button(
                "Save Answer",
                on_click=save_string_answer_callback,
                args=(content, save_path),
            ):
                pass

    if streamlit.button(label="Get financial summary"):
        with streamlit.expander("**Execution scratchpad**", expanded=True):
            response_dict = handle_financial_summary(user_request)
            save_path = "summary_" + "_".join(list(response_dict.keys()))
            if streamlit.button(
                "Save Analysis",
                on_click=save_dict_answer_callback,
                args=(response_dict, save_path),
            ):
                pass

    streamlit.markdown("<br><br>", unsafe_allow_html=True)
    streamlit.markdown("<h3> Stock data analysis </h3>", unsafe_allow_html=True)
    output = streamlit.empty()  # type: ignore
    ticker_list = streamlit.text_input(
        "Enter the quantities that you want to plot for given companies\n"
        "Suggested values: Open, High, Low, Close, Volume, Dividends, Stock Splits."
    )
    start_date = streamlit.date_input("Start Date")
    end_date = streamlit.date_input("End Date")

    # Analyze stock data
    if streamlit.button("Analyze Stock Data"):
        with streamlit.expander("**Execution scratchpad**", expanded=True):
            data, fig = handle_stock_data_analysis(ticker_list, start_date, end_date)  # type: ignore

        if streamlit.button(
            "Save Analysis",
            on_click=save_dataframe_figure_callback,
            args=(ticker_list, data, fig),
        ):
            pass

def handle_stock_info(user_question: Optional[str]) -> Any:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question (str): The user's question or input.
    """
    if user_question is None:
        return None

    streamlit.session_state.tools = [
        "retrieve_symbol_list",
        "get_stock_info",
        "get_conversational_response",
    ]
    set_fc_llm(streamlit.session_state.tools)

    user_request = (
        "Please answer the following query for the following companies "
        "(expressed via their ticker symbols):\n" + user_question + "\n"
        "Reformulate the final answer in the form of a conversational response to the user.\n"
        "Take your time and reason step by step.\n"
    )

    return handle_userinput(user_question, user_request)

def handle_financial_summary(user_question: str) -> Any:
    """
    Handle user input and generate a response, also update chat UI in streamlit app
    """
    streamlit.session_state.tools = [
        "retrieve_symbol_list",
        "get_financial_summary",
        "get_conversational_response",
    ]
    set_fc_llm(streamlit.session_state.tools)

    if user_question is None:
        return None

    user_request = (
        "Please answer the following query for the following companies "
        "(expressed via their ticker symbols):\n" + user_question
    )
    return handle_userinput(user_question, user_request)

def handle_stock_data_analysis(
    user_question: str, start_date: datetime.date, end_date: datetime.date
) -> Tuple[pandas.DataFrame, plotly.graph_objs.Figure]:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        symbol (str): The user's question or input.
    """
    if user_question is None:
        return None

    streamlit.session_state.tools = [
        "retrieve_symbol_quantity_list",
        "get_historical_price",
    ]
    set_fc_llm(
        tools=streamlit.session_state.tools,
        default_tool=None,
    )
    if start_date is not None or end_date is not None:
        user_request = (
            "Please fetch the following market information for the following stocks "
            "(expressed via their ticker symbols) "
            "and within the following dates.\n" + user_question
        )
        user_request += f"\n. The requested dates are from {start_date} to {end_date}"

    response = handle_userinput(user_question, user_request)

    assert (
        isinstance(response, tuple)
        and len(response) == 2
        and isinstance(response[0], pandas.DataFrame)
        and isinstance(response[1], plotly.graph_objs.Figure)
    ), f"Invalid response: {response}"

    return response  # type: ignore