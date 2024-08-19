from typing import Any, List, Optional, Tuple

import pandas
import plotly
import streamlit
from streamlit.elements.widgets.time_widgets import DateWidgetReturn

from financial_insights.streamlit.constants import *
from financial_insights.streamlit.utilities_app import save_historical_price_callback, save_output_callback
from financial_insights.streamlit.utilities_methods import handle_userinput, set_fc_llm


def get_stock_data_analysis() -> None:
    streamlit.markdown('<h2> Stock Data Analysis </h2>', unsafe_allow_html=True)
    streamlit.markdown(
        '<a href="https://pypi.org/project/yfinance/" target="_blank" '
        'style="color:cornflowerblue;text-decoration:underline;"><h3>via Yahoo! Finance API</h3></a>',
        unsafe_allow_html=True,
    )
    streamlit.markdown('<h3> Info retrieval </h3>', unsafe_allow_html=True)

    output = streamlit.empty()

    user_request = streamlit.text_input(
        'Enter the info that you want to retrieve for given companies.',
        key='stock-query',
    )
    dataframe_name = streamlit.selectbox(
        'Select Data Source:',
        [
            'info',
            'history',
            'history_metadata',
            'actions',
            'dividends',
            'splits',
            'capital_gains',
            'shares',
            'income_stmt',
            'quarterly_income_stmt',
            'balance_sheet',
            'quarterly_balance_sheet',
            'cashflow',
            'quarterly_cashflow',
            'major_holders',
            'institutional_holders',
            'mutualfund_holders',
            'insider_transactions',
            'insider_purchases',
            'insider_roster_holders',
            'sustainability',
            'recommendations',
            'recommendations_summary',
            'upgrades_downgrades',
            'earnings_dates',
            'isin',
            'options',
            'news',
        ],
    )
    if streamlit.button('Retrieve stock info'):
        with streamlit.expander('**Execution scratchpad**', expanded=True):
            response = handle_stock_query(user_request, dataframe_name)

            response = user_request + '\n' + response
            save_output_callback(response, HISTORY_PATH)

            if streamlit.button(
                'Save Answer',
                on_click=save_output_callback,
                args=(response, STOCK_QUERY_PATH),
            ):
                pass

    streamlit.markdown('<br><br>', unsafe_allow_html=True)
    streamlit.markdown('<h3> Stock data history </h3>', unsafe_allow_html=True)
    output = streamlit.empty()
    user_request = streamlit.text_input(
        'Enter the quantities that you want to plot for given companies\n'
        'Suggested values: Open, High, Low, Close, Volume, Dividends, Stock Splits.'
    )
    start_date = streamlit.date_input('Start Date')
    end_date = streamlit.date_input('End Date')

    # Analyze stock data
    if streamlit.button('Analyze Historical Stock Data'):
        with streamlit.expander('**Execution scratchpad**', expanded=True):
            fig, data, symbol_list = handle_stock_data_analysis(user_request, start_date, end_date)

        save_historical_price_callback(user_request, symbol_list, data, fig, start_date, end_date, HISTORY_PATH)

        if streamlit.button(
            'Save Analysis',
            on_click=save_historical_price_callback,
            args=(user_request, symbol_list, data, fig, start_date, end_date, STOCK_QUERY_PATH),
        ):
            pass


def handle_stock_query(
    user_question: Optional[str],
    dataframe_name: Optional[str] = None,
) -> Any:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question (str): The user's question or input.
    """
    if user_question is None:
        return None

    if dataframe_name is None:
        dataframe_name = 'None'

    streamlit.session_state.tools = [
        'retrieve_symbol_list',
        'get_stock_info',
        'get_conversational_response',
    ]
    set_fc_llm(streamlit.session_state.tools)

    user_request = (
        'Please answer the following query for the following companies '
        '(expressed via their ticker symbols):\n' + user_question + '\n'
        f'Retrieve the company info using the dataframe "{dataframe_name}".\n'
        'Reformulate the final answer in the form of a conversational response to the user.\n'
        'Take your time and reason step by step.\n'
    )

    return handle_userinput(user_question, user_request)


def handle_stock_data_analysis(
    user_question: str, start_date: DateWidgetReturn, end_date: DateWidgetReturn
) -> Tuple[pandas.DataFrame, plotly.graph_objs.Figure, List[str]]:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        symbol (str): The user's question or input.
    """
    if user_question is None:
        return None

    streamlit.session_state.tools = [
        'retrieve_symbol_quantity_list',
        'get_historical_price',
    ]
    set_fc_llm(
        tools=streamlit.session_state.tools,
        default_tool=None,
    )
    if start_date is not None or end_date is not None:
        user_request = (
            'Please fetch the following market information for the following stocks '
            '(expressed via their ticker symbols) '
            'and within the following dates.\n' + user_question
        )
        user_request += f'\nThe requested dates are from {start_date} to {end_date}'

    response = handle_userinput(user_question, user_request)

    assert (
        isinstance(response, tuple)
        and len(response) == 3
        and isinstance(response[0], plotly.graph_objs.Figure)
        and isinstance(response[1], pandas.DataFrame)
        and isinstance(response[2], list)
        and all([isinstance(i, str) for i in response[2]])
    ), f'Invalid response: {response}'

    return response
