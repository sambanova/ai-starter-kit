import datetime
import json
from typing import Any, List, Optional, Tuple

import pandas
import streamlit
from matplotlib.figure import Figure
from streamlit.elements.widgets.time_widgets import DateWidgetReturn

from financial_assistant.streamlit.constants import *
from financial_assistant.streamlit.utilities_app import save_historical_price_callback, save_output_callback
from financial_assistant.streamlit.utilities_methods import attach_tools, handle_userinput


def get_stock_data_analysis() -> None:
    """Include the app for the stock data analysis."""

    streamlit.markdown('<h2> Stock Data Analysis </h2>', unsafe_allow_html=True)
    streamlit.markdown(
        '<a href="https://pypi.org/project/yfinance/" target="_blank" '
        'style="color:cornflowerblue;text-decoration:underline;"><h3>via Yahoo! Finance API</h3></a>',
        unsafe_allow_html=True,
    )
    streamlit.markdown('<h3> Info retrieval </h3>', unsafe_allow_html=True)

    user_request = streamlit.text_input(
        label='Enter the info that you want to retrieve for a given company. '
        f':sparkles: :violet[{DEFAULT_STOCK_QUERY}]',
        key='stock-query',
        placeholder='E.g. ' + DEFAULT_STOCK_QUERY,
    )

    columns = streamlit.columns(3, vertical_alignment='center')
    dataframe_name = columns[0].selectbox(
        'Select Data Source:',
        sorted(
            [
                'None',
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
        ),
        index=9,
    )

    with open(YFINANCE_COLUMNS_JSON, 'r') as json_file:
        dataframes_json = json.load(json_file)
    columns[2].write('Show columns per dataframe')
    columns[2].json(body=dataframes_json, expanded=False)

    columns[1].markdown(
        f'<span style="color:rgb{SAMBANOVA_ORANGE}">Show selected columns</span>',
        unsafe_allow_html=True,
    )
    if dataframe_name is not None and dataframe_name != 'None':
        dataframe_columns = dataframes_json[dataframe_name]
        columns[1].json(dataframe_columns, expanded=False)
    else:
        columns[1].write(None)

    if streamlit.button('Retrieve stock info'):
        if len(user_request) == 0:
            streamlit.error('Please enter your query.')
        else:
            with streamlit.expander('**Execution scratchpad**', expanded=True):
                response_dict = handle_stock_query(user_request, dataframe_name)

                # Save the output to the history file
                save_output_callback(response_dict, streamlit.session_state.history_path, user_request)

                # Save the output to the stock query file
                if streamlit.button(
                    'Save Answer',
                    on_click=save_output_callback,
                    args=(response_dict, streamlit.session_state.stock_query_path, user_request),
                ):
                    pass

    streamlit.markdown('<br><br>', unsafe_allow_html=True)
    streamlit.markdown('<h3> Stock data history </h3>', unsafe_allow_html=True)
    user_request = streamlit.text_input(
        label='Enter the quantities that you want to plot for given companies. '
        'Suggested values: Open, High, Low, Close, Volume, Dividends, Stock Splits. '
        f':sparkles: :violet[{DEFAULT_HISTORICAL_STOCK_PRICE_QUERY}]',
        key='historical-stock-price-query',
        placeholder='E.g. ' + DEFAULT_HISTORICAL_STOCK_PRICE_QUERY,
    )
    start_date = streamlit.date_input(
        'Start Date', value=datetime.datetime.now() - datetime.timedelta(days=365), key='start-date'
    )
    end_date = streamlit.date_input('End Date', value=datetime.datetime.now(), key='end-date')

    # Analyze stock data
    if streamlit.button('Analyze Historical Stock Data'):
        if len(user_request) == 0:
            streamlit.error('Please enter your query.')
        else:
            with streamlit.expander('**Execution scratchpad**', expanded=True):
                fig, data, symbol_list = handle_stock_data_analysis(user_request, start_date, end_date)

            # Save the output to the history file
            save_historical_price_callback(
                user_request, symbol_list, data, fig, start_date, end_date, streamlit.session_state.history_path
            )

            # Save the output to the stock query file
            if streamlit.button(
                'Save Analysis',
                on_click=save_historical_price_callback,
                args=(
                    user_request,
                    symbol_list,
                    data,
                    fig,
                    start_date,
                    end_date,
                    streamlit.session_state.stock_query_path,
                ),
            ):
                pass


def handle_stock_query(
    user_question: Optional[str],
    dataframe_name: Optional[str] = None,
) -> Any:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question: The user's question or input.
        dataframe_name: The name of the dataframe to be used for analysis.

    Returns:
        The LLM response to the user's question.
    """
    if user_question is None:
        return None

    if dataframe_name is None:
        dataframe_name = 'None'

    # Declare the permitted tools for function calling
    streamlit.session_state.tools = ['get_stock_info']

    # Attach the tools for the LLM to use
    attach_tools(streamlit.session_state.tools)

    # Compose the user request
    user_request = f"""
        Please answer the following query based on a given list of companies:
        {user_question}

        First, extract the company (or companies) from the user query.
        Then, use the dataframe "{dataframe_name}"
        to retrieve the relevant company information before providing your answer.
    """

    return handle_userinput(user_question, user_request)


def handle_stock_data_analysis(
    user_question: str, start_date: DateWidgetReturn, end_date: DateWidgetReturn
) -> Tuple[pandas.DataFrame, Figure, List[str]]:
    """
    Handle the user request for the historical stock data analysis.

    Args:
       user_question: The user's question.
       start_date: The start date of the historical price.
       end_date: The end date of the historical price.

    Returns:
        A tuple with the following elements:
            - The figure with historical price data.
            - A dataframe with historical price data.
            - The list of company ticker symbols.

    Raises:
        ValueError: If `start_date` and `end_date` are both None.
        TypeError: If `response` does not conform to the return type.
    """
    # Check inputs
    assert start_date is not None or end_date is not None, ValueError('Start date or end date must be provided.')

    if user_question is None:
        return None

    # Declare the permitted tools for function calling
    streamlit.session_state.tools = ['get_historical_price']

    # Attach the tools for the LLM to use
    attach_tools(
        tools=streamlit.session_state.tools,
        default_tool=None,
    )

    # Compose the user request
    user_request = f"""
        Fetch historical stock prices for a given list of companies from {start_date} to {end_date}.
        Extract the company (or companies) from the user query.
        
        User query: {user_question}
    """

    # Call the LLM on the user request with the attached tools
    response = handle_userinput(user_question, user_request)

    # Check the final answer of the LLM
    assert (
        isinstance(response, tuple)
        and len(response) == 3
        and isinstance(response[0], Figure)
        and isinstance(response[1], pandas.DataFrame)
        and isinstance(response[2], list)
        and all([isinstance(i, str) for i in response[2]])
    ), TypeError(f'Invalid response: {response}.')

    return response
