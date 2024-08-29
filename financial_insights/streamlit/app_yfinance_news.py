from typing import List, Tuple

import streamlit

from financial_insights.streamlit.constants import *
from financial_insights.streamlit.utilities_app import save_output_callback
from financial_insights.streamlit.utilities_methods import attach_tools, handle_userinput


def get_yfinance_news() -> None:
    streamlit.markdown('<h2> Financial News Scraping </h2>', unsafe_allow_html=True)
    streamlit.markdown(
        '<a href="https://uk.finance.yahoo.com/" target="_blank" '
        'style="color:cornflowerblue;text-decoration:underline;"><h3>via Yahoo! Finance News</h3></a>',
        unsafe_allow_html=True,
    )
    output = streamlit.empty()

    user_request = streamlit.text_input(
        'Enter the yfinance news that you want to retrieve for given companies',
        key='yahoo_news',
    )

    # Retrieve news
    if streamlit.button('Retrieve News'):
        with streamlit.expander('**Execution scratchpad**', expanded=True):
            if user_request is not None:
                answer, url_list = handle_yfinance_news(user_request)
            else:
                raise ValueError('No input provided')

        if answer is not None:
            content = answer + '\n\n'.join(url_list)

            save_output_callback(content, HISTORY_PATH, user_request)

            if streamlit.button(
                'Save Answer',
                on_click=save_output_callback,
                args=(content, YFINANCE_NEWS_PATH, user_request),
            ):
                pass


def handle_yfinance_news(user_question: str) -> Tuple[str, List[str]]:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
    user_request (str): The user's question or input.
    """
    streamlit.session_state.tools = [
        'scrape_yahoo_finance_news',
    ]
    attach_tools(
        tools=streamlit.session_state.tools,
        default_tool=None,
    )
    user_request = (
        'You are an expert in the stock market. '
        + 'Please answer the following question, that could be general or for a given list of companies. \n'
        + user_question
        + 'First possibly retrieve the list of ticker symbols from the list of company names within the query.\n'
        + 'Then retrieve the news articles from webscraping Yahoo Finance.\n'
        + 'Finally, provide the answer to the user.'
    )

    response = handle_userinput(user_question, user_request)

    assert (
        isinstance(response, tuple)
        and len(response) == 2
        and isinstance(response[0], str)
        and isinstance(response[1], list)
        and all(isinstance(item, str) for item in response[1])
    ), f'Invalid response: {response}'

    return response
