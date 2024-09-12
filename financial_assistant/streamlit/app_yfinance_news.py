from typing import List, Tuple

import streamlit

from financial_assistant.streamlit.constants import *
from financial_assistant.streamlit.utilities_app import save_output_callback
from financial_assistant.streamlit.utilities_methods import attach_tools, handle_userinput


def get_yfinance_news() -> None:
    streamlit.markdown('<h2> Financial News Scraping </h2>', unsafe_allow_html=True)
    streamlit.markdown(
        '<a href="https://uk.finance.yahoo.com/" target="_blank" '
        'style="color:cornflowerblue;text-decoration:underline;"><h3>via Yahoo! Finance News</h3></a>',
        unsafe_allow_html=True,
    )

    user_request = streamlit.text_input(
        label='Enter the yfinance news that you want to retrieve for given companies. '
        f':sparkles: :violet[{DEFAULT_RAG_QUERY}]',
        key='yahoo_news',
        placeholder='E.g. ' + DEFAULT_RAG_QUERY,
    )

    # Retrieve news
    if streamlit.button('Retrieve News'):
        if len(user_request) == 0:
            streamlit.error('Please enter your query.')
        else:
            with streamlit.expander('**Execution scratchpad**', expanded=True):
                if user_request is not None:
                    answer, url_list = handle_yfinance_news(user_request)
                else:
                    raise ValueError('No input provided')

            if answer is not None:
                content = answer + '\n\n'.join(url_list)

                # Save the query and answer to the history text file
                save_output_callback(content, streamlit.session_state.history_path, user_request)

                # Save the query and answer to the Yahoo Finance News text file
                if streamlit.button(
                    'Save Answer',
                    on_click=save_output_callback,
                    args=(content, streamlit.session_state.yfinance_news_path, user_request),
                ):
                    pass


def handle_yfinance_news(user_question: str) -> Tuple[str, List[str]]:
    """
    Handle the user request for the Yahoo News data.

    Args:
        user_question: The user input question that is used to retrieve the Yahoo Finance News data.

    Returns:
        A tuple containing the following pair:
            1. The answer to the user query.
            2. A list of links to articles that have been used for retrieval to answer the user query.

    Raises:
        TypeError: If the LLM response does not conform to the return type.
    """
    # Declare the permitted tools for function calling
    streamlit.session_state.tools = [
        'scrape_yahoo_finance_news',
    ]

    # Attach the tools for the LLM to use
    attach_tools(
        tools=streamlit.session_state.tools,
        default_tool=None,
    )
    user_request = f"""
        You are an expert in the stock market.
        Please answer the following question, which may be general or related to a specific list of companies:
        {user_question}

        First, extract the company (or companies) from the user query, if applicable.
        Then, retrieve the relevant news articles by web scraping Yahoo Finance,
        and then provide the answer to the user.
    """

    # Call the LLM on the user request with the attached tools
    response = handle_userinput(user_question, user_request)

    # Check the final answer of the LLM
    assert (
        isinstance(response, tuple)
        and len(response) == 2
        and isinstance(response[0], str)
        and isinstance(response[1], list)
        and all(isinstance(item, str) for item in response[1])
    ), TypeError(f'Invalid response: {response}.')

    return response
