import json
import os
import re
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Any, Callable, Generator, List, Optional, Tuple

import pandas
import streamlit
from matplotlib.figure import Figure
from PIL import Image

from financial_assistant.constants import *
from financial_assistant.src.exceptions import (
    LLM_EXCEPTION_MESSAGE,
    LLMException,
    TableNotFoundException,
    VectorStoreException,
)
from financial_assistant.src.tools import get_conversational_response
from financial_assistant.src.tools_database import create_stock_database, query_stock_database
from financial_assistant.src.tools_filings import retrieve_filings
from financial_assistant.src.tools_pdf_generation import pdf_rag
from financial_assistant.src.tools_stocks import (
    get_historical_price,
    get_stock_info,
)
from financial_assistant.src.tools_yahoo_news import scrape_yahoo_finance_news
from financial_assistant.src.utilities import get_logger
from financial_assistant.streamlit.llm_model import sambanova_llm

logger = get_logger()

# tool mapping of available tools
TOOLS = {
    'get_stock_info': get_stock_info,
    'get_historical_price': get_historical_price,
    'scrape_yahoo_finance_news': scrape_yahoo_finance_news,
    'get_conversational_response': get_conversational_response,
    'retrieve_filings': retrieve_filings,
    'create_stock_database': create_stock_database,
    'query_stock_database': query_stock_database,
    'pdf_rag': pdf_rag,
}


@contextmanager
def st_capture(output_func: Callable[[Any], Any]) -> Generator[None, None, None]:
    """
    Context manager to catch `stdout` and send it to an output function.

    Args:
        output_func: Function to which the terminal output is written.

    Returns:
        A generator that redirects `stdout` to the `output_func`.
    """
    stdout = StringIO()
    with redirect_stdout(stdout):
        try:
            yield
        finally:
            output_func(stdout.getvalue())


def handle_userinput(user_question: str, user_query: str) -> Optional[Any]:
    """
    Handle the user input and generate a response, also update chat UI in the Streamlit app.

    Args:
        user_question: The resulting prompt or input.
        user_query: The original user's query.

    Returns:
        The LLM reponse.
    """
    output = streamlit.empty()

    with streamlit.spinner('Processing...'):
        with st_capture(output.code):
            try:
                # Invoke the tools on the user query
                response = sambanova_llm.invoke_tools(query=user_query)

                # Stream the duration of the LLM calls
                stream_time_llm()

                # Delete LLM time json file
                os.remove(streamlit.session_state.time_llm_path)

            except LLMException:
                streamlit.error(LLM_EXCEPTION_MESSAGE)
                streamlit.stop()
            except VectorStoreException:
                streamlit.error('Could not instantiate the vectorstore.')
                streamlit.stop()
            except TableNotFoundException:
                streamlit.error('No relevant SQL tables found.')
                streamlit.stop()

    streamlit.session_state.chat_history.append(user_question)
    streamlit.session_state.chat_history.append(response)

    with streamlit.chat_message('user'):
        streamlit.write(f'{user_question}')

    stream_response_object(response)

    return response


def stream_time_llm() -> None:
    """Stream LLM call duration."""

    if os.path.exists(streamlit.session_state.time_llm_path):
        with open(streamlit.session_state.time_llm_path, 'r') as file:
            data = json.load(file)
            while data:
                streamlit.markdown(
                    f'- **<span style="color:green">{data[0][0]}</span>'
                    f'** took **<span style="color:green">'
                    f'{float(data[0][1]):.2f}</span>** seconds.',
                    unsafe_allow_html=True,
                )
                data.pop(0)
    return


def stream_chat_history() -> None:
    """Stream the chat history."""

    for question, answer in zip(
        streamlit.session_state.chat_history[::2],
        streamlit.session_state.chat_history[1::2],
    ):
        with streamlit.chat_message('user'):
            streamlit.write(question)

        stream_response_object(answer)


def stream_response_object(response: Any) -> Any:
    """Stream the LLM response."""

    if (
        isinstance(response, (str, float, int, Figure, pandas.DataFrame))
        or isinstance(response, list)
        or isinstance(response, dict)
    ):
        return stream_complex_response(response)

    elif isinstance(response, tuple):
        json_response = list()
        for item in response:
            json_response.append(stream_complex_response(item))
        return json.dumps(json_response)
    else:
        return


def stream_complex_response(response: Any) -> Any:
    """Stream a complex LLM response."""

    if isinstance(response, (str, float, int, Figure, pandas.DataFrame)):
        stream_single_response(response)

    elif isinstance(response, list):
        for item in response:
            stream_single_response(item)

    elif isinstance(response, dict):
        for key, value in response.items():
            if isinstance(value, (str, float, int)):
                if isinstance(value, float):
                    value = round(value, 2)
                stream_single_response(str(value))
            elif isinstance(value, list):
                # If all values are strings
                stream_single_response(', '.join([str(item) for item in value]))
            elif isinstance(value, (pandas.Series, pandas.DataFrame)):
                stream_single_response(value)

    try:
        return json.dumps(response)
    except:
        return ''


def stream_single_response(response: Any) -> None:
    """Stream a simple LLM response."""

    # If response is a string
    if isinstance(response, (str, float, int)):
        response = str(response)
        with streamlit.chat_message(
            'ai',
            avatar=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
        ):
            if isinstance(response, str):
                # If images are not present in the response, treat it as pure text
                if '.png' not in response:
                    # Clean and write the response
                    markdown_response = escape_markdown(response)
                    streamlit.write(markdown_response)

                # If images are present in the response,
                # treat it as the combination of a possible text and a list of images
                else:
                    # Extract the list of images and any remaining text from response
                    png_paths, text = extract_png_paths(response)

                    # If there is text in the response, write it first
                    if len(text) > 0:
                        # Clean and write the response
                        markdown_response = escape_markdown(text)
                        streamlit.write(text)

                    # Then, display each image
                    for path in png_paths:
                        try:
                            # Load the image
                            image = Image.open(path)
                            # Display the image
                            streamlit.image(image, use_container_width=True)
                        except FileNotFoundError:
                            logger.error(f'Image file not found: {path}.')
                        except Exception as e:
                            logger.error(f'Error displaying image: {path}. Error: {str(e)}.')

    # If response is a figure
    elif isinstance(response, Figure):
        # Display the image
        streamlit.pyplot(response)

    # If response is a series or a dataframe, display its head
    elif isinstance(response, (pandas.Series, pandas.DataFrame)):
        if len(response) <= 30:
            streamlit.write(response)
        else:
            streamlit.write(response.head(30))


def extract_png_paths(sentence: str) -> Tuple[List[str], str]:
    """Extract all png paths and any remaining text from a string."""

    # png image pattern
    png_pattern = re.compile(r'\/\b\S+\.png\b')

    # Extract all image absolute paths
    png_paths: List[str] = re.findall(png_pattern, sentence)

    # Extract the path list
    path_list = [path.strip() for path in png_paths]

    # Extract any remaining text
    text = re.sub(png_pattern, '', sentence).strip()

    # Patterns for removing some combinations of special characters
    colon_space_regex = re.compile(r': , ')
    final_colon_space_regex = re.compile(r': \.')

    # Replace colon and period spaces
    text = colon_space_regex.sub(', ', text)
    text = final_colon_space_regex.sub('.', text)

    # Replace any final colon with a period
    if text.endswith(':'):
        text = text[:-1] + '.'

    return path_list, text


def escape_markdown(text: str) -> str:
    """
    Escape special characters in the given text to make it compatible with `Markdown`.

    Args:
        text: The input text to be escaped.

    Returns:
        The escaped text.
    """
    # List of characters that need to be escaped in Markdown
    special_chars = ['\\', '{', '}', '[', ']', '(', ')', '>', '#', '$', '`', '|']

    # Escape each character by adding a backslash
    for char in special_chars:
        text = text.replace(char, '\\' + char)

    return text
