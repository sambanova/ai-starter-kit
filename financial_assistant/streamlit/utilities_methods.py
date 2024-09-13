import json
import re
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Any, Callable, Generator, List, Optional, Tuple, Type

import pandas
import streamlit
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import StructuredTool, Tool
from matplotlib.figure import Figure
from PIL import Image

from financial_assistant.src.llm import SambaNovaLLM
from financial_assistant.src.tools import get_conversational_response, get_logger
from financial_assistant.src.tools_database import create_stock_database, query_stock_database
from financial_assistant.src.tools_filings import retrieve_filings
from financial_assistant.src.tools_pdf_generation import pdf_rag
from financial_assistant.src.tools_stocks import (
    get_historical_price,
    get_stock_info,
)
from financial_assistant.src.tools_yahoo_news import scrape_yahoo_finance_news
from financial_assistant.streamlit.constants import *

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


def attach_tools(
    tools: Optional[List[str]] = None,
    default_tool: Optional[StructuredTool | Tool | Type[BaseModel]] = None,
) -> None:
    """
    Attach the tools to the Streamlit session for the LLM to use.

    Args:
        tools: list of tools to be used.
    """
    if tools is not None:
        set_tools = [TOOLS[name] for name in tools]
    else:
        set_tools = [default_tool]
    streamlit.session_state.llm = SambaNovaLLM(tools=set_tools, default_tool=default_tool)


def handle_userinput(user_question: Optional[str], user_query: Optional[str]) -> Optional[Any]:
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
            response = streamlit.session_state.llm.invoke_tools(query=user_query)

    streamlit.session_state.chat_history.append(user_question)
    streamlit.session_state.chat_history.append(response)

    with streamlit.chat_message('user'):
        streamlit.write(f'{user_question}')

    stream_response_object(response)

    return response


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

    if isinstance(response, (str, float, int, Figure, pandas.DataFrame)):
        return stream_complex_response(response)

    elif isinstance(response, list) or isinstance(response, dict):
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
            stream_single_response(item + '\n')

    elif isinstance(response, dict):
        for key, value in response.items():
            if isinstance(value, (str, float, int)):
                if isinstance(value, float):
                    value = round(value, 2)
                stream_single_response(str(value) + '\n')
            elif isinstance(value, list):
                # If all values are strings
                stream_single_response(', '.join([str(item) for item in value]) + '\n')
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
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
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
                            streamlit.image(image, use_column_width=True)
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
