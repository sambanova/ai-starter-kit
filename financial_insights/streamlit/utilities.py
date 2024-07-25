import datetime
import json
import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from datetime import date
from io import StringIO
from time import sleep
from typing import (Any, Callable, Dict, Generator, List, Optional, Tuple,
                    Type, Union)

import pandas
import plotly
import plotly.graph_objects as go
import streamlit
import yaml
import yfinance
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import load_prompt
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import \
    SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import StructuredTool, Tool
from PIL import Image
from secedgar import FilingType, filings

from financial_insights.src.function_calling import \
    FunctionCallingLlm  # type: ignore
from financial_insights.src.function_calling import ConversationalResponse
from financial_insights.src.tools_filings import retrieve_filings
from financial_insights.src.tools_stocks import (get_conversational_response,
                                                 get_financial_summary,
                                                 get_historical_price,
                                                 get_stock_info,
                                                 plot_price_over_time,
                                                 retrieve_symbol_list,
                                                 retrieve_symbol_quantity_list)
from financial_insights.src.tools_yahoo_news import scrape_yahoo_finance_news
from function_calling.src.tools import get_time  # type: ignore
from function_calling.src.tools import (calculator, python_repl, query_db, rag,
                                        translate)

logging.basicConfig(level=logging.INFO)
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

# tool mapping of available tools
TOOLS = {
    'get_time': get_time,
    'calculator': calculator,
    'python_repl': python_repl,
    'query_db': query_db,
    'translate': translate,
    'rag': rag,
    'get_stock_info': get_stock_info,
    'get_historical_price': get_historical_price,
    'retrieve_symbol_list': retrieve_symbol_list,
    'retrieve_symbol_quantity_list': retrieve_symbol_quantity_list,
    'scrape_yahoo_finance_news': scrape_yahoo_finance_news,
    'get_financial_summary': get_financial_summary,
    'get_conversational_response': get_conversational_response,
    'retrieve_filings': retrieve_filings,
}

TEMP_DIR = 'financial_insights/streamlit/cache/'
SOURCE_DIR = 'financial_insights/streamlit/cache/sources/'
CONFIG_PATH = 'financial_insights/config.yaml'


@contextmanager  # type: ignore
def st_capture(output_func: Callable[[Any], None]) -> None:
    """
    context manager to catch stdout and send it to an output streamlit element

    Args:
        output_func (function to write terminal output in

    Yields:
        Generator:
    """
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string: str) -> int:
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write  # type: ignore
        yield


def set_fc_llm(
    tools: List[str],
    default_tool: Optional[
        Union[StructuredTool, Tool, Type[BaseModel]]
    ] = ConversationalResponse,
) -> None:
    """
    Set the FunctionCallingLlm object with the selected tools

    Args:
        tools (list): list of tools to be used
    """
    set_tools = [TOOLS[name] for name in tools]
    streamlit.session_state.fc = FunctionCallingLlm(
        tools=set_tools, default_tool=default_tool
    )


def handle_userinput(user_question: Optional[str], user_query: Optional[str]) -> Optional[Any]:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question (str): The user's question or input.
    """
    output = streamlit.empty()

    with streamlit.spinner('Processing...'):
        with st_capture(output.code):  # type: ignore
            tool_messages, response = streamlit.session_state.fc.function_call_llm(
                query=user_query,
                max_it=streamlit.session_state.max_iterations,
                debug=True,
            )

    streamlit.session_state.chat_history.append(user_question)
    streamlit.session_state.chat_history.append(response)

    with streamlit.chat_message('user'):
        streamlit.write(f'{user_question}')

    # Convert JSON string to dictionary
    try:
        # Try to convert the string to a dictionary
        response_dict = json.loads(response)
        # Check if the result is a dictionary
        if isinstance(response_dict, dict):
            response = response_dict

    except (json.JSONDecodeError, TypeError):
        # If JSON decoding fails, return the original string
        pass

    if isinstance(response, str):
        with streamlit.chat_message(
            'ai',
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            if not response.endswith('.png'):
                streamlit.write(response)
            else:
                # Load the image
                image = Image.open(response)

                # Display the image
                streamlit.image(
                    image, use_column_width=True
                )

    elif isinstance(response, dict):
        for key, value in response.items():
            if isinstance(value, str):
                with streamlit.chat_message(
                    'ai',
                    avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
                ):
                    if not value.endswith('.png'):
                        streamlit.write(value)
                    else:
                        # Load the image
                        image = Image.open(value)

                        # Display the image
                        streamlit.image(
                            value, use_column_width=True
                        )

    return response


def handle_stock_info(user_question: Optional[str]) -> Any:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question (str): The user's question or input.
    """
    if user_question is None:
        return None

    streamlit.session_state.tools = [
        'retrieve_symbol_list',
        'get_stock_info',
        'get_conversational_response',
    ]
    set_fc_llm(streamlit.session_state.tools)

    user_request = (
        'Please answer the following query for the following companies '
        '(expressed via their ticker symbols):\n' + user_question + '\n'
        'Reformulate the final answer in the form of a conversational response to the user.\n'
        'Take your time and reason step by step.\n'
    )

    return handle_userinput(user_question, user_request)


def handle_financial_summary(user_question: str) -> Any:
    """
    Handle user input and generate a response, also update chat UI in streamlit app
    """
    streamlit.session_state.tools = [
        'retrieve_symbol_list',
        'get_financial_summary',
        'get_conversational_response',
    ]
    set_fc_llm(streamlit.session_state.tools)

    if user_question is None:
        return None

    user_request = (
        'Please answer the following query for the following companies '
        '(expressed via their ticker symbols):\n' + user_question
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
        user_request += (
            f'\n. The requested dates are from {start_date} to {end_date}'
        )

    response = handle_userinput(user_question, user_request)

    if isinstance(response, tuple):
        data = response[0]
        fig = response[1]
    else:
        raise ValueError('Invalid response')

    return data, fig


def handle_yfinance_news(user_question: str) -> Tuple[str, List[str]]:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
    user_request (str): The user's question or input.
    """
    streamlit.session_state.tools = [
        'retrieve_symbol_list',
        'scrape_yahoo_finance_news',
    ]
    set_fc_llm(
        tools=streamlit.session_state.tools,
        default_tool=None,
    )
    user_request = (
        'Please fetch the following market information for the following companies '
        '(preferably expressed via their ticker symbols) ' + user_question
    )

    response = handle_userinput(user_question, user_request)

    # Load the dataframe from the text file
    try:
        df = pandas.read_csv(SOURCE_DIR + 'scraped_data_with_urls.csv')
    except FileNotFoundError:
        logging.error('No scraped data found.')

    # Convert DataFrame rows into Document objects
    documents = []
    for _, row in df.iterrows():
        document = Document(
            page_content=row['text'],
            metadata={'url': row['url'], 'type': row['type']},
        )
        documents.append(document)

    response = get_qa_response(documents, user_request)

    answer = response['answer']
    url_list = list(
        {doc.metadata['url'] for doc in response['source_documents']}
    )


    with streamlit.chat_message('user'):
        streamlit.write(user_request)

    with streamlit.chat_message(
        'ai',
        avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    ):
        streamlit.write(answer)
        for url in url_list:
            streamlit.write(url)

    return answer, url_list


def handle_financial_filings(
    user_question: str,
    company_name: str,
    filing_type: Optional[str] = '10-K',
    filing_quarter: Optional[int] = 0,
    selected_year: Optional[int] = 2023,
) -> Tuple[str, str]:
    streamlit.session_state.tools = ['retrieve_symbol_list', 'retrieve_filings']
    set_fc_llm(streamlit.session_state.tools)

    user_request = (
        'You are an expert in the stock market.\n'
        + user_question
        + f'In order to provide context for your question, please retrieve the given SEC EDGAR financial filing type: {filing_type}, and filing quarter: '
        f'{filing_quarter} for the company {company_name} for the year {selected_year}.\n'
    )

    filename = handle_userinput(user_question, user_request)

    # Load the dataframe from the text file
    try:
        df = pandas.read_csv(SOURCE_DIR + f'{filename}' + '.csv')
    except FileNotFoundError:
        logging.error('No scraped data found.')

    # Convert DataFrame rows into Document objects
    documents = []
    for _, row in df.iterrows():
        document = Document(page_content=row['text'])
        documents.append(document)

    response = get_qa_response(documents, user_request)

    answer = response['answer']

    with streamlit.chat_message('user'):
        streamlit.write(user_request)

    with streamlit.chat_message(
        'ai',
        avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    ):
        streamlit.write(answer)

    return answer, filename


def get_qa_response(
    documents: List[Document],
    user_request: str,
) -> Any:
    # Set up the embedding model and vector store
    embedding_model = SentenceTransformerEmbeddings(
        model_name='paraphrase-mpnet-base-v2'
    )
    vectorstore = Chroma.from_documents(documents, embedding_model)

    # Load config
    config = _get_config_info(CONFIG_PATH)

    # Load retrieval prompt
    prompt = load_prompt(
        os.path.join(
            kit_dir, 'prompts/llama30b-web_crawling_data_retriever.yaml'
        )
    )
    retriever = vectorstore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            'score_threshold': config['tools']['rag']['retrieval']['score_threshold'],  # type: ignore
            'k': config['tools']['rag']['retrieval']['k_retrieved_documents'],  # type: ignore
        },
    )

    # Create a retrieval-based QA chain
    qa_chain = RetrievalQA.from_llm(
        llm=streamlit.session_state.fc.llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        verbose=True,
        input_key='question',
        output_key='answer',
        prompt=prompt,
    )

    # Function to answer questions based on the news data
    response = qa_chain.invoke({'question': user_request})

    return response


def _get_config_info(config_path: str = CONFIG_PATH) -> Dict[str, str]:
    """
    Loads json config file
    Args:
        path (str, optional): The path to the config file. Defaults to CONFIG_PATH.
    Returns:
        api_info (string): string containing API to use SambaStudio or Sambaverse.
        embedding_model_info (string): String containing embedding model type to use, SambaStudio or CPU.
        llm_info (dict): Dictionary containing LLM parameters.
        retrieval_info (dict): Dictionary containing retrieval parameters
        web_crawling_params (dict): Dictionary containing web crawling parameters
        extra_loaders (list): list containing extra loader to use when doing web crawling
            (only pdf available in base kit)
    """
    # Read config file
    with open(config_path, 'r') as yaml_file:
        config_file = yaml.safe_load(yaml_file)

    # Convert the config file to a dictionary
    config = dict(config_file)

    return config


# Save dataframe and figure callback for streamlit button
def save_dataframe_figure_callback(
    ticker_list: str, data: pandas.DataFrame, fig: plotly.graph_objs.Figure
) -> None:
    # Create temporary cache for storing historical price data
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    # Write the dataframe to a csv file
    data.to_csv(TEMP_DIR + f'stock_data_{ticker_list}.csv', index=False)
    # Save the plots
    fig_bytes = fig.to_image(format='png')
    with open(TEMP_DIR + f'stock_data_{ticker_list}.png', 'wb') as f:
        f.write(fig_bytes)
    fig.write_image(TEMP_DIR + f'stock_data_{ticker_list}.png')


def save_dict_answer_callback(response_dict: str, save_path: str) -> None:
    # Create temporary cache for storing historical price data
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Specify the filename
    filename = TEMP_DIR + save_path

    # Writing the dictionary to a JSON file
    with open(filename, 'a') as json_file:
        json.dump(response_dict, json_file)


def save_string_answer_callback(response: str, save_path: str) -> None:
    # Create temporary cache for storing historical price data
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Specify the filename
    filename = TEMP_DIR + save_path

    # Writing the string to a txt file
    with open(filename, 'a') as text_file:
        text_file.write(response + '\n')


def list_files_in_directory(directory: str) -> List[str]:
    """List all files in the given directory."""
    # Create temporary cache for storing historical price data
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    return [
        f
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]


def clear_directory(directory: str) -> None:
    """Delete all files in the given directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            streamlit.error(f'Error deleting file {file_path}: {e}')
