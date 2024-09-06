import datetime
import json
import os
import re
import shutil
import time
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pandas
import schedule
import streamlit
import yaml
from matplotlib.figure import Figure
from streamlit.elements.widgets.time_widgets import DateWidgetReturn

from financial_insights.src.tools import get_logger
from financial_insights.streamlit.constants import *
from utils.visual.env_utils import initialize_env_variables

logger = get_logger()


def _get_config_info(config_path: str = CONFIG_PATH) -> Dict[str, str]:
    """
    Loads json config file.

    Args:
        path: The path to the config file.
        Defaults to CONFIG_PATH.
    Returns:
        A dictionary with the config information:
            - api_info: string containing API to use:
                `fastapi`, `sambastudio` or `sambaverse`.
            - embedding_model_info:
                String containing embedding model type to use,
                `sambastudio` or `cpu`.
            - llm_info: Dictionary containing LLM parameters.
            - retrieval_info:
                Dictionary containing retrieval parameters.
            - web_crawling_params:
                Dictionary containing web crawling parameters.
            - extra_loaders:
                List containing extra loader to use when doing web crawling
                (only pdf available in base kit).
    """
    # Read config file
    with open(config_path, 'r') as yaml_file:
        config_file = yaml.safe_load(yaml_file)

    # Convert the config file to a dictionary
    config = dict(config_file)

    return config


def save_historical_price_callback(
    user_query: str,
    symbol_list: List[str],
    data: pandas.DataFrame,
    fig: Figure,
    start_date: DateWidgetReturn,
    end_date: DateWidgetReturn,
    save_path: Optional[str] = None,
) -> None:
    """Save dataframe and figure callback for streamlit button."""

    # Derive the directory name
    dir_name = streamlit.session_state.history_figures_dir

    # Create the directory for storing historical price data
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Derive the filename
    filename = dir_name + f"stock_data_{'_'.join(symbol_list)}_{start_date}_{end_date}"

    # Write the dataframe to a csv file
    data.to_csv(filename + '.csv', index=True)

    # Save the plots
    fig.savefig(f'{filename}.png', bbox_inches='tight')

    # Compose the content including the user query and the filename
    content = '\n\n' + user_query + '\n\n' + f'{filename}.png' + '\n\n'

    # Save the content path to a file
    if save_path is not None:
        save_output_callback(content, save_path)


def save_output_callback(
    response: str | List[str] | Dict[str, str],
    save_path: str,
    user_request: Optional[str] = None,
) -> None:
    """Save the output callback for streamlit button."""
    # Check the inputs
    assert isinstance(response, (str, list, dict, tuple, pandas.Series, pandas.DataFrame)), TypeError(
        f'Response must be a string, a list, a dictionary, a tuple. a series, or a dataframe. Got type {type(response)}'
    )
    assert isinstance(save_path, str), TypeError('Save path must be a string.')
    assert isinstance(user_request, (str, type(None))), TypeError('User request must be a string.')

    # Specify the filename
    filename = save_path

    # Opening space
    with open(filename, 'a') as text_file:
        text_file.write('\n\n')

    # Add the user query
    if user_request is not None:
        with open(filename, 'a') as text_file:
            text_file.write(user_request)
            text_file.write('\n\n')

    # If the response is a set, convert it to a list
    if isinstance(response, set):
        response = list(response)

    # If the reponse is a simple element, save it as it is
    if isinstance(response, (str, float, int, pandas.Series, pandas.DataFrame)):
        # Write the string to a txt file
        save_simple_output(response, filename)

    # If the response is a list, split it up and write each element individually
    elif isinstance(response, list):
        for elem in response:
            save_simple_output(elem + '\n', filename)

    # If the response is a dict, split it up and write pair of key and value individually
    elif isinstance(response, dict):
        for key, value in response.items():
            with open(filename, 'a') as text_file:
                if isinstance(value, str):
                    save_simple_output(value + '\n', filename)
                elif isinstance(value, list):
                    save_simple_output(', '.join([str(item) for item in value]) + '.' + '\n', filename)

    # If the response is a tuple, split it up and write each element individually
    elif isinstance(response, tuple):
        for elem in response:
            save_simple_output(response + '\n', filename)

    else:
        raise ValueError('Invalid response type')

    # Closing space
    with open(filename, 'a') as text_file:
        text_file.write('\n\n')


def save_simple_output(
    response: Any,
    filename: str,
) -> None:
    """
    Saves the response to a file in the specified path.

    Args:
        response: The response to be saved.
        filename: The path to save the response in.
    """
    # If the response is a string or number, write it directly into the text file
    if isinstance(response, (str, float, int)):
        with open(filename, 'a') as text_file:
            text_file.write(str(response))

    # If the response is a Series or DataFrame, convert it to a dictionary and then dump it to a JSON file
    elif isinstance(response, (pandas.Series, pandas.DataFrame)):
        # Write the dataframe to a CSV file
        response_dict = response.to_dict()

        # Convert the dictionary into a string
        stripped_string = dump_stripped_json(response_dict)

        # Write the stripped string to a txt file
        with open(filename, 'a') as text_file:
            text_file.write(stripped_string)
    else:
        raise ValueError('Invalid response type')

    # Spaces between elements or closing space
    with open(filename, 'a') as text_file:
        text_file.write('\n\n')


def dump_stripped_json(data: Any, indent: int = 2) -> str:
    """
    Dump a JSON-serializable object to a string, stripping parentheses and apostrophes.

    Args:
        data: The JSON-serializable object (dict or list) to dump.
        indent: The indentation level for pretty-printing (default is 2).

    Returns:
        A string representation of the JSON data without parentheses and apostrophes.
    """
    # Dump the data to a JSON string
    json_string = json.dumps(data, indent=indent)

    # Remove parentheses and apostrophes
    stripped_string = re.sub(r'["\'{}]', '', json_string)

    # Remove any empty lines that might have been created by stripping
    stripped_string = '\n'.join(line for line in stripped_string.split('\n') if line.strip())

    return stripped_string


def list_files_in_directory(directory: str) -> List[str]:
    """List all files in the given directory."""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def list_directory(directory: str) -> Tuple[List[str], List[str]]:
    """List subdirectories and files in the given directory."""
    subdirectories = []
    files = []

    if os.getcwd() == repo_dir:
        pass
    elif os.getcwd() == kit_dir:
        os.chdir(os.path.realpath(os.path.dirname(os.getcwd())))

    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            subdirectories.append(name)
        else:
            files.append(name)
    return subdirectories, files


def display_directory_contents(path: str, default_path: str) -> None:
    """Display subdirectories and files in the current path."""
    subdirectories, files = list_directory(path)

    dir_name = Path(path).name
    if dir_name.startswith('cache'):
        dir_name = 'cache'

    streamlit.sidebar.markdown(f'### Directory: {dir_name}')

    if subdirectories:
        streamlit.sidebar.markdown('#### Subdirectories:')
        for idx, subdir in enumerate(subdirectories):
            if streamlit.sidebar.button(f'📁 {subdir}', key=f'{subdir}_{idx}'):
                streamlit.session_state.current_path = os.path.join(streamlit.session_state.current_path, subdir)

                files_subdir = list_files_in_directory(os.path.join(path, subdir))
                for file in files_subdir:
                    download_file(streamlit.session_state.current_path + '/' + file)

    if files:
        streamlit.sidebar.markdown('#### Files:')
        for file in files:
            download_file(path + '/' + file)

    if len(subdirectories + files) == 0:
        streamlit.write('No files found')

    return


def clear_directory(directory: str) -> None:
    """Delete all files in the given directory."""

    # List subdirectories and files
    subdirectories, files = list_directory(directory)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            streamlit.error(f'Error deleting file {file_path}: {e}')


def clear_cache(delete: bool = False) -> None:
    """Clear and/or delete the cache."""

    # Clear the cache
    clear_directory(streamlit.session_state.cache_dir)
    subdirectories = os.listdir(streamlit.session_state.cache_dir)
    # Delete all directories in the cache
    for directory in subdirectories:
        path = os.path.join(streamlit.session_state.cache_dir, directory)
        clear_directory(path)
        subdirectories = os.listdir(path)
        for subdirectory in subdirectories:
            sub_path = os.path.join(path, subdirectory)
            clear_directory(sub_path)
            if delete:
                # Delete the subdirectory
                os.rmdir(sub_path)
        if delete:
            # Delete the directory
            os.rmdir(path)
    # Delete the cache
    os.rmdir(streamlit.session_state.cache_dir)


def download_file(filename: str) -> None:
    """Add a button to download the file."""

    # Extract the format from the filename
    format = Path(filename).suffix[1:]

    # Extract the correct mime type from the format
    if format == 'txt':
        file_mime = 'text/plain'
    elif format == 'csv':
        file_mime = 'text/csv'
    elif format == 'png':
        file_mime = 'image/png'
    elif format == 'pdf':
        file_mime = 'application/pdf'
    else:
        return
    try:
        with open(filename, 'rb') as f:
            data = f.read()
        streamlit.sidebar.download_button(
            label=f'{Path(filename).name}',
            data=data,
            file_name=filename,
            mime=file_mime,
        )
    except Exception as e:
        logger.warning('Error reading file', str(e))
    except FileNotFoundError as e:
        logger.warning('File not found', str(e))


def initialize_session(
    session_state: streamlit.runtime.state.session_state_proxy.SessionStateProxy,
    prod_mode: bool = False,
) -> None:
    """Initialize the Streamlit `session_state`."""

    # Initialize credentials
    initialize_env_variables(prod_mode)

    # Initialize SEC EDGAR credentials
    if 'SEC_API_ORGANIZATION' not in streamlit.session_state:
        if prod_mode:
            streamlit.session_state.SEC_API_ORGANIZATION = None
        else:
            streamlit.session_state.SEC_API_ORGANIZATION = os.getenv('SEC_API_ORGANIZATION')
    if 'SEC_API_EMAIL' not in streamlit.session_state:
        if prod_mode:
            streamlit.session_state.SEC_API_EMAIL = None
        else:
            streamlit.session_state.SEC_API_EMAIL = os.getenv('SEC_API_EMAIL')

    # Initialize the chat history
    if 'chat_history' not in session_state:
        session_state.chat_history = list()
    # Initialize function calling
    if 'fc' not in session_state:
        session_state.fc = None

    # Initialize the session id
    if 'session_id' not in session_state:
        session_state.session_id = str(uuid4())

    # Initialize cache directory
    if prod_mode:
        if 'cache_dir' not in session_state:
            session_state.cache_dir = CACHE_DIR[:-1] + '/cache' + f'_{session_state.session_id}/'
    else:
        if 'cache_dir' not in session_state:
            session_state.cache_dir = CACHE_DIR

    # Main cache directories
    if 'history_path' not in session_state:
        session_state.history_path = os.path.join(session_state.cache_dir, 'chat_history.txt')
    if 'pdf_generation_directory' not in session_state:
        session_state.pdf_generation_directory = os.path.join(streamlit.session_state.cache_dir, 'pdf_generation/')
    if 'stock_query_path' not in session_state:
        session_state.stock_query_path = os.path.join(streamlit.session_state.cache_dir, 'stock_query.txt')
    if 'db_query_path' not in session_state:
        session_state.db_query_path = os.path.join(streamlit.session_state.cache_dir, 'db_query.txt')
    if 'yfinance_news_path' not in session_state:
        session_state.yfinance_news_path = os.path.join(streamlit.session_state.cache_dir, 'yfinance_news.txt')
    if 'filings_path' not in session_state:
        session_state.filings_path = os.path.join(streamlit.session_state.cache_dir, 'filings.txt')
    if 'pdf_rag_path' not in session_state:
        session_state.pdf_rag_path = os.path.join(streamlit.session_state.cache_dir, 'pdf_rag.txt')
    if 'web_scraping_path' not in session_state:
        session_state.web_scraping_path = os.path.join(streamlit.session_state.cache_dir, 'web_scraping.csv')
    if 'llm_class_logger_path' not in session_state:
        session_state.llm_class_logger_path = os.path.join(streamlit.session_state.cache_dir, 'llm_calls_logger.txt')

    # Main source directories
    if 'source_dir' not in session_state:
        session_state.source_dir = os.path.join(streamlit.session_state.cache_dir, 'sources/')
    if 'db_path' not in session_state:
        session_state.db_path = os.path.join(session_state.source_dir, 'stock_database.db')
    if 'yfinance_news_txt_path' not in session_state:
        session_state.yfinance_news_txt_path = os.path.join(session_state.source_dir, 'yfinance_news_documents.txt')
    if 'yfinance_news_csv_path' not in session_state:
        session_state.yfinance_news_csv_path = os.path.join(session_state.source_dir, 'yfinance_news_documents.csv')
    if 'pdf_sources_directory' not in session_state:
        session_state.pdf_sources_directory = os.path.join(session_state.source_dir, 'pdf_sources/')

    # Main figures directories
    if 'stock_query_figures_dir' not in session_state:
        session_state.stock_query_figures_dir = os.path.join(session_state.cache_dir, 'stock_query_figures/')
    if 'history_figures_dir' not in session_state:
        session_state.history_figures_dir = os.path.join(session_state.cache_dir, 'history_figures/')
    if 'db_query_figures_dir' not in session_state:
        session_state.db_query_figures_dir = os.path.join(session_state.cache_dir, 'db_query_figures/')

    # Launch time
    if 'launch_time' not in session_state:
        session_state.launch_time = datetime.datetime.now()

    # Cache creation
    if 'cache_created' not in streamlit.session_state:
        streamlit.session_state.cache_created = False


def submit_sec_edgar_details() -> None:
    """Add the SEC-EDGAR details to the session state."""

    key = 'sidebar-sec-edgar'
    # Populate SEC-EDGAR credentials
    if streamlit.session_state.SEC_API_ORGANIZATION is None:
        streamlit.session_state.SEC_API_ORGANIZATION = streamlit.text_input(
            'For SEC-EDGAR: "<your organization>"', None, key=key + '-organization'
        )
    if streamlit.session_state.SEC_API_EMAIL is None:
        streamlit.session_state.SEC_API_EMAIL = streamlit.text_input(
            'For SEC-EDGAR: "<name.surname@email_provider.com>"', None, key=key + '-email'
        )
    # Save button
    if streamlit.session_state.SEC_API_ORGANIZATION is None or streamlit.session_state.SEC_API_EMAIL is None:
        if streamlit.button('Save SEC EDGAR details', key=key + '-button'):
            if (
                streamlit.session_state.SEC_API_ORGANIZATION is not None
                and streamlit.session_state.SEC_API_EMAIL is not None
            ):
                streamlit.success('SEC EDGAR details saved successfully!')
            else:
                streamlit.warning('Please enter both SEC_API_ORGANIZATION and SEC_API_KEY')


def create_temp_dir_with_subdirs(dir: str, subdirs: List[str] = []) -> None:
    """Create a temporary directory with specified subdirectories."""

    os.makedirs(dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(subdir, exist_ok=True)


def delete_temp_dir(temp_dir: str) -> None:
    """Delete the temporary directory and its contents."""

    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f'Temporary directory {temp_dir} deleted.')
        except:
            logger.warning(f'Could not delete temporary directory {temp_dir}.')


def schedule_temp_dir_deletion(temp_dir: str, delay_minutes: int) -> None:
    """Schedule the deletion of the temporary directory after a delay."""

    schedule.every(delay_minutes).minutes.do(delete_temp_dir, temp_dir).tag(temp_dir)

    def run_scheduler() -> None:
        while schedule.get_jobs(temp_dir):
            schedule.run_pending()
            time.sleep(1)

    # Run scheduler in a separate thread to be non-blocking
    Thread(target=run_scheduler, daemon=True).start()


def set_css_styles() -> None:
    """Set the CSS style for the streamlit app."""

    streamlit.markdown(
        """
        <style>
        /* General body styling */

        html, body {
            font-size: 16px,
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #e0e0e0;
            background-color: #1e1e1e;
        }

        /* Header styling */
        h1, h2, h3, h4, h5, h6 {
            color: #EE7624;
            margin-bottom: 1em;
        }

        /* Paragraph and text styling */
        p, label {
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 0.5em;
            color: #e0e0e0;
        }

        /* Button styling */
        .stButton > button {
            background-color: #3A8EBA;
            color: white;
            padding: 0.75em 1.5em;
            font-size: 1;
            border: none;
            border-radius: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }

        /* Radio button styling */
        .stRadio > label {
            font-size: 1;
        }
        .stRadio > div > div > label {
            font-size: 1;
            padding: 0.25em 0.75em;
            cursor: pointer;
            color: #e0e0e0;
        }
        .stRadio > div > div {
            margin-bottom: 0.5em;
        }

        /* Input field styling */
        input[type="text"], input[type="date"] select {
            width: 100%;
            padding: 0.75em;
            margin: 0.5em 0 1em 0;
            display: inline-block;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 16px;
            background-color: #2c2c2c;
            color: #e0e0e0;
        }

        /* Checkbox styling */
        .stCheckbox > label {
            font-size: 16px;
        }

        /* Container styling */
        .main {
            font-size: 16px;
            padding: 2em;
            background: #2c2c2c;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 2em;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1e1e1e;
        }
        .css-1d391kg .css-1v3fvcr, .css-1d391kg .css-1l5dyp6 {
            color: #e0e0e0;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def get_blue_button_style() -> str:
    """Get the CSS style for a blue button."""
    return """
        button {
            background-color: #2C3E50;
            color: white;
            padding: 0.75em 1.5em;
            font-size: 1;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }"""
