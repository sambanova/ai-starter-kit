import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas
import streamlit
import yaml
from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO)
from streamlit.elements.widgets.time_widgets import DateWidgetReturn

from financial_insights.streamlit.constants import *


def _get_config_info(config_path: str = CONFIG_PATH) -> Dict[str, str]:
    """
    Loads json config file.

    Args:
        path (str, optional): The path to the config file.
        Defaults to CONFIG_PATH.
    Returns:
        api_info (string): string containing API to use:
            SambaStudio or Sambaverse.
        embedding_model_info (string):
            String containing embedding model type to use,
            SambaStudio or CPU.
        llm_info (dict): Dictionary containing LLM parameters.
        retrieval_info (dict):
            Dictionary containing retrieval parameters
        web_crawling_params (dict):
            Dictionary containing web crawling parameters
        extra_loaders (list):
            List containing extra loader to use when doing web crawling
            (only pdf available in base kit)
    """
    # Read config file
    with open(config_path, 'r') as yaml_file:
        config_file = yaml.safe_load(yaml_file)

    # Convert the config file to a dictionary
    config = dict(config_file)

    return config


# Save dataframe and figure callback for streamlit button
def save_historical_price_callback(
    user_query: str,
    symbol_list: List[str],
    data: pandas.DataFrame,
    fig: Figure,
    start_date: DateWidgetReturn,
    end_date: DateWidgetReturn,
    save_path: Optional[str] = None,
) -> None:
    dir_name = HISTORY_FIGURES_DIR
    # Create temporary cache for storing historical price data
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    filename = dir_name + f"stock_data_{'_'.join(symbol_list)}_{start_date}_{end_date}"

    # Write the dataframe to a csv file
    data.to_csv(filename + '.csv', index=True)

    # Save the plots
    fig.savefig(f'{filename}.png', bbox_inches='tight')

    content = '\n\n' + user_query + '\n\n' + f'{filename}.png' + '\n\n'

    # Save the figure path to a file
    if save_path is not None:
        save_output_callback(content, save_path)


def save_output_callback(
    response: Union[str, List[str], Dict[str, str]],
    save_path: str,
    user_request: Optional[str] = None,
) -> None:
    assert isinstance(response, (str, list, dict, tuple, pandas.DataFrame)), TypeError(
        f'Response must be a string, a list, a dictionary, or a tuple. Got type {type(response)}'
    )
    assert isinstance(save_path, str), TypeError('Save path must be a string.')
    assert isinstance(user_request, (str, type(None))), TypeError('User request must be a string.')

    # Specify the filename
    filename = save_path

    # Opening space
    with open(filename, 'a') as text_file:
        text_file.write('\n\n')

    if user_request is not None:
        with open(filename, 'a') as text_file:
            text_file.write(user_request)
            text_file.write('\n\n')

    # Create temporary cache for storing historical price data
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # If the response is a set, convert it to a list
    if isinstance(response, set):
        response = list(response)

    # If the reponse is a simple element, save it as is
    if isinstance(response, (str, float, int, pandas.Series, pandas.DataFrame)):
        # Write the string to a txt file
        save_simple_output(response, filename)

    # If the response is a list, try to dump it to a JSON file,
    # otherwise split it up and write each element individually
    elif isinstance(response, list):
        for elem in response:
            save_simple_output(elem + '\n', filename)

    # If the response is a list, try to dump it to a JSON file,
    # otherwise split it up and write pair of key and value individually
    elif isinstance(response, dict):
        for key, value in response.items():
            with open(filename, 'a') as text_file:
                if isinstance(value, str):
                    save_simple_output(value + '\n', filename)
                elif isinstance(value, list):
                    save_simple_output(', '.join([str(item) for item in value]) + '.' + '\n', filename)

    # If the response is a tuple, try to dump it to a JSON file,
    # otherwise split it up and write each element individually
    elif isinstance(response, tuple):
        for elem in response:
            # If the elements of the dictionary are not JSON serializable
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

        stripped_string = dump_stripped_json(response_dict)

        # Write the stripped string to a txt file
        with open(filename, 'a') as text_file:
            text_file.write(stripped_string)
    else:
        raise ValueError('Invalid response type')

    # Spaces between elements
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
    # First, dump the data to a JSON string
    json_string = json.dumps(data, indent=indent)

    # Remove parentheses and apostrophes
    stripped_string = re.sub(r'["\'{}]', '', json_string)

    # Remove any empty lines that might have been created by stripping
    stripped_string = '\n'.join(line for line in stripped_string.split('\n') if line.strip())

    return stripped_string


def list_files_in_directory(directory: str) -> List[str]:
    """List all files in the given directory."""
    # Create temporary cache for storing historical price data
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def list_directory(directory: str) -> Tuple[List[str], List[str]]:
    """
    List subdirectories and files in the given directory.
    """
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
    """
    Display subdirectories and files in the current path.
    """
    subdirectories, files = list_directory(path)

    streamlit.sidebar.markdown(f'### Directory: {path}')

    if subdirectories:
        streamlit.sidebar.markdown('#### Subdirectories:')
        for idx, subdir in enumerate(subdirectories):
            if streamlit.sidebar.button(f'📁 {subdir}', key=subdir):
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


def download_file(file: str) -> None:
    try:
        # with open(file, 'r') as f:
        with open(file, encoding='utf8', errors='ignore') as f:
            file_content = f.read()
            streamlit.sidebar.download_button(
                label=f'{file}',
                data=file_content,
                file_name=file,
                mime='text/plain',
            )
    except Exception as e:
        logging.warning('Error reading file', str(e))
    except FileNotFoundError as e:
        logging.warning('File not found', str(e))


def set_css_styles() -> None:
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
