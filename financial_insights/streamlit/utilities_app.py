import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import pandas
import plotly
import streamlit
import yaml

logging.basicConfig(level=logging.INFO)
from streamlit.elements.widgets.time_widgets import DateWidgetReturn

from financial_insights.streamlit.constants import *


def _get_config_info(config_path: str = CONFIG_PATH) -> Dict[str, str]:
    """
    Loads json config file
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
    fig: plotly.graph_objs.Figure,
    start_date: DateWidgetReturn,
    end_date: DateWidgetReturn,
    save_path: Optional[str] = None,
) -> None:
    dir_name = CACHE_DIR + 'history_figures/'
    # Create temporary cache for storing historical price data
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    filename = dir_name + f'stock_data_{'_'.join(symbol_list)}_{start_date}_{end_date}'

    # Write the dataframe to a csv file
    data.to_csv(filename + '.csv', index=True)
    # Save the plots
    fig_bytes = fig.to_image(format='png')
    with open(f'{filename}.png', 'wb') as f:
        f.write(fig_bytes)

    content = '\n\n' + user_query + '\n\n' + f'{filename}.png' + '\n\n'
    # Save the figure path to a file
    save_output_callback(content, HISTORY_PATH)

    if save_path is not None:
        save_output_callback(content, save_path)


def save_output_callback(response: Union[str, List[str], Dict[str, str]], save_path: str) -> None:
    # Create temporary cache for storing historical price data
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Specify the filename
    filename = save_path

    if isinstance(response, str):
        # Writing the string to a txt file
        with open(filename, 'a') as text_file:
            text_file.write('\n\n' + response + '\n\n')

    elif isinstance(response, dict):
        # Writing the dictionary to a JSON file
        with open(filename, 'a') as json_file:
            json_file.write('\n\n')
            json.dump(response, json_file)
            json_file.write('\n\n')

    elif isinstance(response, list):
        # Writing the list to a JSON file
        with open(filename, 'a') as json_file:
            json_file.write('\n\n')
            json.dump(response, json_file)
            json_file.write('\n\n')

    elif isinstance(response, tuple):
        # Writing the tuple to a JSON file
        with open(filename, 'a') as json_file:
            json_file.write('\n\n')
            json.dump(response, json_file)
            json_file.write('\n\n')
    else:
        raise ValueError('Invalid response type')


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
        font-size: 1,
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #e0e0e0;
        background-color: #1e1e1e;
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        margin-bottom: 1em;
    }

    /* Paragraph and text styling */
    p, label {
        font-size: 1;
        line-height: 1.6;
        margin-bottom: 0.5em;
        color: #e0e0e0;
    }

    /* Button styling */
    .stButton > button {
        background-color: green;
        color: white;
        padding: 0.75em 1.5em;
        font-size: 1;
        border: none;
        border-radius: 5px;
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
    input[type="text"], input[type="date"], select {
        width: 100%;
        padding: 0.75em;
        margin: 0.5em 0 1em 0;
        display: inline-block;
        border: 1px solid #ccc;
        border-radius: 4px;
        box-sizing: border-box;
        font-size: 1.1em;
        background-color: #2c2c2c;
        color: #e0e0e0;
    }

    /* Checkbox styling */
    .stCheckbox > label {
        font-size: 1.1em;
    }

    /* Container styling */
    .main {
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
            background-color: blue;
            color: black;
            padding: 0.75em 1.5em;
            font-size: 1;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }"""


def get_orange_button_style() -> str:
    return """
        button {
            background-color: orange;
            color: black;
            padding: 0.75em 1.5em;
            font-size: 1;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }"""
