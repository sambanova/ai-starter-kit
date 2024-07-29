import json
import logging
import os
import sys
from typing import Dict, List

import pandas
import plotly
import streamlit
import yaml  # type: ignore

logging.basicConfig(level=logging.INFO)
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

TEMP_DIR = 'financial_insights/streamlit/cache/'
SOURCE_DIR = 'financial_insights/streamlit/cache/sources/'
CONFIG_PATH = 'financial_insights/config.yaml'

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


