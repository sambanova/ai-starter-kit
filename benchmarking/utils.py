import json
import os
from typing import Any, Dict

import pandas as pd

import benchmarking.benchmarking_utils as benchmarking_utils

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))

SAMBANOVA_API_BASE = 'https://api.sambanova.ai/v1'
CONFIG_PATH = os.path.join(project_root, 'config.yaml')


def build_standard_prompt(model_name: str, user_prompt: str) -> Dict[str, Any]:
    """
    Build a prompt dict using the same chat-template logic as the kit.
    Args:
        model_name (str): The model name (to determine prompt template)
        user_prompt (str): The user prompt text
    Returns:
        dict: {'name': ..., 'template': ...}
    """
    family_model_type = benchmarking_utils.find_family_model_type(model_name)
    if family_model_type == 'llama3':
        prompt_template = f"""<|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|>\n\
            <|start_header_id|>assistant<|end_header_id|>"""
    else:
        prompt_template = f'[INST]{user_prompt}[/INST]'
    return {'name': 'chat_prompt', 'template': prompt_template}


def read_perf_eval_json_files(folder_path: str, type: str = 'individual_responses') -> pd.DataFrame:
    """Read all JSON files in a folder and return a DataFrame.

    Args:
        folder_path (str): Path to the folder containing JSON files.
        type (str, optional): Type of JSON files to read. Defaults to 'individual_responses'.

    Returns:
        pd.DataFrame: DataFrame containing the data from the JSON files.
    """
    data = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file ends with 'individual_responses.json'
        if type in filename:
            file_path = os.path.join(folder_path, filename)

            # Open and load the JSON file
            with open(file_path, 'r') as file:
                try:
                    json_data = json.load(file)
                    if type == 'individual_responses':
                        json_data = [{'filename': filename, **request_response} for request_response in json_data]
                        data.extend(json_data)
                    else:
                        data.append(json_data)
                except json.JSONDecodeError as e:
                    print(f'Error reading {file_path}: {e}')

    df = pd.DataFrame(data)
    df = df.rename(
        columns=lambda x: x.replace('results_', '').replace('_quantiles', '').replace('_per_request', '')
    ).copy()

    return df
