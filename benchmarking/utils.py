import json
import os

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))

SAMBANOVA_API_BASE = 'https://api.sambanova.ai/v1'
CONFIG_PATH = os.path.join(project_root, 'config.yaml')


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
