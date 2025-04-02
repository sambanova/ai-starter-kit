import os
import json

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))

SAMBANOVA_URL = 'https://api.sambanova.ai/v1/chat/completions'
CONFIG_PATH = os.path.join(project_root, 'config.yaml')

# TODO: update method name to other files using it
def read_synthetic_json_files(folder_path, type: str='individual_responses'):
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
                    print(f"Error reading {file_path}: {e}")

    df = pd.DataFrame(data)
    df = df.rename(columns=lambda x: x.replace('results_', '').replace('_quantiles', '').replace('_per_request', '')).copy()

    return df

# TODO: implement based on the real workloads
def read_real_workloads_json_files(folder_path, type: str='individual_responses'):
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
                    print(f"Error reading {file_path}: {e}")

    df = pd.DataFrame(data)
    df = df.rename(columns=lambda x: x.replace('results_', '').replace('_quantiles', '').replace('_per_request', '')).copy()

    return df