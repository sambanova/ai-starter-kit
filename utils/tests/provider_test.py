import logging
import os
import sys
from typing import Dict, List
import uuid

import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = os.getcwd()
curr_dir = current_dir
repo_dir = os.path.abspath(os.path.join(curr_dir, '..'))
logger.info(f'kit_dir: {curr_dir}')
logger.info(f'repo_dir: {repo_dir}')

sys.path.append(curr_dir)
sys.path.append(repo_dir)
os.path.join(repo_dir, 'utils', 'tests', 'config.yaml')

from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import asyncio
import csv
import json
import time

from utils.tests.utils_test import read_json_file, function_calling, mcp_client
from utils.tests.schemas import (
    DataExtraction, ContactForm, Solution, YFinanceSourceList, Person, ChickenEgg, CountriesExtraction, UserIntent,
      AnswerSchema, PodcastGeneration, ExtractName)


def get_curl(
    messages,
    tools,
    json_schema,
    model,
    url='<openai_compatible_url>',
    tool_choice='auto',
    parallel_tool_calls=False,
    temperature=0.0,
):
    def convert_to_json_str(input):
        out = str(input)
        out = out.replace('"', '\\"').replace("'", '"')
        out = out.replace(' None', ' null')
        out = out.replace('False', ' false')
        out = out.replace('True', ' true')
        return out

    parallel_tool_calls = convert_to_json_str(parallel_tool_calls)
    messages = convert_to_json_str(messages)

    if tools is not None:
        tools = convert_to_json_str(tools)
        curl = f"""curl --location '{url}/chat/completions' \\
--header 'Content-Type: application/json' \\
--header 'Authorization: Bearer ••••••' \\
--data '{{"messages": {messages}, "model": "{model}", "tools": {tools}, "tool_choice": "{tool_choice}", "parallel_tool_calls": {parallel_tool_calls}, "temperature": {temperature}}}'"""
    elif json_schema is not None:
        json_schema = convert_to_json_str(json_schema)
        curl = f"""curl --location '{url}/chat/completions' \\
--header 'Content-Type: application/json' \\
--header 'Authorization: Bearer ••••••' \\
--data '{{"messages": {messages}, "model": "{model}", "temperature": {temperature}, "response_format": {{"type": "json_schema", "json_schema": {json_schema}}}}}'"""
    else:
        curl = f"""curl --location '{url}/chat/completions' \\
--header 'Content-Type: application/json' \\
--header 'Authorization: Bearer ••••••' \\
--data '{{"messages": {messages}, "model": "{model}", "temperature": {temperature}}}'"""

    return curl


# Environment setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.getcwd(), 'tests', 'config.yaml')
with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

client_base_url = config['urls']['base_client_url']
fc_models = config['models']['function-calling']
sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY', '')

available_tools = read_json_file('tests/data/tools.json')
available_mcp_tools = read_json_file('tests/data/mcp_servers.json')
available_schemas = read_json_file('tests/data/schemas.json')


model_mappings = {
    'Meta-Llama-3.1-405B-Instruct': {
        'SambaNova': 'Meta-Llama-3.1-405B-Instruct',
        'Fireworks': 'accounts/fireworks/models/llama-v3p1-405b-instruct-long',
        'Groq': 'no_available',
        'Cerebras': 'no_available',
        'Together': 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo'
    },
    'Meta-Llama-3.3-70B-Instruct': {
        'SambaNova': 'Meta-Llama-3.3-70B-Instruct',
        'Fireworks': 'accounts/fireworks/models/llama-v3p3-70b-instruct',
        'Groq': 'llama-3.3-70b-versatile',
        'Cerebras': 'llama-3.3-70b',
        'Together': 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
    },
    'Llama-4-Scout-17B-16E-Instruct': {
        'SambaNova': 'Llama-4-Scout-17B-16E-Instruct',
        'Fireworks': 'accounts/fireworks/models/llama4-scout-instruct-basic',
        'Groq': 'meta-llama/llama-4-scout-17b-16e-instruct',
        'Cerebras': 'llama-4-scout-17b-16e-instruct',
        'Together': 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
    },
    'Llama-4-Maverick-17B-128E-Instruct': {
        'SambaNova': 'Llama-4-Maverick-17B-128E-Instruct',
        'Fireworks': 'accounts/fireworks/models/llama4-maverick-instruct-basic',
        'Groq': 'meta-llama/llama-4-maverick-17b-128e-instruct',
        'Cerebras': 'no_available',
        'Together': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'
    },
    'DeepSeek-V3-0324': {
        'SambaNova': 'DeepSeek-V3-0324',
        'Fireworks': 'accounts/fireworks/models/deepseek-v3-0324',
        'Groq': 'no_available',
        'Cerebras': 'no_available',
        'Together': 'deepseek-ai/DeepSeek-V3'
    },
    'Qwen3-32B': {
        'SambaNova': 'Qwen3-32B',
        'Fireworks': 'accounts/fireworks/models/qwen3-30b-a3b',
        'Groq': 'no_available',
        'Cerebras': 'qwen-3-32b',
        'Together': 'Qwen/Qwen3-235B-A22B-fp8-tput'
    }
}


# Define test cases
tool_calling_test_cases = read_json_file('tests/data/fc_examples.json')
mcp_test_cases = read_json_file('tests/data/mcp_examples.json')

class_mapping = {
    'data_extraction': DataExtraction,
    'contact_form': ContactForm,
    'math_reasoning': Solution,
    'YFinanceSourceList': YFinanceSourceList,
    'person': Person,
    'chicken-egg': ChickenEgg,
    'countries_extraction': CountriesExtraction,
    'user_intent': UserIntent,
    'AnswerSchema': AnswerSchema,
    'potcastGeneration': PodcastGeneration,
    'Extract_name': ExtractName

}

structured_output_test_cases = read_json_file('tests/data/so_examples.json')

providers = {
    'SambaNova': OpenAI(base_url=config['urls']['base_client_url'], api_key=os.environ.get('SAMBANOVA_API_KEY', ''), max_retries=2),
    # 'Fireworks': OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ.get('FIREWORKS_API_KEY', ''),max_retries=2),
    # 'Groq': OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get('GROQ_API_KEY', ''), max_retries=2),
    # 'Cerebras': OpenAI(base_url="https://api.cerebras.ai/v1", api_key=os.environ.get('CEREBRAS_API_KEY', ''), max_retries=2),
    # 'Together': OpenAI(base_url="https://api.together.xyz/v1", api_key=os.environ.get('TOGETHER_API_KEY', ''), max_retries=2),
    
}

# Prepare CSV
csv_file = 'fc_test_results.csv'
csv_headers = [
    'run_id',
    'test',
    'type',
    'provider',
    'data_of_test',
    'environment',
    'PR',
    'multiturn',
    'tool_use',
    'ref_schema',
    'system_prompt',
    'test_curl',
    'Meta-Llama-3.1-405B-Instruct',
    'Meta-Llama-3.3-70B-Instruct',
    'Llama-4-Scout-17B-16E-Instruct',
    'Llama-4-Maverick-17B-128E-Instruct',
    'DeepSeek-V3-0324',
    'Qwen3-32B',
]
today = datetime.now().strftime('%Y-%m-%d')

not_failure_errors = ['401', '403', '429', 'This model does not support response format `json_schema`']


def write_to_csv(rows: List[Dict[str, str]]) -> None:
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, delimiter=';')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_tests() -> None:
    total_tests = 0
    failed_tests = 0
    csv_rows = []
    run_id = str(uuid.uuid4())

    for test_type, test_cases in [
        ('Tool Calling', tool_calling_test_cases),
        # ('MCP Server', mcp_test_cases),
        ('Structured Output', structured_output_test_cases),
    ]:
        for test_case in test_cases:
            dummy_model = '<sambanova_cloud_model>'
            if test_type == 'Tool Calling':
                test_curl = get_curl(test_case['messages'],
                                      [available_tools[tool] for tool in test_case['tools']], None, dummy_model)
            # elif test_type == 'MCP Server':
            #     test_curl = get_curl(test_case['messages'],
            #                           [available_mcp_tools[tool] for tool in test_case['tools']], None, dummy_model)
            elif test_type == 'Structured Output':
                test_curl = get_curl(test_case['messages'], None, available_schemas[test_case['schema']], dummy_model)

            base_row = {
                'run_id': run_id,
                'test': 'Test ' + str(test_case['id']),
                'type': test_type,
                'data_of_test': today,
                'environment': 'Cloud Prod',
                'PR': 'N/A',
                'multiturn': test_case['multiturn'],
                'tool_use': test_case['tool_use'] if test_type == 'Tool Calling' else 'N/A',
                'ref_schema': test_case['ref_schema'],
                'system_prompt': test_case['system_prompt'],
                'test_curl': test_curl if test_type != 'MCP Server' else '',
                'Meta-Llama-3.1-405B-Instruct': '',
                'Meta-Llama-3.3-70B-Instruct': '',
                'Llama-4-Scout-17B-16E-Instruct': '',
                'Llama-4-Maverick-17B-128E-Instruct': '',
                'DeepSeek-V3-0324': '',
                'Qwen3-32B': '',
            }

            for provider_name, provider_client in providers.items():
                row = base_row.copy()
                row['provider'] = provider_name
                for model in fc_models:
                    resolved_model = model_mappings.get(model, {}).get(provider_name, model)
                    if resolved_model == 'no_available':
                        row[model] = {'status': 'not_available', 'response': 'model not available'}
                        continue
                    try:
                        if test_type == 'Tool Calling':
                            start_time = time.time()
                            response = function_calling(
                                client=provider_client, model=resolved_model, messages=test_case['messages'],
                                tools=[available_tools[tool] for tool in test_case['tools']], stream=False
                            )
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            if isinstance(response, str):
                                if any([error for error in not_failure_errors if error in response]):
                                    row[model] = {'status': 'error', 'response': f'{response}'}
                                else:
                                    row[model] = {'status': 'failed', 'response': f'{response}'}
                            elif (
                                (response.tool_calls is None and test_case['tool_use'] == 'tool_required')
                                or (
                                    response.tool_calls is None
                                    and response.content is None
                                    and test_case['tool_use'] == 'tool_optional'
                                )
                                or (response.tool_calls is not None and test_case['tool_use'] == 'tool_not_required')
                            ):
                                row[model] = {'status': 'failed', 'response': f'{response}'}
                                failed_tests += 1
                            else:
                                row[model] = {'status': 'passed', 'response': f'{response}',
                                               'response_time': f'{elapsed_time}'}
                        elif test_type == 'MCP Server':
                            start_time = time.time()
                            response = asyncio.run(mcp_client(test_case['tools'], provider_client, resolved_model,
                                                         test_case['messages'], False, test_case['mcp']))
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            if (
                                isinstance(response, str)
                                or (response.tool_calls is None and test_case['tool_use'] == 'tool_required')
                                or (
                                    response.tool_calls is None
                                    and response.content is None
                                    and test_case['tool_use'] == 'tool_optional'
                                )
                                or (response.tool_calls is not None and test_case['tool_use'] == 'tool_not_required')
                            ):
                                row[model] = {'status': 'failed', 'response': f'{response}'}
                                failed_tests += 1
                            else:
                                row[model] = {'status': 'passed', 'response': f'{response}',
                                               'response_time': f'{elapsed_time}'}
                        else:
                            response_format = {
                                'type': 'json_schema',
                                'json_schema': available_schemas[test_case['schema']],
                            }
                            start_time = time.time()
                            response = function_calling(
                                client=provider_client,
                                model=resolved_model,
                                messages=test_case['messages'],
                                response_format=response_format,
                                stream=False,
                            )
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            if isinstance(response, str):
                                if any([error for error in not_failure_errors if error in response]):
                                    row[model] = {'status': 'error', 'response': f'{response}'}
                                else:
                                    row[model] = {'status': 'failed', 'response': f'{response}'}
                            else:
                                try:
                                    class_mapping[test_case['schema']](**json.loads(response.content))
                                    row[model] = {'status': 'passed', 'response': f'{response}',
                                                'response_time': f'{elapsed_time}'}
                                except Exception as e:
                                    row[model] = {'status': 'failed', 'response': f'json schema not parsable or not expected: {response.content}',
                                                'response_time': f'{elapsed_time}'}
                    except Exception as e:
                        logger.error(f'[{model} - {test_type}] Test failed: {e}')
                        row[model] = row[model] = {'status': 'failed', 'response': f'{e}'}
                        failed_tests += 1

                    total_tests += 1

                csv_rows.append(row)

    write_to_csv(csv_rows)
    logger.info(f'\nTotal: {total_tests}, Passed: {total_tests - failed_tests}, Failed: {failed_tests}')
    logger.info(f'Results saved to: {csv_file}')


if __name__ == '__main__':
    run_tests()
