#!/usr/bin/env python3
"""
Function Calling API models Test Script

This script tests the function calling for API models compatible using unittest.

Usage:
    python utils/tests/fc_testing.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import logging
import os
import sys
import time
import unittest
from typing import Any, Dict, List, Type

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

import asyncio
import json

from dotenv import load_dotenv
from openai import OpenAI

from utils.tests.schemas import ContactForm, DataExtraction, Solution, YFinanceSourceList
from utils.tests.utils_test import function_calling, mcp_client, read_json_file

load_dotenv()

CONFIG_PATH = os.path.join(os.getcwd(), 'tests', 'config.yaml')

with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

client_base_url = config['urls']['base_client_url']

fc_models = config['models']['function-calling']

tools_path = 'tests/data/tools.json'
schemas_path = 'tests/data/schemas.json'
available_tools = read_json_file(tools_path)
available_schemas = read_json_file(schemas_path)

tool_calling_test_cases = [
    (
        [
            {
                'role': 'user',
                'content': "I am based in Cambridge and need to catch a train soon. What's the current time?",
            }
        ],
        ['get_current_time', 'tavily_search'],
    ),
    (
        [{'role': 'user', 'content': 'What is beta risk metric?'}],
        [
            'get_current_time',
            'tavily_search',
            'yahoo_finance_search',
            'exa_news_search',
        ],
    ),
    (
        [{'role': 'user', 'content': 'How are you?'}],
        ['get_current_time', 'tavily_search'],
    ),
    (
        [{'role': 'user', 'content': 'What is a Large Language Model'}],
        [
            'get_current_time',
            'tavily_search',
            'yahoo_finance_search',
            'exa_news_search',
        ],
    ),
    ([{'role': 'user', 'content': 'What time is it right now?'}], ['get_current_time']),
    ([{'role': 'user', 'content': 'Find recent news about AI advancements.'}], ['tavily_search']),
    (
        [{'role': 'user', 'content': 'Tell me the time and also find the latest SpaceX launch updates.'}],
        ['tavily_search'],
    ),
    (
        [
            {
                'role': 'user',
                'content': 'Search for research papers about large language models published in the last 6 months.',
            }
        ],
        ['tavily_search'],
    ),
    (
        [{'role': 'user', 'content': 'What is the current weather in San Francisco?'}],
        ['get_current_weather'],
    ),
    (
        [
            {'role': 'user', 'content': 'What is 1 + 2'},
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': '8007d97c-eb39-43de-8bef-a8e6b83b8244',
                        'type': 'function',
                        'function': {'name': 'my_adder_tool', 'arguments': '{"a":1,"b":2}'},
                    }
                ],
            },
            {'role': 'tool', 'content': '{"result": 3}', 'tool_call_id': '8007d97c-eb39-43de-8bef-a8e6b83b8244'},
            {'role': 'assistant', 'content': '{"result": 3}'},
            {'role': 'user', 'content': 'What is 3 + 4'},
        ],
        ['sum_of_integers'],
    ),
    (
        [
            {'role': 'user', 'content': "What's the weather like in scotland today?"},
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': '8007d97c-eb39-43de-8bef-a8e6b83b8244',
                        'type': 'function',
                        'function': {'name': 'get_weather', 'arguments': '{"city":"Scotland"}'},
                    }
                ],
            },
            {'role': 'tool', 'content': '{"city": "Scotland" ,"temperature_celsius": 25}'},
        ],
        ['get_weather'],
    ),
    (
        [ {"role": "system", "content": "You are a helpful math and conversion assistant. Use tools when needed."}, {"role": "user", "content": "How many cups are in 3 gallons?"} ],
        ['solve_addition', 'convert_units', 'convert_currency', 'calculate_date_difference'],
    ),
    (
        [ {"role": "system", "content": "You are a helpful math and conversion assistant. Use tools when needed."}, {"role": "user", "content": "what is 4 + 5?"} ],
        ['solve_addition', 'convert_units', 'convert_currency', 'calculate_date_difference']
    ),
    (
        [ {"role": "system", "content": "You are a helpful math and conversion assistant. Use tools when needed."}, {"role": "user", "content": "what is 5 km in miles?"} ],
        ['solve_addition', 'convert_units', 'convert_currency', 'calculate_date_difference']
    ),
    (
        [ {"role": "system", "content": "You are a helpful math and conversion assistant. Use tools when needed."}, {"role": "user", "content": "how many pints are in 6 cups?"} ],
        ['solve_addition', 'convert_units', 'convert_currency', 'calculate_date_difference']
    ),
    (
        [ {"role": "system", "content": "You are a helpful math and conversion assistant. Use tools when needed."}, {"role": "user", "content": "I need to pay my card. It says I owe 347 euros, but I pay in dollars. How much is that?"} ],
        ['solve_addition', 'convert_units', 'convert_currency', 'calculate_date_difference']
    )
]

structured_output_test_cases = [
    (
        [
            {
                'role': 'system',
                'content': """You are an expert at structured data extraction. You will be given unstructured text
                  should convert it into the given structure.""",
            },
            {'role': 'user', 'content': 'the section 24 has appliances, and videogames'},
        ],
        available_schemas['data_extraction'],
        DataExtraction,
    ),
    (
        [
            {
                'role': 'user',
                'content': 'generate the contact info from following customer message \n hi my name is jason tiller'
                ' i have an issue with my new tv, my number is 35554774, plese call me back',
            }
        ],
        available_schemas['contact_form'],
        ContactForm,
    ),
    (
        [
            {
                'role': 'system',
                'content': 'You are a helpful math tutor. Guide the user through the solution step by step.',
            },
            {'role': 'user', 'content': 'how can I solve 8x + 7 = -23'},
        ],
        available_schemas['math_reasoning'],
        Solution,
    ),
    (
        [{ "role": "user", "content": "YFinance results:\nMeta: income 234M, debt: 34M, headcount: 38314\nGoogle: income 634M, debt: 314M, headcount: 667314\n\nWhat is the current income and loans that meta has?"} ],
        available_schemas['YFinanceSourceList'],
        YFinanceSourceList
    )
]


class TestFCAPIModel(unittest.TestCase):
    time_start: float
    client: OpenAI
    sambanova_api_key: str

    @classmethod
    def setUpClass(cls: Type['TestFCAPIModel']) -> None:
        cls.time_start = time.time()
        cls.sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY', '')
        cls.client = cls.setup_client()

    @classmethod
    def setup_client(cls: Type['TestFCAPIModel']) -> OpenAI:
        client = OpenAI(
            base_url=client_base_url,
            api_key=cls.sambanova_api_key,
        )
        return client

    def test_client_function_calling(self) -> None:
        for test_case in tool_calling_test_cases:
            for model in fc_models:
                messages = test_case[0]
                function_names = test_case[1]
                response = function_calling(
                    client=self.client, model=model, messages=messages,
                      tools=[available_tools[tool] for tool in function_names], stream=False
                )

                self.assertTrue(hasattr(response, 'content'))
                self.assertTrue(hasattr(response, 'tool_calls'))
                self.assertIsNone(response.content)
                self.assertIsNotNone(response.tool_calls)
                self.assertTrue(response.tool_calls[0].function.name in function_names)

    def test_client_function_calling_mcp(self) -> None:
        for test_case in tool_calling_test_cases:
            for model in fc_models:
                messages = test_case[0]
                function_names = test_case[1]

                response = asyncio.run(mcp_client([available_tools[tool] for tool in function_names], self.client, model, messages, False))
                self.assertTrue(hasattr(response, 'content'))
                self.assertTrue(hasattr(response, 'tool_calls'))
                self.assertIsNone(response.content)
                self.assertIsNotNone(response.tool_calls)
                self.assertTrue(response.tool_calls[0].function.name in function_names)

    def test_client_structured_output(self) -> None:
        for test_case in structured_output_test_cases:
            for model in fc_models:
                messages = test_case[0]
                response_format = {
                    'type': 'json_schema',
                    'json_schema': test_case[1],
                }
                response = function_calling(
                    client=self.client, model=model, messages=messages, response_format=response_format, stream=False
                )

                test_case[2](**json.loads(response.content))

    @classmethod
    def tearDownClass(cls: Type['TestFCAPIModel']) -> None:
        time_end = time.time()
        total_time = time_end - cls.time_start
        logger.info(f'Total execution time: {total_time:.2f} seconds')


class CustomTextTestResult(unittest.TextTestResult):
    test_results: List[Dict[str, Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.test_results: List[Dict[str, Any]] = []

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self.test_results.append({'name': test._testMethodName, 'status': 'PASSED'})

    def addFailure(self, test: unittest.TestCase, err: Any) -> None:
        super().addFailure(test, err)
        self.test_results.append({'name': test._testMethodName, 'status': 'FAILED', 'message': str(err[1])})

    def addError(self, test: unittest.TestCase, err: Any) -> None:
        super().addError(test, err)
        self.test_results.append({'name': test._testMethodName, 'status': 'ERROR', 'message': str(err[1])})


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFCAPIModel)
    test_result = unittest.TextTestRunner(resultclass=CustomTextTestResult).run(suite)

    logger.info('\nTest Results:')
    assert hasattr(test_result, 'test_results')
    for result in test_result.test_results:
        logger.info(f"{result['name']}: {result['status']}")
        if 'message' in result:
            logger.info(f"  Message: {result['message']}")

    failed_tests = len(test_result.failures) + len(test_result.errors)
    logger.info(f'\nTests passed: {test_result.testsRun - failed_tests}/{test_result.testsRun}')

    if failed_tests:
        logger.error(f'Number of failed tests: {failed_tests}')
        return failed_tests
    else:
        logger.info('All tests passed successfully!')
        return 0


if __name__ == '__main__':
    sys.exit(main())
