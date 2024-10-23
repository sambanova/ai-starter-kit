#!/usr/bin/env python3
"""
Model Wrappers (MW) Test Script

This script tests the functionality of the Model Wrappers using unittest.

Usage:
    python tests/model_wrappers_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import logging
import os
import sys

# from time import sleep
import time
import unittest
from typing import Any, Dict, List, Type

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths and global variables
current_dir = os.getcwd()
kit_dir = current_dir  # absolute path for function_calling kit dir
repo_dir = os.path.abspath(os.path.join(kit_dir, '../..'))  # absolute path for ai-starter-kit root repo

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from pydantic import ValidationError
from utils.model_wrappers.api_gateway import APIGateway
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv
import yaml

load_dotenv(os.path.join(repo_dir, '.env'), override=True)

def load_test_config(config_path: str) -> Any:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


test_config = load_test_config(kit_dir + '/tests/test_config.yaml')
interface = APIGateway
bad_format_chat_model_params = test_config["bad_format_chat_model"]
sn_chat_model_params = test_config["sn_chat_model"]
ss_chat_model_params = test_config["sambastudio_chat_model"]

sn_chat_model = interface.load_chat(
    **sn_chat_model_params
)

ss_chat_model = interface.load_chat(
    **ss_chat_model_params
)

class ModelWrapperTestCase(unittest.TestCase):
    time_start: float
    sn_chat_model: BaseChatModel

    @classmethod
    def setUpClass(cls: Type['ModelWrapperTestCase']) -> None:
        cls.time_start = time.time()
        cls.sn_chat_model = sn_chat_model
        cls.ss_chat_model = ss_chat_model

    def test_chat_model_creation(self) -> None:
        self.assertIsNotNone(self.sn_chat_model, 'chat model class could not be created')
        self.assertIsNotNone(self.ss_chat_model, 'chat model class could not be created')

    def test_chat_model_params(self) -> None:
        self.assertEqual(self.sn_chat_model.dict(), {"model": "llama3-8b", "streaming": False, "max_tokens": 1024,
                                                "temperature": 0.7, "top_k": 1, "top_p": 0.01,
                                                "stream_options": {"include_usage": True},
                                                "_type": "sambanovacloud-chatmodel"
                                            })
    
    def test_chat_model_validation(self) -> None:
        with self.assertRaises(ValidationError) as context:
            interface.load_chat(**bad_format_chat_model_params[0])
        self.assertIn("Input should be a valid string", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            interface.load_chat(**bad_format_chat_model_params[1])
        self.assertIn("Input should be a valid boolean", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            interface.load_chat(**bad_format_chat_model_params[2])
        self.assertIn("Input should be a valid integer", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            interface.load_chat(**bad_format_chat_model_params[3])
        self.assertIn("Input should be a valid number", str(context.exception))

        with self.assertRaises(ValidationError) as context:
            interface.load_chat(**bad_format_chat_model_params[4])
        self.assertIn("Input should be a valid dictionary", str(context.exception))

    def test_chat_model_param(self) -> None:
        test_cases = [
            {"type":"sncloud", "model": "foo", "top_k": 1, "top_p": 0.01},
            {"type":"sambastudio", "model": "foo", "top_k": 1, "top_p": 0.01, "temperature": 0.0},
        ]

        for case in test_cases:
            llm = interface.load_chat(**case)
            self.assertEqual(llm.model, "foo", "Model name should be 'foo'")
            self.assertTrue(hasattr(llm, "temperature"), "temperature attribute should exist")
            self.assertEqual(llm.temperature, 0.0, "temperature default should be set to 0.0")

    @classmethod
    def tearDownClass(cls: Type['ModelWrapperTestCase']) -> None:
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
    suite = unittest.TestLoader().loadTestsFromTestCase(ModelWrapperTestCase)
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
        