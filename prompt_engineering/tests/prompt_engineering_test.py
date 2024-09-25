#!/usr/bin/env python3
"""
Prompt Engineering Test Script

This script tests the functionality of the Prompt Engineering kit using unittest.

Test cases:
    test_prompt_use_cases_template: checks if the prompt use cases are strings are not empty
    test_prompt_template: checks if the prompts are strings are not empty
    test_llm_answer: checks if the llm response is a string and is not empty

Usage:
    python tests/prompt_engineering_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import logging
import os
import sys
import time
import unittest
from typing import Any, Dict, List, Type

from langchain_core.language_models.llms import LLM

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from dotenv import load_dotenv

from prompt_engineering.src.llm_management import LLMManager

# load env variables
load_dotenv(os.path.join(repo_dir, '.env'))

MODEL = 'llama3-405b'
MODEL_PROMPT = 'Llama3'


class PETestCase(unittest.TestCase):
    time_start: float
    llm_manager: LLMManager
    llm: Any
    prompt_uses_cases: List[str]
    prompts: List[str]

    @classmethod
    def setUpClass(cls: Type['PETestCase']) -> None:
        cls.time_start = time.time()
        cls.llm_manager = LLMManager()
        cls.llm = cls.get_llm()
        cls.prompt_uses_cases = cls.get_prompt_use_cases()
        cls.prompts = cls.get_prompts()

    @classmethod
    def get_llm(cls: Type['PETestCase']) -> LLM:
        return cls.llm_manager.set_llm(MODEL)

    @classmethod
    def get_prompt_use_cases(cls: Type['PETestCase']) -> List[str]:
        return cls.llm_manager.prompt_use_cases

    @classmethod
    def get_prompts(cls: Type['PETestCase']) -> List[str]:
        return [cls.llm_manager.get_prompt_template(MODEL_PROMPT, i) for i in cls.prompt_uses_cases]

    # Add assertions
    def test_prompt_use_cases_template(self) -> None:
        self.assertTrue(len(self.prompt_uses_cases) > 0, 'There should be at least one prompt use case')
        for i in range(len(self.prompt_uses_cases)):
            self.assertTrue(
                isinstance(self.prompt_uses_cases[i], str),
                f'Prompt Use Case: {self.prompt_uses_cases[i]} should be a string',
            )
            self.assertGreaterEqual(
                len(self.prompt_uses_cases[i]), 1, 'The prompt use case should not be an empty string'
            )

    def test_prompt_template(self) -> None:
        self.assertTrue(len(self.prompts) > 0, 'There should be at least one prompt')
        for i in range(len(self.prompts)):
            self.assertTrue(isinstance(self.prompts[i], str), f'Prompt: {self.prompts[i]} should be a string')
            self.assertGreaterEqual(len(self.prompts[i]), 1, 'The prompt should not be an empty string')

    def test_llm_answer(self) -> None:
        for i in self.prompts:
            response = self.llm.invoke(i)
            self.assertTrue(isinstance(response, str), 'The response should be a string')
            self.assertGreaterEqual(len(response), 1, 'The response should not be an empty string')

    @classmethod
    def tearDownClass(cls: Type['PETestCase']) -> None:
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
    suite = unittest.TestLoader().loadTestsFromTestCase(PETestCase)
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
