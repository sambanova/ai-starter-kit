#!/usr/bin/env python3
"""
Function Calling (FC) Test Script

This script tests the functionality of the Function Calling using unittest.

Usage:
    python tests/fc_test.py

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
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))  # absolute path for ai-starter-kit root repo

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from function_calling.src.function_calling import FunctionCallingLlm
from function_calling.src.tools import QueryDb, calculator, get_time, python_repl

tools = ['get_time', 'calculator', 'python_repl', 'query_db']


class FCTestCase(unittest.TestCase):
    time_start: float
    fc: FunctionCallingLlm

    @classmethod
    def setUpClass(cls: Type['FCTestCase']) -> None:
        cls.time_start = time.time()
        cls.fc = FunctionCallingLlm(tools=tools)  # type: ignore

    # Add assertions
    def test_fc_initialization(self) -> None:
        self.assertIsInstance(self.fc, FunctionCallingLlm, 'fc is not of type FunctionCallingLlm')

    def test_get_time_tool(self) -> None:
        result = get_time.invoke({'kind': 'both'})
        logger.info('\nInvoking get_time:')
        logger.info(result)
        self.assertTrue(result, "get_time result shouldn't be empty")

    def test_calculator_tool(self) -> None:
        assert hasattr(calculator, 'invoke')
        result = calculator.invoke('5*23.7 -5')
        logger.info('\nInvoking calculator:')
        logger.info(result)
        self.assertTrue(result, "calculator result shouldn't be empty")

    def test_python_repl_tool(self) -> None:
        assert hasattr(python_repl, 'invoke')
        result = python_repl.invoke({'command': 'for i in range(0,5):\n\tprint(i)'})
        logger.info('\nInvoking python_repl:')
        logger.info(result)
        self.assertTrue(result, "python_repl result shouldn't be empty")

    def test_query_db_tool(self) -> None:
        result = QueryDb().get_tool().invoke({'query': 'How many genres of music are in the chinook db'})
        logger.info('\nInvoking query_db:')
        logger.info(result)
        self.assertTrue(result, "query_db result shouldn't be empty")

    def test_fc_pipeline(self) -> None:
        response = self.fc.function_call_llm('what time is it?', max_it=5)
        logger.info('\nFunction calling pipeline with get_time:')
        logger.info(response)
        self.assertTrue(response, "LLM final response shouldn't be empty")

        response = self.fc.function_call_llm('calculate 5*10-3', max_it=5)
        logger.info('\nFunction calling pipeline with calculator:')
        logger.info(response)
        self.assertTrue(response, "LLM final response shouldn't be empty")

        response = self.fc.function_call_llm(
            "sort this list of elements alphabetically ['screwdriver', 'pliers', 'hammer']", max_it=5
        )
        logger.info('\nFunction calling pipeline with python_repl:')
        logger.info(response)
        self.assertTrue(response, "LLM final response shouldn't be empty")

        response = self.fc.function_call_llm(
            'whats the price in colombian pesos of the track "Snowballed" in the db if one usd is equal to 3800 cop?',
            max_it=5,
        )
        logger.info('\nFunction calling pipeline with query_db and calculator:')
        logger.info(response)
        self.assertTrue(response, "LLM final response shouldn't be empty")

    @classmethod
    def tearDownClass(cls: Type['FCTestCase']) -> None:
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
    suite = unittest.TestLoader().loadTestsFromTestCase(FCTestCase)
    test_result = unittest.TextTestRunner(resultclass=CustomTextTestResult).run(suite)

    logger.info('\nTest Results:')
    assert hasattr(test_result, 'test_results')
    for result in test_result.test_results:
        logger.info(f'{result["name"]}: {result["status"]}')
        if 'message' in result:
            logger.info(f'  Message: {result["message"]}')

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
