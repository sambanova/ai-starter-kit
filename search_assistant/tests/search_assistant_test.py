#!/usr/bin/env python3
"""
Search Assistant Test Script

This script tests the functionality of the Search Assistant kit using unittest.

Test cases:
    test_search_assistant_class_creation: checks if the SearchAssistant class is successfully created
    test_search_and_scrape: checks if the vectorstore is successfully created
    test_basic_call: tests the 'search and answer' functionality of the kit, verifying that both the LLM answer and
    sources exist and are not empty
    test_retrieval_call: tests the 'search and scrape sites' functionality of the kit, verifying that both the LLM
    answer and source_documents exist and are not empty

Usage:
    python tests/search_assistant_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import logging
import os
import sys
import time
import unittest

import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths and global variables
file_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(file_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from typing import Any, Dict, List, Optional, Type

from search_assistant.src.search_assistant import SearchAssistant


def load_test_config(config_path: str) -> Any:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


test_config = load_test_config(kit_dir + '/tests/test_config.yaml')
tool = test_config['tool']
search_engine = test_config['search_engine']
max_results = test_config['max_results']
query = test_config['query']


class SearchAssistantTestCase(unittest.TestCase):
    time_start: float
    sambanova_api_key: str
    serpapi_api_key: str
    search_assistant: SearchAssistant
    scraper_state: Optional[Dict[str, str]]

    @classmethod
    def setUpClass(cls: Type['SearchAssistantTestCase']) -> None:
        cls.time_start = time.time()
        cls.sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY', '')
        cls.serpapi_api_key = os.environ.get('SERPAPI_API_KEY', '')
        cls.search_assistant = SearchAssistant(
            sambanova_api_key=cls.sambanova_api_key, serpapi_api_key=cls.serpapi_api_key
        )
        cls.scraper_state = cls.search_assistant.search_and_scrape(
            query=query,
            search_method=tool[0],
            max_results=max_results,
            search_engine=search_engine,
        )

    # Add assertions
    def test_search_assistant_class_creation(self) -> None:
        self.assertIsNotNone(self.search_assistant, 'SearchAssistant class could not be created')

    def test_search_and_scrape(self) -> None:
        self.assertIsNotNone(self.search_assistant.vector_store, 'Vector store could not be created')

    def test_basic_call(self) -> None:
        user_question = 'who is the president of America'
        reformulated_query = self.search_assistant.reformulate_query_with_history(user_question)
        response = self.search_assistant.basic_call(
            query=user_question,
            reformulated_query=reformulated_query,
            search_method=tool[0],
            max_results=max_results,
            search_engine=search_engine,
            conversational=True,
        )

        logger.info('Basic call:\n')
        logger.info(user_question)
        logger.info(response['sources'])  # list[str]
        logger.info(response['answer'])  # str

        self.assertIn('sources', response, "Response should have a 'sources' key")
        self.assertGreaterEqual(len(response['sources']), 1, 'There should be at least one source link')
        self.assertIn('answer', response, "Response should have an 'answer' key")
        self.assertTrue(response['answer'], "LLM answer shouldn't be empty")

    def test_retrieval_call(self) -> None:
        user_question = 'who is Albert Einsten?'
        response = self.search_assistant.retrieval_call(user_question)

        logger.info('Retrieval call:\n')
        logger.info(user_question)
        logger.info(response['source_documents'])  # list[Document]
        logger.info(response['answer'])  # str

        self.assertIn('source_documents', response, "Response should have a 'source_documents' key")
        self.assertGreaterEqual(len(response['source_documents']), 1, 'There should be at least one source documents')
        self.assertIn('answer', response, "Response should have an 'answer' key")
        self.assertTrue(response['answer'], "LLM answer shouldn't be empty")

    @classmethod
    def tearDownClass(cls: Type['SearchAssistantTestCase']) -> None:
        time_end = time.time()
        total_time = time_end - cls.time_start
        logger.info(f'Total execution time: {total_time:.2f} seconds')


class CustomTextTestResult(unittest.TextTestResult):
    test_results: List[Dict[str, Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.test_results: List[Dict[str, Any]] = []

    def _get_test_name(self, test: unittest.TestCase) -> str:
        """Handle both TestCase and _ErrorHolder objects."""
        return getattr(test, '_testMethodName', str(test))

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self.test_results.append({'name': self._get_test_name(test), 'status': 'PASSED'})

    def addFailure(self, test: unittest.TestCase, err: Any) -> None:
        super().addFailure(test, err)
        self.test_results.append({'name': self._get_test_name(test), 'status': 'FAILED', 'message': str(err[1])})

    def addError(self, test: unittest.TestCase, err: Any) -> None:
        super().addError(test, err)
        self.test_results.append({'name': self._get_test_name(test), 'status': 'ERROR', 'message': str(err[1])})


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(SearchAssistantTestCase)
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
