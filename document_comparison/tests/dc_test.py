#!/usr/bin/env python3
"""
Document Comparison Test Script

This script tests the functionality of the Document Comparison starter kit using unittest.
It parses documents using the SambaParse service, and tests parsing and LLM capabilites.

Usage:
    python tests/dc_test.py

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
kit_dir = current_dir
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
logger.info(f'kit_dir: {kit_dir}')
logger.info(f'repo_dir: {repo_dir}')

sys.path.append(kit_dir)
sys.path.append(repo_dir)


from io import BytesIO

from document_comparison.src.document_analyzer import DocumentAnalyzer

PERSIST_DIRECTORY = os.path.join(kit_dir, 'tests', 'tmp')
TEST_DATA_PATH = os.path.join(kit_dir, 'tests', 'data')
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
TEST_CASE_FILE_PATH = os.path.join(TEST_DATA_PATH, 'test_case.yaml')


class DCTestCase(unittest.TestCase):
    time_start: float
    sambanova_api_key: str
    document_analyzer: DocumentAnalyzer
    test_case: Dict[str, str]
    document1_text: str
    document2_text: str
    user_input: str
    prompt_messages: List[List[str]]
    llm_output: str
    llm_usage: str

    @classmethod
    def setUpClass(cls: Type['DCTestCase']) -> None:
        if not os.path.exists(PERSIST_DIRECTORY):
            os.makedirs(PERSIST_DIRECTORY)
        cls.time_start = time.time()
        cls.sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY', '')
        cls.document_analyzer = DocumentAnalyzer(sambanova_api_key=cls.sambanova_api_key)
        with open(TEST_CASE_FILE_PATH, 'r') as yaml_file:
            cls.test_case = yaml.safe_load(yaml_file)
        cls.document1_text = cls.parse_document('document1')
        cls.document2_text = cls.parse_document('document2')
        cls.user_input = cls.test_case['user_input']
        cls.prompt_messages = cls.document_analyzer.generate_prompt_messages(
            cls.user_input,
            'document1',
            cls.document1_text,
            'document2',
            cls.document2_text,
        )
        cls.llm_output, cls.llm_usage = cls.document_analyzer.get_analysis(cls.prompt_messages)

    @classmethod
    def parse_document(cls: Type['DCTestCase'], document_name: str = 'document1') -> str:
        if document_name + '_file_name' in cls.test_case:
            file_name = cls.test_case[document_name + '_file_name']
            with open(os.path.join(TEST_DATA_PATH, file_name), 'rb') as doc_file:
                doc = BytesIO(doc_file.read())
                doc.name = file_name
                return cls.document_analyzer.parse_document(doc, document_name)
        else:
            return cls.test_case[document_name + '_text']

    # Add assertions
    def test_document_parsing(self) -> None:
        token_count_1 = self.document_analyzer.get_token_count(self.document1_text)
        token_count_2 = self.document_analyzer.get_token_count(self.document2_text)
        self.assertGreaterEqual(token_count_1, 1, 'Document 1 should parse to at least 1 token')
        self.assertGreaterEqual(token_count_2, 1, 'Document 2 should parse to at least 1 token')

    def test_folder_deletion(self) -> None:
        results = os.listdir(PERSIST_DIRECTORY)
        self.assertEqual(len(results), 0, f'tmp folder should be empty. It contains the following: {results}')

    def test_prompt_generation(self) -> None:
        self.assertGreaterEqual(
            len(self.prompt_messages), 3, 'Generated prompt messages should contain at least 3 entries'
        )

    def test_llm_response(self) -> None:
        self.assertEqual(
            self.test_case['expected_output'].strip(),
            self.llm_output.strip(),
            f"LLM output '{self.llm_output.strip()}' does not match expected output\
                 '{self.test_case['expected_output'].strip()}'",
        )

    def test_llm_usage(self) -> None:
        self.assertIn('completion_tokens_per_sec', self.llm_usage, 'LLM usage does not completion_tokens_per_sec')
        self.assertIn('time_to_first_token', self.llm_usage, 'LLM usage does not time_to_first_token')
        self.assertIn('completion_tokens', self.llm_usage, 'LLM usage does not completion_tokens')

    @classmethod
    def tearDownClass(cls: Type['DCTestCase']) -> None:
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
    suite = unittest.TestLoader().loadTestsFromTestCase(DCTestCase)
    test_result = unittest.TextTestRunner(resultclass=CustomTextTestResult).run(suite)
    # test_result = unittest.TextTestRunner().run(suite)

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
