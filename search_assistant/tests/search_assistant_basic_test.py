#!/usr/bin/env python3
"""
Search Assistant (Basic Query) Test Script

This script tests the 'search and answer' functionality of the Search Assistant kit using unittest.

Usage:
    python tests/search_assistant_basic_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import os
import sys
import unittest
import logging
import time
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

from search_assistant.src.search_assistant import SearchAssistant

def load_test_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

test_config = load_test_config(kit_dir+'/tests/test_config.yaml')
tool = test_config['tool'] 
search_engine = test_config['search_engine'] 
max_results = test_config['max_results']

class SearchAssistantBasicTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_start = time.time()

        cls.search_assistant = SearchAssistant()
    
    # Add assertions
    def test_search_assistant_class_creation(self):
        self.assertIsNotNone(self.search_assistant, "SearchAssistant class could not be created")
    
    def test_basic_call(self):
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
        
        logger.info(user_question)
        logger.info(response["sources"]) # list[str]
        logger.info(response["answer"]) # str

        self.assertIn('sources', response, "Response should have a 'sources' key")
        self.assertGreaterEqual(len(response["sources"]), 1, "There should be at least one source link")
        self.assertIn('answer', response, "Response should have an 'answer' key")
        self.assertTrue(response["answer"], "LLM answer shouldn't be empty")

    @classmethod
    def tearDownClass(cls):
        time_end = time.time()
        total_time = time_end - cls.time_start
        logger.info(f"Total execution time: {total_time:.2f} seconds")   

class CustomTextTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_results: List[Dict[str, Any]] = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_results.append({"name": test._testMethodName, "status": "PASSED"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.test_results.append({"name": test._testMethodName, "status": "FAILED", "message": str(err[1])})

    def addError(self, test, err):
        super().addError(test, err)
        self.test_results.append({"name": test._testMethodName, "status": "ERROR", "message": str(err[1])})

def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(SearchAssistantBasicTestCase)
    test_result = unittest.TextTestRunner(resultclass=CustomTextTestResult).run(suite)

    logger.info("\nTest Results:")
    for result in test_result.test_results:
        logger.info(f"{result['name']}: {result['status']}")
        if 'message' in result:
            logger.info(f"  Message: {result['message']}")

    failed_tests = len(test_result.failures) + len(test_result.errors)
    logger.info(f"\nTests passed: {test_result.testsRun - failed_tests}/{test_result.testsRun}")

    if failed_tests:
        logger.error(f"Number of failed tests: {failed_tests}")
        return failed_tests
    else:
        logger.info("All tests passed successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
