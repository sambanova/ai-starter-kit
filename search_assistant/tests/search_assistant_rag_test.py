#!/usr/bin/env python3
"""
Search Assistant (RAG Query) Test Script

This script tests the functionality of the Search Assistant kit (search and scrape sites) using unittest.

Usage:
    python tests/search_assistant_rag_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import os
import sys
import unittest
import logging
import time

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

method = 'rag_query'
input_disabled = False
tool = ['serpapi'] # serpapi, serper, openserp
search_engine = 'google' # google, bing, baidu
max_results = 5
query = 'Albert Einstein'

class SearchAssistantRAGTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_start = time.time()

        cls.search_assistant = SearchAssistant()
        cls.scraper_state = cls.search_assistant.search_and_scrape(
                            query=query,
                            search_method=tool[0],
                            max_results=max_results,
                            search_engine=search_engine,
                        )

    # Add assertions
    def test_search_assistant_class_creation(self):
        self.assertIsNotNone(self.search_assistant, "SearchAssistant class shouldn't be empty")
    
    def test_search_and_scrape(self):
        self.assertIsNotNone(self.search_assistant.vector_store, "Vector store shouldn't be empty")

    def test_retrieval_call(self):
        user_question = 'who is Albert Einsten?'
        response = self.search_assistant.retrieval_call(user_question)

        logger.info(user_question)
        logger.info(response["source_documents"]) # list[Document]
        logger.info(response["answer"]) # str

        self.assertIn('source_documents', response, "Response should have a 'source_documents' key")
        self.assertGreaterEqual(len(response["source_documents"]), 1, "There should be at least one source documents")
        self.assertIn('answer', response, "Response should have an 'answer' key")
        self.assertIsNotNone(response["answer"], "LLM answer shouldn't be empty")

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
    suite = unittest.TestLoader().loadTestsFromTestCase(SearchAssistantRAGTestCase)
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


