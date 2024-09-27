#!/usr/bin/env python3
"""
Web Crawling Test Script

This script tests the functionality of the Web Crawling Data Retriever kit using unittest.

Usage:
    python tests/web_crawling_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import logging
import os
import sys
import time
import unittest

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths and global variables
file_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(file_dir, '..'))  # absolute path to kit directory
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))  # absolute path to ai-starter-kit directory

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from typing import Any, Dict, List, Type

from dotenv import load_dotenv
from langchain.docstore.document import Document

from web_crawled_data_retriever.src.web_crawling_retriever import WebCrawlingRetrieval

load_dotenv(os.path.join(repo_dir, '.env'))


def set_retrieval_qa_chain(documents: List[Document] = [], config: Dict[str, Any] = {}, save: bool = False) -> Any:
    web_crawling_retrieval = WebCrawlingRetrieval(documents, config)
    web_crawling_retrieval.init_llm_model()
    if save:
        web_crawling_retrieval.create_and_save_local(
            input_directory=config.get('input_directory', None),
            persist_directory=config.get('persist_directory', None),
            update=config.get('update', False),
        )
    else:
        web_crawling_retrieval.create_load_vector_store(
            force_reload=config.get('force_reload', False), update=config.get('update', False)
        )
    web_crawling_retrieval.retrieval_qa_chain()
    return web_crawling_retrieval


# Scrap sites
base_urls_list = set(
    [
        'https://www.espn.com',
        'https://lilianweng.github.io/posts/2023-06-23-agent/',
        'https://sambanova.ai/',
    ]
)


class WebCrawlingTestCase(unittest.TestCase):
    time_start: float
    crawler: WebCrawlingRetrieval
    docs: List[Document]
    sources: List[str]
    conversation: Any

    @classmethod
    def setUpClass(cls: Type['WebCrawlingTestCase']) -> None:
        cls.time_start = time.time()

        # Scrap sites
        cls.crawler = WebCrawlingRetrieval()
        cls.docs, cls.sources = cls.crawler.web_crawl(base_urls_list, depth=1)

        # Initialize the LLM and embedding, chunk the text, create a vectorstore, and initialize the retrieval qa chain
        config = {'force_reload': True}
        cls.conversation = set_retrieval_qa_chain(cls.docs, config=config)  # WebCrawlingRetrieval

    # Add assertions
    def test_web_parsing(self) -> None:
        logger.info(self.sources)
        self.assertTrue(self.docs, "Parsed docs shouldn't be empty")  # list[str]
        self.assertTrue(self.sources, "URL sources shouldn't be empty")  # list[Document]

    def test_conversation_chain_creation(self) -> None:
        self.assertIsNotNone(self.conversation.qa_chain, "Conversation chain shouldn't be empty")

    def test_question_answering(self) -> None:
        user_question = 'which kinds of memory can an agent have?'
        response = self.conversation.qa_chain.invoke({'question': user_question})
        logger.info(user_question)
        logger.info(response['answer'])

        self.assertIn('answer', response, "Response should have an 'answer' key")
        self.assertTrue(response['answer'], 'The answer should not be empty')

    @classmethod
    def tearDownClass(cls: Type['WebCrawlingTestCase']) -> None:
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
    suite = unittest.TestLoader().loadTestsFromTestCase(WebCrawlingTestCase)
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
