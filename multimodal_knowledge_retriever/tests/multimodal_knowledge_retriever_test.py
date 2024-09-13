#!/usr/bin/env python3
"""
Multimodal Knowledge Retriever Test Script

This script tests the functionality of the Multimodal Knowledge Retriever kit using unittest.

Test cases:
    test_document_parsing: checks if the images path, chunks of texts and tables are not empty
    test_vector_store_creation: checks if the vectorstore is not None
    test_conversation_chain_creation: checks if the conversation chain is not None
    test_question_answering: checks if the answer and source_documents exist and are not empty

Usage:
    python tests/multimodal_knowledge_retriever_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import logging
import os
import sys
import time
import unittest
from types import TracebackType
from typing import Any, Callable, Dict, List, Tuple, Type

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from langchain.retrievers.multi_vector import MultiVectorRetriever

from multimodal_knowledge_retriever.src.multimodal import MultimodalRetrieval

IMAGE_TEST_DATA_PATH = os.path.join(kit_dir, 'data/sample_docs')
PDF_TEST_DATA = os.path.join(IMAGE_TEST_DATA_PATH, 'invoicesample.pdf')


class MKRTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls: Type['MKRTestCase']) -> None:
        cls.time_start = time.time()
        cls.multimodal_retriever = MultimodalRetrieval()
        cls.text_docs, cls.table_docs, cls.image_paths = cls.parse_data()
        cls.vectorstore = cls.create_vectorstore()
        cls.chain = cls.create_retrieval_chain()

    @classmethod
    def parse_data(cls: Type['MKRTestCase']) -> Tuple[List, List, List]:
        raw_pdf_elements, _ = cls.multimodal_retriever.extract_pdf(PDF_TEST_DATA)
        text_docs, table_docs, image_paths = cls.multimodal_retriever.process_raw_elements(
            raw_pdf_elements, [IMAGE_TEST_DATA_PATH]
        )
        logger.info(f'Number of chunks of text: {len(text_docs)}')
        logger.info(f'Number of chunks of table: {len(table_docs)}')
        logger.info(f'Number of images: {len(image_paths)}')
        return text_docs, table_docs, image_paths

    @classmethod
    def create_vectorstore(cls: Type['MKRTestCase']) -> MultiVectorRetriever:
        retriever = cls.multimodal_retriever.create_vectorstore()
        return cls.multimodal_retriever.vectorstore_ingest(
            retriever, cls.text_docs, cls.table_docs, cls.image_paths, summarize_texts=True, summarize_tables=True
        )

    @classmethod
    def create_retrieval_chain(cls: Type['MKRTestCase']) -> Callable:
        return cls.multimodal_retriever.get_retrieval_chain(cls.vectorstore, image_retrieval_type='summary')

    # Add assertions
    def test_document_parsing(self) -> None:
        self.assertTrue(len(self.text_docs) > 0, 'There should be at least one parsed chunk of text')
        self.assertTrue(len(self.table_docs) > 0, 'There should be at least one parsed chunk of table text')
        self.assertTrue(len(self.image_paths) > 0, 'There should be at least one image')

    def test_vector_store_creation(self) -> None:
        self.assertIsNotNone(self.vectorstore, 'Vector store could not be created')

    def test_conversation_chain_creation(self) -> None:
        self.assertIsNotNone(self.chain, 'Conversation chain could not be created')

    def test_question_answering(self) -> None:
        user_question = 'How many apples they bought?'
        response = self.chain(user_question)

        self.assertIn('source_documents', response, "Response should have a 'source_documents' key")
        self.assertGreaterEqual(len(response['source_documents']), 1, 'There should be at least one source chunk')
        self.assertIn('answer', response, "Response should have an 'answer' key")
        self.assertTrue(response['answer'], 'The response should not be empty')

    @classmethod
    def tearDownClass(cls: Type['MKRTestCase']) -> None:
        time_end = time.time()
        total_time = time_end - cls.time_start
        logger.info(f'Total execution time: {total_time:.2f} seconds')


class CustomTextTestResult(unittest.TextTestResult):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.test_results: List[Dict[str, Any]] = []

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self.test_results.append({'name': test._testMethodName, 'status': 'PASSED'})

    def addFailure(
        self, test: unittest.TestCase, err: Tuple[Type[BaseException], BaseException, TracebackType]
    ) -> None:
        super().addFailure(test, err)
        self.test_results.append({'name': test._testMethodName, 'status': 'FAILED', 'message': str(err[1])})

    def addError(self, test: unittest.TestCase, err: Tuple[Type[BaseException], BaseException, TracebackType]) -> None:
        super().addError(test, err)
        self.test_results.append({'name': test._testMethodName, 'status': 'ERROR', 'message': str(err[1])})


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(MKRTestCase)
    test_result = unittest.TextTestRunner(resultclass=CustomTextTestResult).run(suite)

    logger.info('\nTest Results:')
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
