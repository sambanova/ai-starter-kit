#!/usr/bin/env python3
"""
Enterprise Knowledge Retriever (EKR) Test Script

This script tests the functionality and quality of the Enterprise Knowledge Retriever using unittest.
It parses documents using the SambaParse service, creates a vector store, and tests the question-answering capabilities.

Usage:
    python tests/ekr_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import asyncio
import logging
import os
import shutil
import sys
import time
import unittest
from typing import Any, Dict, List, Type

import weave
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

from langchain_classic.docstore.document import Document
from langchain_core.embeddings import Embeddings

from enterprise_knowledge_retriever.src.document_retrieval import DocumentRetrieval, RetrievalQAChain
from utils.eval.dataset import WeaveDatasetManager
from utils.eval.models import CorrectnessLLMJudge, WeaveDummyModel

PERSIST_DIRECTORY = os.path.join(kit_dir, 'tests', 'vectordata', 'my-vector-db')
TEST_DATA_PATH = os.path.join(kit_dir, 'tests', 'data', 'test')
CONFIG_PATH = os.path.join(repo_dir, 'utils', 'eval', 'config.yaml')

with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

judge_info = config['eval_llm']

rag_info = config['rag']['llm']


# Let's use this as a template for further CLI tests. setup, tests, teardown and assert at the end.
class EKRTestCase(unittest.TestCase):
    time_start: float
    sambanova_api_key: str
    judge: CorrectnessLLMJudge
    rag_model: WeaveDummyModel
    data_manager: WeaveDatasetManager
    dataset: List[Dict[str, str]]
    document_retrieval: DocumentRetrieval
    additional_metadata: Dict[str, Any]
    text_chunks: List[Document]
    embeddings: Embeddings
    vectorstore: Any
    conversation: RetrievalQAChain

    @classmethod
    def setUpClass(cls: Type['EKRTestCase']) -> None:
        cls.time_start = time.time()
        cls.sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY', '')
        cls.judge = CorrectnessLLMJudge(**judge_info)
        cls.rag_model = WeaveDummyModel(model_kwargs=rag_info)
        cls.data_manager = WeaveDatasetManager()
        cls.dataset = cls.get_data()
        cls.document_retrieval = DocumentRetrieval(sambanova_api_key=cls.sambanova_api_key)
        cls.additional_metadata = {}
        cls.text_chunks = cls.parse_documents()
        cls.embeddings = cls.document_retrieval.load_embedding_model()
        cls.vectorstore = cls.create_vectorstore()
        cls.conversation = cls.create_conversation_chain()

    @classmethod
    def get_data(cls: Type['EKRTestCase']) -> List[Dict[str, str]]:
        data = cls.data_manager.create_raw_dataset(os.path.join(kit_dir, 'tests', 'data', 'rag_data.csv'))
        return data

    @classmethod
    def parse_documents(cls: Type['EKRTestCase']) -> List[Document]:
        text_chunks = cls.document_retrieval.parse_doc(doc_folder=TEST_DATA_PATH)
        logger.info(f'Number of chunks: {len(text_chunks)}')
        return text_chunks

    @classmethod
    def create_vectorstore(cls: Type['EKRTestCase']) -> Any:
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            logger.info('The directory Chroma has been deleted.')
        return cls.document_retrieval.create_vector_store(cls.text_chunks, cls.embeddings, output_db=PERSIST_DIRECTORY)

    @classmethod
    def create_conversation_chain(cls: Type['EKRTestCase']) -> RetrievalQAChain:
        cls.document_retrieval.init_retriever(cls.vectorstore)
        return cls.document_retrieval.get_qa_retrieval_chain()

    # Add assertions
    def test_document_parsing(self) -> None:
        self.assertGreaterEqual(len(self.text_chunks), 1, 'There should be at least one parsed chunk')

    def test_vector_store_creation(self) -> None:
        self.assertIsNotNone(self.vectorstore, 'Vector store could not be created')

    def test_conversation_chain_creation(self) -> None:
        self.assertIsNotNone(self.conversation, 'Conversation chain could not be created')

    def test_question_answering(self) -> None:
        user_question = 'What is a composition of experts?'
        response = self.conversation.invoke({'question': user_question})

        self.assertIn('source_documents', response, "Response should have a 'source_documents' key")
        self.assertGreaterEqual(len(response['source_documents']), 1, 'There should be at least one source document')
        self.assertIn('answer', response, "Response should have an 'answer' key")
        self.assertTrue(response['answer'], 'The response should not be empty')

    def test_quality_context_answer(self) -> None:
        for i in range(len(self.dataset)):
            response = self.conversation.invoke({'question': self.dataset[i].get('query', '')})
            self.dataset[i]['context'] = response.get('source_documents', '')
            self.dataset[i]['completion'] = response.get('answer', '')

        evaluation = weave.Evaluation( 
            name=' '.join(str(value) for value in judge_info.values()), dataset=self.dataset, scorers=[self.judge] # type: ignore[arg-type]
        )

        evaluation_results = asyncio.run(evaluation.evaluate(self.rag_model))
        judge_results = evaluation_results.get('CorrectnessLLMJudge')
        answer_results = judge_results.get('answer_score')
        context_results = judge_results.get('context_score')

        self.assertGreaterEqual(answer_results.get('mean'), 0.6, 'Quality of the answer should be greater than 0.7')
        self.assertGreaterEqual(context_results.get('mean'), 0.7, 'Quality of the context should be greater than 0.7')

    @classmethod
    def tearDownClass(cls: Type['EKRTestCase']) -> None:
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
    suite = unittest.TestLoader().loadTestsFromTestCase(EKRTestCase)
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
