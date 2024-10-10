import functools
from langchain_core.language_models.llms import LLM
from llama_index.llms import LangChainLLM
import logging
import os
import sys
import shutil
import time
from typing import Any, Dict, List, Tuple, Type
import unittest
import yaml # type: ignore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Main directories
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
logger.info(f'kit_dir: {kit_dir}')
logger.info(f'repo_dir: {repo_dir}')
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.model_wrappers.api_gateway import APIGateway # type: ignore
from fine_tuning_embeddings.src.generate_fine_tune_embed_dataset import CorpusLoader, QueryGenerator, save_dict_safely # type: ignore
from fine_tuning_embeddings.src.finetune_embedding_model import DatasetLoader

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
DATA_DIRECTORY = os.path.join(kit_dir, 'sample_data')
OUTPUT_PATH = os.path.join(kit_dir, "data")
TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'train_corpus.json')
VAL_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'val_corpus.json')

with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
llm_info = config['llm']
api = config['api']

logger.info(config)


def timing_decorator(func: Any):
    @functools.wraps(func)
    def wrapper(*args: tuple, **kwargs: dict[str, Any]):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f'Function {func.__name__} took {end - start:.2g} seconds to run.')
        return result
    return wrapper


class GenerateEmbeddingDataTestCase(unittest.TestCase):
    time_start: float
    corpus_loader: CorpusLoader
    train_corpus: Dict[str, str]
    val_corpus: Dict[str, str]
    query_generator: QueryGenerator
    langchain_llm: LangChainLLM
    train_queries: Dict[str, str]
    train_relevant_docs: Dict[str, List[str]]
    val_queries: Dict[str, str]
    val_relevant_docs: Dict[str, List[str]]
    train_dataset: Dict[str, object]
    val_dataset: Dict[str, object]
    train_dataset_loader: DatasetLoader
    val_dataset_loader: DatasetLoader
    

    @classmethod
    def setUpClass(cls: Type['GenerateEmbeddingDataTestCase']) -> None:
        cls.time_start = time.time()
        cls.corpus_loader = CorpusLoader(directory=DATA_DIRECTORY, val_ratio=0.2)
        cls.train_corpus = cls.corpus_loader.load_corpus(cls.corpus_loader.train_files)
        cls.val_corpus = cls.corpus_loader.load_corpus(cls.corpus_loader.val_files)
        cls.corpus_loader.save_corpus(cls.train_corpus, TRAIN_OUTPUT_PATH)
        cls.corpus_loader.save_corpus(cls.val_corpus, VAL_OUTPUT_PATH)
        cls.langchain_llm = cls.initialize_llm()
        cls.query_generator = QueryGenerator(cls.langchain_llm)
        cls.train_queries, cls.train_relevant_docs = cls.query_generator.generate_queries(cls.train_corpus, 
                                                                                          verbose=True)
        cls.val_queries, cls.val_relevant_docs = cls.query_generator.generate_queries(cls.val_corpus, 
                                                                                          verbose=True)
        cls.train_dataset, cls.val_dataset = cls.create_dataset()
        cls.save_dataset()
        cls.train_dataset_loader = DatasetLoader(dataset_path=TRAIN_OUTPUT_PATH)
        cls.val_dataset_loader = DatasetLoader(dataset_path=VAL_OUTPUT_PATH)
        

    @classmethod
    def initialize_llm(cls: Type['GenerateEmbeddingDataTestCase']) -> LangChainLLM:

        llm: LLM = APIGateway.load_llm(
            type=api,
            streaming=True,
            coe=llm_info['coe'],
            do_sample=llm_info['do_sample'],
            max_tokens_to_generate=llm_info['max_tokens_to_generate'],
            temperature=llm_info['temperature'],
            select_expert=llm_info['select_expert'],
            process_prompt=False,
        )
        langchain_llm = LangChainLLM(llm=llm)

        return langchain_llm
    
    @classmethod
    def create_dataset(cls: Type['GenerateEmbeddingDataTestCase']) -> Tuple[Dict[str, object], Dict[str, object]]:

        train_dataset = {'queries': cls.train_queries, 
                         'corpus': cls.train_corpus, 
                         'relevant_docs': cls.train_relevant_docs}
        val_dataset = {'queries': cls.val_queries, 
                       'corpus': cls.val_corpus, 
                       'relevant_docs': cls.val_relevant_docs}
        
        return train_dataset, val_dataset
    
    @classmethod
    def save_dataset(cls: Type['GenerateEmbeddingDataTestCase']) -> None:

        save_dict_safely(cls.train_dataset, TRAIN_OUTPUT_PATH)
        save_dict_safely(cls.val_dataset, VAL_OUTPUT_PATH)

    # Assertions
    @timing_decorator
    def test_sample_data(self) -> None:
        self.assertTrue(os.path.isdir(DATA_DIRECTORY), f'{DATA_DIRECTORY} does not exist.')

    @timing_decorator
    def test_train_corpus(self) -> None:
        self.assertGreater(len(self.train_corpus), 0, 'the train corpus should not be empty.')

    @timing_decorator
    def test_val_corpus(self) -> None:
        self.assertGreater(len(self.val_corpus), 0, 'the train corpus should not be empty.')

    @timing_decorator
    def test_query_generator_creation(self) -> None:
        self.assertIsNotNone(self.query_generator, 'query generator could not be created.')

    @timing_decorator
    def test_train_queries_generated(self) -> None:
        self.assertNotEqual(self.train_queries, {}, f'{self.train_queries} should not be empty')

    @timing_decorator
    def test_val_queries_generated(self) -> None:
        self.assertNotEqual(self.val_queries, {}, f'{self.val_queries} should not be empty')

    @timing_decorator
    def test_train_docs_generated(self) -> None:
        self.assertNotEqual(self.train_relevant_docs, {}, f'{self.train_relevant_docs} should not be empty')

    @timing_decorator
    def test_val_docs_generated(self) -> None:
        self.assertNotEqual(self.val_relevant_docs, {}, f'{self.val_relevant_docs} should not be empty')

    @timing_decorator
    def test_train_dataset_generated(self) -> None:
        self.assertNotEqual(self.train_dataset, {}, f'{self.train_dataset} should not be empty')

    @timing_decorator
    def test_val_dataset_generated(self) -> None:
        self.assertNotEqual(self.val_dataset, {}, f'{self.val_dataset} should not be empty')

    @timing_decorator
    def test_save_train_corpus(self) -> None:
        self.assertTrue(os.path.isfile(TRAIN_OUTPUT_PATH), f'{TRAIN_OUTPUT_PATH} does not exist.')

    @timing_decorator
    def test_save_val_corpus(self) -> None:
        self.assertTrue(os.path.isfile(VAL_OUTPUT_PATH), f'{VAL_OUTPUT_PATH} does not exist.')

    @timing_decorator
    def test_train_dataset_loader_corpus(self) -> None:
        self.assertNotEqual(self.train_dataset_loader.corpus, {}, f'{self.train_dataset_loader.corpus} should not' +
                            'be empty')
    
    @timing_decorator
    def test_val_dataset_loader_corpus(self) -> None:
        self.assertNotEqual(self.val_dataset_loader.corpus, {}, f'{self.val_dataset_loader.corpus} should not' +
                            'be empty')

    @classmethod
    def tearDownClass(cls: Type['GenerateEmbeddingDataTestCase']) -> None:
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

    
    def main_suite() -> unittest.TestSuite:
        """Test suite to define the order of the test execution."""

        # List all the test cases here in order of execution
        suite_list = [
            'test_sample_data',
            'test_train_corpus',
            'test_val_corpus',
            'test_query_generator_creation',
            'test_train_queries_generated',
            'test_val_queries_generated',
            'test_train_docs_generated',
            'test_val_docs_generated',
            'test_train_dataset_generated',
            'test_val_dataset_generated',
            'test_save_train_corpus',
            'test_save_val_corpus',
            'test_train_dataset_loader_corpus',
            'test_val_dataset_loader_corpus'
        ]

        # Add all the tests to the suite
        suite = unittest.TestSuite()
        for suite_item in suite_list:
            suite.addTest(GenerateEmbeddingDataTestCase(suite_item))

        return suite


    def suite_github_pull_request() -> unittest.TestSuite:
        """Test suite for GitHub actions on `pull_request`."""

        # List all the test cases here in order of execution
        suite_list = ['test_train_corpus']

        # Add all the tests to the suite
        suite = unittest.TestSuite()
        for suite_item in suite_list:
            suite.addTest(GenerateEmbeddingDataTestCase(suite_item))

        return suite


    # Suite registry for the tests
    suite_registry = {
        'main': main_suite(),
        'github_pull_request': suite_github_pull_request(),
    }

def main() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(GenerateEmbeddingDataTestCase)
    try:
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
    finally:
        try:
            logging.info(f'removing {OUTPUT_PATH}')
            shutil.rmtree(OUTPUT_PATH)
            logging.info(f'{OUTPUT_PATH} removed')
        except OSError as e:
            logging.error(f'Error: {e.filename} - {e.strerror}')

if __name__ == '__main__':
    sys.exit(main())