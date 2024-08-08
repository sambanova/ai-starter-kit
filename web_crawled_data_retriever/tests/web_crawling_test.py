#!/usr/bin/env python3
"""
Web Crawling Test Script

This script tests the functionality of the Web Crawling Data Retriever using unittest.

Usage:
    python tests/web_crawling_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import os
import sys
import yaml
import unittest
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths and global variables
current_dir = os.getcwd()
kit_dir = current_dir
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from src.web_crawling_retriever import WebCrawlingRetrieval
from dotenv import load_dotenv

load_dotenv(os.path.join(repo_dir,'.env'))
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

def get_config_info():
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        web_crawling_params = config["web_crawling"]
        
        return  web_crawling_params

def set_retrieval_qa_chain(documents=None, config=None, save=False):
    if config is None:
        config = {}
    if documents is None:
        documents = []
    web_crawling_retrieval = WebCrawlingRetrieval(documents, config)
    web_crawling_retrieval.init_llm_model()
    if save:
        web_crawling_retrieval.create_and_save_local(input_directory=config.get("input_directory",None) , persist_directory=config.get("persist_directory"), update=config.get("update",False))
    else:
        web_crawling_retrieval.create_load_vector_store(force_reload=config.get("force_reload",False), update = config.get("update",False))
    web_crawling_retrieval.retrieval_qa_chain()
    return web_crawling_retrieval

# Read config
web_crawling_params = get_config_info()

# Scrap sites
filtered_sites = ["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "telagram.me", "reddit.com", "whatsapp.com", "wa.me"]
base_urls_list=["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/", "https://sambanova.ai/"]

class WebCrawlingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_start = time.time()
        
        # Scrap sites
        cls.crawler = WebCrawlingRetrieval()
        cls.docs, cls.sources = cls.crawler.web_crawl(base_urls_list, depth=1)
        print(cls.sources) # list[str] (urls)
        #print(docs) # list[Document], each Document has page_content and metadata

        # Initialize the LLM and embedding, chunk the text, create a vectorstore, and initialize the retrieval chain
        db_path = None
        config = config ={"force_reload":True}
        cls.conversation = set_retrieval_qa_chain(cls.docs, config=config) # WebCrawlingRetrieval

    # Add assertions
    def test_question_answering(self):
        user_question = "which kinds of memory can an agent have?"
        response = self.conversation.qa_chain.invoke({"question": user_question})
        print(response["answer"])

        self.assertIn('answer', response, "Response should have an 'answer' key")
        self.assertTrue(response['answer'], "The answer should not be empty")

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
    suite = unittest.TestLoader().loadTestsFromTestCase(WebCrawlingTestCase)
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






