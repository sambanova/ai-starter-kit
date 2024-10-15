#!/usr/bin/env python3
"""
Image Search Test Script

This script tests the functionality of the Image Search kit using unittest.

Test cases:
    test_image_search_creation: checks if the image_search, client, collection and embeddings are not empty
    test_get_images: checks if the images path and images are not empty
    test_search_image_by_text: checks if the response is not empty
    test_search_image_by_image: checks if the response is not empty

Usage:
    python tests/image_search_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import logging
import os
import sys
import time
import unittest
from typing import Any, Dict, List, Tuple, Type

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from image_search.src.image_search import ImageSearch

IMAGE_TEST_DATA_PATH = os.path.join(kit_dir, 'tests/data/')


class ImageSearchTestCase(unittest.TestCase):
    time_start: float
    image_search: ImageSearch
    image_paths: List[str]
    images: List[float]

    @classmethod
    def setUpClass(cls: Type['ImageSearchTestCase']) -> None:
        cls.time_start = time.time()
        cls.image_search = ImageSearch()
        cls.image_paths, cls.images = cls.get_images()
        cls.init_collection()
        cls.add_images()

    @classmethod
    def get_images(cls: Type['ImageSearchTestCase']) -> Tuple[List[str], Any]:
        paths, images = cls.image_search.get_images(IMAGE_TEST_DATA_PATH)
        return paths, images

    @classmethod
    def init_collection(cls: Type['ImageSearchTestCase']) -> None:
        cls.image_search.init_collection()

    @classmethod
    def add_images(cls: Type['ImageSearchTestCase']) -> None:
        cls.image_search.add_images(IMAGE_TEST_DATA_PATH)

    # Add assertions
    def test_image_search_creation(self) -> None:
        self.assertIsNotNone(self.image_search, 'Image Search could not be created')
        self.assertIsNotNone(self.image_search.client, 'Vector store could not be created')
        self.assertIsNotNone(self.image_search.embedding_function, 'Embeddings could not be created')
        self.assertIsNotNone(self.image_search.collection, 'Collection could not be created')

    def test_get_images(self) -> None:
        self.assertGreaterEqual(len(self.image_paths), 1, 'There should be at least one path')
        self.assertGreaterEqual(len(self.images), 1, 'There should be at least one image')

    def test_search_image_by_text(self) -> None:
        user_question = 'Show me a graph'
        image_paths, distances = self.image_search.search_image_by_text(user_question)

        self.assertGreaterEqual(len(image_paths), 1, 'There should be at least one image path')
        self.assertGreaterEqual(len(distances), 1, 'There should be at least one distance result')

    def test_search_image_by_image(self) -> None:
        image_paths, distances = self.image_search.search_image_by_image(
            os.path.join(IMAGE_TEST_DATA_PATH, 'sample.png')
        )

        self.assertGreaterEqual(len(image_paths), 1, 'There should be at least one image path')
        self.assertGreaterEqual(len(distances), 1, 'There should be at least one distance result')

    @classmethod
    def tearDownClass(cls: Type['ImageSearchTestCase']) -> None:
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
    suite = unittest.TestLoader().loadTestsFromTestCase(ImageSearchTestCase)
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
