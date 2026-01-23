#!/usr/bin/env python3
"""
API models Test Script

This script tests the functionality all API models using unittest.

Usage:
    python utils/api_testing.py

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
curr_dir = current_dir
repo_dir = os.path.abspath(os.path.join(curr_dir, '..'))
logger.info(f'kit_dir: {curr_dir}')
logger.info(f'repo_dir: {repo_dir}')

sys.path.append(curr_dir)
sys.path.append(repo_dir)
os.path.join(repo_dir, 'utils', 'tests', 'config.yaml')

import openai
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_sambanova import ChatSambaNova
from openai import OpenAI

from utils.tests.utils_test import (
    # audio_requests,
    image_to_base64,
    load_encode_audio,
)

load_dotenv()

CONFIG_PATH = os.path.join(os.getcwd(), 'tests', 'config.yaml')

with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

client_base_url = config['urls']['base_client_url']

text_models = config['models']['text']
image_models = config['models']['text-image']
audio_models = config['models']['text-audio']

image_path = 'tests/data/sample.png'
base64_image = image_to_base64(image_path)

audio_path = 'tests/data/samplerecord.mp3'
base64_audio = load_encode_audio(audio_path)

text_prompt = 'Translate this sentence from English to French: I love programming.'
image_prompt = 'What is in this image?'
audio_prompt = 'what is the previous audio about?'


class TestAPIModel(unittest.TestCase):
    time_start: float
    client: OpenAI
    sambanova_api_key: str

    @classmethod
    def setUpClass(cls: Type['TestAPIModel']) -> None:
        cls.time_start = time.time()
        cls.sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY', '')
        cls.client = cls.setup_client()

    @classmethod
    def setup_client(cls: Type['TestAPIModel']) -> OpenAI:
        client = OpenAI(
            base_url=client_base_url,
            api_key=cls.sambanova_api_key,
        )
        return client

    def test_client_chat_completion_text(self) -> None:
        for model in text_models:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': text_prompt}],
                stream=False,
            )

            self.assertTrue(hasattr(response, 'id'))
            self.assertTrue(hasattr(response, 'choices'))
            self.assertIsInstance(response.choices[0].message.content, str)
            self.assertGreater(len(response.choices[0].message.content), 0)
            self.assertTrue(hasattr(response, 'model'))
            self.assertIn(response.model, [model, f'{model}-Text'])
            self.assertTrue(hasattr(response, 'usage'))

    def test_client_chat_completion_text_streaming(self) -> None:
        for model in text_models:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': text_prompt}],
                stream=True,
            )

            for chunk in response:
                self.assertTrue(hasattr(chunk, 'id'))
                self.assertTrue(hasattr(chunk, 'choices'))
                self.assertIsInstance(chunk.choices[0].delta.content, str)
                self.assertTrue(hasattr(chunk, 'model'))
                self.assertIn(chunk.model, [model, f'{model}-Text'])
                self.assertTrue(hasattr(chunk, 'usage'))

    def test_client_chat_completion_image(self) -> None:
        for model in image_models:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': image_prompt,
                            },
                            {
                                'type': 'image_url',
                                'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'},
                            },
                        ],
                    }
                ],
                stream=False,
            )

            self.assertTrue(hasattr(response, 'id'))
            self.assertTrue(hasattr(response, 'choices'))
            self.assertIsInstance(response.choices[0].message.content, str)
            self.assertGreater(len(response.choices[0].message.content), 0)
            self.assertTrue(hasattr(response, 'model'))
            self.assertEqual(response.model, model)
            self.assertTrue(hasattr(response, 'usage'))

    # def test_client_chat_completion_audio(self) -> None:
    #     for model in audio_models:
    #         response = self.client.chat.completions.create(
    #             model=model,
    #             messages=[
    #                 {
    #                     'role': 'user',
    #                     'content': [
    #                         {
    #                             'type': 'text',
    #                             'text': audio_prompt,
    #                         },
    #                         {
    #                             'type': 'audio_content',
    #                             'audio_content': {
    #                                 'content': f'data:audio/mp3;base64,{load_encode_audio(audio_path)}'
    #                             },
    #                         },
    #                     ],
    #                 }
    #             ],
    #             stream=False,
    #         )

    #         self.assertTrue(hasattr(response, 'id'))
    #         self.assertTrue(hasattr(response, 'choices'))
    #         self.assertIsInstance(response.choices[0].message.content, str)
    #         self.assertGreater(len(response.choices[0].message.content), 0)
    #         self.assertTrue(hasattr(response, 'model'))
    #         self.assertEqual(response.model, model)
    #         self.assertTrue(hasattr(response, 'token_usage'))

    # def test_request_audio(self) -> None:
    #     for model in audio_models:
    #         for url in ['transcription_url', 'translation_url']:
    #             response = audio_requests(
    #                 config['urls'][url], self.sambanova_api_key, file_path=audio_path, model=model
    #             )
    #             self.assertIn('text', response)
    #             self.assertIsInstance(response.get('text'), str)
    #             self.assertGreater(len(response.get('text')), 0)

    def test_client_bad_model(self) -> None:
        model = 'parrot'
        with self.assertRaises(openai.NotFoundError) as context:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': text_prompt}],
                stream=False,
            )
            logger.info(f'Bad request: {response}')
        self.assertIn("Error code: 404 - {'error': 'Model not found'}", str(context.exception))

    def test_langchain_chat_completion_text(self) -> None:
        messages = [
            ('human', text_prompt),
        ]
        for model in text_models:
            if model == 'Meta-Llama-Guard-3-8B':
                continue

            llm_model = ChatSambaNova(api_key=self.sambanova_api_key, model=model)
            response = llm_model.invoke(messages)

            self.assertTrue(hasattr(response, 'content'))
            self.assertIsInstance(response.content, str)
            self.assertGreater(len(response.content), 0)
            self.assertTrue(hasattr(response, 'response_metadata'))
            self.assertIn('token_usage', response.response_metadata)

    def test_langchain_chat_completion_image(self) -> None:
        message = [
            HumanMessage(
                content=[
                    {'type': 'text', 'text': image_prompt},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'},
                    },
                ],
            )
        ]
        for model in image_models:
            llm_model = ChatSambaNova(api_key=self.sambanova_api_key, model=model)
            response = llm_model.invoke(message)

            self.assertTrue(hasattr(response, 'content'))
            self.assertIsInstance(response.content, str)
            self.assertGreater(len(response.content), 0)
            self.assertTrue(hasattr(response, 'response_metadata'))
            self.assertIn('token_usage', response.response_metadata)

    # def test_langchain_chat_completion_audio(self) -> None:
    #     messages = [
    #         HumanMessage(
    #             content=[
    #                 {'type': 'audio_content', 'audio_content': {'content': f'data:audio/mp3;base64,{base64_audio}'}}
    #             ]
    #         ),
    #         HumanMessage(audio_prompt),
    #     ]
    #     for model in audio_models:
    #         llm_model = ChatSambaNova(api_key=self.sambanova_api_key, model=model)
    #         response = llm_model.invoke(messages)

    #         self.assertTrue(hasattr(response, 'content'))
    #         self.assertIsInstance(response.content, str)
    #         self.assertGreater(len(response.content), 0)
    #         self.assertTrue(hasattr(response, 'response_metadata'))
    #         self.assertIn('token_usage', response.response_metadata)

    @classmethod
    def tearDownClass(cls: Type['TestAPIModel']) -> None:
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
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAPIModel)
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
