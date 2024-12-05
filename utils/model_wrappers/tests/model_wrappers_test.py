#!/usr/bin/env python3
"""
Model Wrappers (MW) Test Script

This script tests the functionality of the Model Wrappers using unittest.

Usage:
    python tests/model_wrappers_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import logging
import os
import sys

# from time import sleep
import time
import unittest
from typing import Any, Dict, List, Tuple, Type

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths and global variables
current_dir = os.getcwd()
kit_dir = current_dir
repo_dir = os.path.abspath(os.path.join(kit_dir, '../..'))  # absolute path for ai-starter-kit root repo

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import yaml
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from utils.model_wrappers.api_gateway import APIGateway
from utils.model_wrappers.langchain_chat_models import ChatSambaNovaCloud, ChatSambaStudio
from utils.model_wrappers.langchain_llms import SambaNovaCloud, SambaStudio
from utils.model_wrappers.tests.schemas import (
    EmbeddingsBaseModel,
    LLMBaseModel,
    SambaStudioGenericV1Response,
    SambaStudioGenericV2Response,
    SambaStudioOpenAIResponse,
    SambaStudioOpenAIResponseMetadata,
    SNCloudBaseResponse,
    SNCloudChatCompletionChunk,
)
from utils.model_wrappers.tests.utils_test import add, multiply

load_dotenv(os.path.join(repo_dir, '.env'), override=True)


def load_test_config(config_path: str) -> Any:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


test_config = load_test_config(kit_dir + '/tests/test_config.yaml')
bad_format_embeddings_model_params = test_config['bad_format_embeddings_model']
bad_format_llm_model_params = test_config['bad_format_llm_model']
bad_format_chat_model_params = test_config['bad_format_chat_model']

interface = APIGateway


class ModelWrapperTestCase(unittest.TestCase):
    time_start: float
    embed_params: Dict[str, Any]
    sn_llm_params: Dict[str, Any]
    sn_llm_params_output: Dict[str, Any]
    ss_llm_params: Dict[str, Any]
    ss_llm_params_output: Dict[str, Any]
    sn_chat_model_params: Dict[str, Any]
    sn_chat_model_params_output: Dict[str, Any]
    ss_chat_model_params: Dict[str, Any]
    ss_chat_model_params_output: Dict[str, Any]
    sn_llm_model: LLM
    ss_llm_model: LLM
    sn_chat_model: BaseChatModel
    ss_chat_model: BaseChatModel
    embeddings_model: Embeddings

    @classmethod
    def setUpClass(cls: Type['ModelWrapperTestCase']) -> None:
        cls.time_start = time.time()
        (
            cls.embed_params,
            cls.sn_llm_params,
            cls.sn_llm_params_output,
            cls.ss_llm_params,
            cls.ss_llm_params_output,
            cls.sn_chat_model_params,
            cls.sn_chat_model_params_output,
            cls.ss_chat_model_params,
            cls.ss_chat_model_params_output,
        ) = cls.get_params()
        cls.embeddings_model = cls.init_embeddings_model()
        cls.sn_llm_model, cls.ss_llm_model = cls.init_llm_models()
        cls.sn_chat_model, cls.ss_chat_model = cls.init_chat_models()

    @classmethod
    def get_params(
        cls: Type['ModelWrapperTestCase'],
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        embed_params = test_config['embedding_model']

        sn_llm_params = test_config['sn_llm']
        sn_llm_params_output = sn_llm_params.copy()
        sn_llm_params_output['_type'] = 'sambanovacloud-llm'

        ss_llm_params = test_config['sambastudio_llm']
        ss_llm_params_output = sn_llm_params.copy()
        ss_llm_params_output['_type'] = 'sambastudio-llm'

        sn_chat_model_params = test_config['sn_chat_model']
        sn_chat_model_params_output = sn_chat_model_params.copy()
        sn_chat_model_params_output['_type'] = 'sambanovacloud-chatmodel'

        ss_chat_model_params = test_config['sambastudio_chat_model']
        ss_chat_model_params_output = ss_chat_model_params.copy()
        ss_chat_model_params_output['_type'] = 'sambastudio-chatmodel'

        return (
            embed_params,
            sn_llm_params,
            sn_llm_params_output,
            ss_llm_params,
            ss_llm_params_output,
            sn_chat_model_params,
            sn_chat_model_params_output,
            ss_chat_model_params,
            ss_chat_model_params_output,
        )

    @classmethod
    def init_embeddings_model(cls: Type['ModelWrapperTestCase']) -> Embeddings:
        embeddings_model = interface.load_embedding_model(**cls.embed_params)
        return embeddings_model

    @classmethod
    def init_llm_models(cls: Type['ModelWrapperTestCase']) -> Tuple[LLM, LLM]:
        sn_llm_model = SambaNovaCloud(**cls.sn_llm_params)
        ss_llm_model = SambaStudio(**cls.ss_llm_params)

        return sn_llm_model, ss_llm_model

    @classmethod
    def init_chat_models(cls: Type['ModelWrapperTestCase']) -> Tuple[BaseChatModel, BaseChatModel]:
        sn_chat_model = ChatSambaNovaCloud(**cls.sn_chat_model_params)
        ss_chat_model = ChatSambaStudio(**cls.ss_chat_model_params)

        return sn_chat_model, ss_chat_model

    def test_embeddings_model_creation(self) -> None:
        self.assertIsNotNone(self.embeddings_model, 'embeddings model class could not be created')

    def test_llm_model_creation(self) -> None:
        self.assertIsNotNone(self.sn_llm_model, 'llm model class could not be created')
        self.assertIsNotNone(self.ss_llm_model, 'llm model class could not be created')

    def test_chat_model_creation(self) -> None:
        self.assertIsNotNone(self.sn_chat_model, 'chat model class could not be created')
        self.assertIsNotNone(self.ss_chat_model, 'chat model class could not be created')

    def test_llm_model_params(self) -> None:
        for i in self.sn_llm_params_output:
            params = self.sn_llm_model.dict()
            self.assertIn(i, params, f'{i} should be a param of llm class')
            self.assertEqual(self.sn_llm_params_output.get(i), params.get(i), 'llm model params not equal')

    def test_chat_model_params(self) -> None:
        self.assertEqual(self.sn_chat_model.dict(), self.sn_chat_model_params_output, 'chat model params not equal')
        self.assertEqual(self.ss_chat_model.dict(), self.ss_chat_model_params_output, 'chat model params not equal')

    def test_embeddings_model_validation(self) -> None:
        for i in bad_format_embeddings_model_params.get('type'):
            with self.assertRaises(ValueError) as context:
                interface.load_embedding_model(**i)
            self.assertIn('string is not a valid embedding model type', str(context.exception))

        for i in bad_format_embeddings_model_params.get('string'):
            with self.assertRaises(ValidationError) as context:
                EmbeddingsBaseModel(**i)
            self.assertIn('Input should be a valid string', str(context.exception))

        for i in bad_format_embeddings_model_params.get('boolean'):
            with self.assertRaises(ValidationError) as context:
                EmbeddingsBaseModel(**i)
            self.assertIn('Input should be a valid boolean', str(context.exception))

        for i in bad_format_embeddings_model_params.get('integer'):
            with self.assertRaises(ValidationError) as context:
                EmbeddingsBaseModel(**i)
            self.assertIn('Input should be a valid integer', str(context.exception))

    def test_llm_model_validation(self) -> None:
        for i in bad_format_llm_model_params.get('string'):
            with self.assertRaises(ValidationError) as context:
                LLMBaseModel(**i)
            self.assertIn('Input should be a valid string', str(context.exception))

        for i in bad_format_llm_model_params.get('boolean'):
            with self.assertRaises(ValidationError) as context:
                LLMBaseModel(**i)
            self.assertIn('Input should be a valid boolean', str(context.exception))

        for i in bad_format_llm_model_params.get('integer'):
            with self.assertRaises(ValidationError) as context:
                LLMBaseModel(**i)
            self.assertIn('Input should be a valid integer', str(context.exception))

        for i in bad_format_llm_model_params.get('number'):
            with self.assertRaises(ValidationError) as context:
                LLMBaseModel(**i)
            self.assertIn('Input should be a valid number', str(context.exception))

    def test_chat_model_validation(self) -> None:
        for i in bad_format_chat_model_params.get('string'):
            with self.assertRaises(ValidationError) as context:
                ChatSambaNovaCloud(**i)
                ChatSambaStudio(**i)
            self.assertIn('Input should be a valid string', str(context.exception))

        for i in bad_format_chat_model_params.get('boolean'):
            with self.assertRaises(ValidationError) as context:
                ChatSambaNovaCloud(**i)
                ChatSambaStudio(**i)
            self.assertIn('Input should be a valid boolean', str(context.exception))

        for i in bad_format_chat_model_params.get('integer'):
            with self.assertRaises(ValidationError) as context:
                ChatSambaNovaCloud(**i)
                ChatSambaStudio(**i)
            self.assertIn('Input should be a valid integer', str(context.exception))

        for i in bad_format_chat_model_params.get('number'):
            with self.assertRaises(ValidationError) as context:
                ChatSambaStudio(**i)
            self.assertIn('Input should be a valid number', str(context.exception))

        for i in bad_format_chat_model_params.get('sambastudio_boolean'):
            with self.assertRaises(ValidationError) as context:
                ChatSambaNovaCloud(**i)
                ChatSambaStudio(**i)
            self.assertIn('Input should be a valid boolean', str(context.exception))

        for i in bad_format_chat_model_params.get('sambastudio_dict'):
            with self.assertRaises(ValidationError) as context:
                ChatSambaStudio(**i)
            self.assertIn('Input should be a valid dictionary', str(context.exception))

    def test_embeddings_model_response(self) -> None:
        query = 'What is computer science?'
        queries = ['tell me a 50 word tale', 'tell me a joke']
        response = self.embeddings_model.embed_query(query)
        multiple_response = self.embeddings_model.embed_documents(queries)
        self.assertEqual(len(response), 4096, 'Embeddings dimension should be 4096')
        self.assertEqual(len(multiple_response), 2, 'Response length should be 2')

    def test_llm_model_response(self) -> None:
        query = 'How many moons does Jupiter have?'
        sn_response = self.sn_llm_model.invoke(query)
        ss_response = self.ss_llm_model.invoke(query)
        sn_raw_response = self.sn_llm_model._handle_request(query).json()

        try:
            SNCloudBaseResponse(**sn_raw_response)
        except ValidationError as e:
            self.fail(f'Schema validation failed: {e}')

        self.assertGreaterEqual(len(sn_response), 1, 'Response should be a non-empty string')
        self.assertGreaterEqual(len(ss_response), 1, 'Response should be a non-empty string')

    def test_chat_model_response(self) -> None:
        query = 'Where is alpha centauri located?'
        format_query = [{'role': 'user', 'content': query}]
        messages = [
            HumanMessage(content=query),
        ]

        sn_cloud_response = self.sn_chat_model.invoke(query)
        ss_response = self.ss_chat_model.invoke(query)
        sn_cloud_raw_response = self.sn_chat_model._handle_request(format_query).json()
        ss_raw_response = self.ss_chat_model._handle_request(messages)
        ss_raw_response_dict = ss_raw_response.json()

        if 'chat/completions' in self.ss_chat_model.sambastudio_url:
            try:
                content = ss_raw_response_dict['choices'][0]['message']['content']
                id = ss_raw_response_dict['id']
                response_metadata = SambaStudioOpenAIResponseMetadata(
                    finish_reason=ss_raw_response_dict['choices'][0]['finish_reason'],
                    usage=ss_raw_response_dict.get('usage'),
                    model_name=ss_raw_response_dict['model'],
                    system_fingerprint=ss_raw_response_dict['system_fingerprint'],
                    created=ss_raw_response_dict['created'],
                )
                SambaStudioOpenAIResponse(content=content, id=id, response_metadata=response_metadata)
            except ValidationError as e:
                self.fail(f'Invalid SambaStudioOpenAI response: {e}')

        elif 'api/v2/predict/generic' in self.ss_chat_model.sambastudio_url:
            try:
                validated_response = SambaStudioGenericV2Response(items=ss_raw_response_dict['items'])
                content = validated_response.items[0].value['completion']
                id = validated_response.items[0].id
                response_metadata = validated_response.items[0].dict()
                return {'content': content, 'id': id, 'response_metadata': response_metadata}
            except ValidationError as e:
                self.fail(f'Invalid SambaStudio Generic V2 response: {e}')

        elif 'api/predict/generic' in self.ss_chat_model.sambastudio_url:
            try:
                validated_response = SambaStudioGenericV1Response(predictions=ss_raw_response_dict['predictions'])
                content = validated_response.predictions[0]['completion']
                id = None
                response_metadata = ss_raw_response_dict
                return {'content': content, 'id': id, 'response_metadata': response_metadata}
            except Exception as e:
                self.fail(f'Invalid SambaStudio Generic V1 response: {e}')

        try:
            SNCloudBaseResponse(**sn_cloud_raw_response)
        except ValidationError as e:
            self.fail(f'Schema validation failed: {e}')

        sn_cloud_response_dict = sn_cloud_response.model_dump()
        ss_response_dict = ss_response.model_dump()

        self.assertIn('content', sn_cloud_response_dict, "Response should have a 'content' key")
        self.assertGreaterEqual(len(sn_cloud_response_dict['content']), 1, 'Content should be a non-empty string')
        self.assertIn('response_metadata', sn_cloud_response_dict, "Response should have a 'response_metadata' key")
        self.assertIn(
            'usage', sn_cloud_response_dict['response_metadata'], "Response metadata should have a 'usage' key"
        )

        self.assertIn('content', ss_response_dict, "Response should have a 'content' key")
        self.assertGreaterEqual(len(ss_response_dict['content']), 1, 'Content should be a non-empty string')
        self.assertIn('response_metadata', ss_response_dict, "Response should have a 'response_metadata' key")
        self.assertIn('usage', ss_response_dict['response_metadata'], "Response metadata should have a 'usage' key")

    def test_function_calling_response(self) -> None:
        tools = [add, multiply]
        tools_data = {
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_time',
                        'description': 'Returns current date, current time or both.\n\n Args:\n  kind: date, time or\
                             both',
                        'parameters': {'properties': {'kind': {'default': 'both', 'type': 'string'}}, 'type': 'object'},
                    },
                }
            ],
            'tool_choice': 'auto',
            'parallel_tool_calls': False,
        }
        query = 'whats time is it?'
        format_query = [{'role': 'user', 'content': query}]
        raw_response = self.sn_chat_model._handle_request(format_query, **tools_data)

        llm = ChatSambaNovaCloud(model='Meta-Llama-3.1-70B-Instruct', max_tokens=1024, temperature=0.0)
        llm_with_tools = llm.bind_tools(tools)
        query_2 = 'What is 3 * 12? Also, what is 11 + 49?'
        response = llm_with_tools.invoke(query_2).tool_calls

        for res in response:
            self.assertIn('name', res, "Response 'tool_calls' should have a 'name' key")
            self.assertIn('args', res, "Response 'tool_calls' should have a 'args' key")
            self.assertIn('id', res, "Response 'tool_calls' should have a 'id' key")
            self.assertIn('type', res, "Response 'tool_calls' should have a 'type' key")

        try:
            SNCloudBaseResponse(**raw_response.json())
        except ValidationError as e:
            self.fail(f'Schema validation failed: {e}')

    def test_chat_model_response_streaming(self) -> None:
        query = 'Where is alpha centauri located?'
        format_query = [{'role': 'user', 'content': query}]

        raw_response = self.sn_chat_model._handle_request(format_query)

        for chunk in self.sn_chat_model._process_stream_response(raw_response):
            try:
                SNCloudChatCompletionChunk(**chunk)
            except ValidationError as e:
                self.fail(f'Schema validation failed: {e}')

        for chunk in self.sn_chat_model.stream(query):
            self.assertIn('content', chunk.model_dump(), "Response should have a 'content' attribute")
            self.assertTrue(isinstance(chunk.content, str), 'Content should be a string')

        for chunk in self.ss_chat_model.stream(query):
            self.assertIn('content', chunk.model_dump(), "Response should have a 'content' attribute")
            self.assertTrue(isinstance(chunk.content, str), 'Content should be a string')

    @classmethod
    def tearDownClass(cls: Type['ModelWrapperTestCase']) -> None:
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
    suite = unittest.TestLoader().loadTestsFromTestCase(ModelWrapperTestCase)
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
