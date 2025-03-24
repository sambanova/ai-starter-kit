import base64
import mimetypes
import os
import unittest
import uuid
from typing import List

from dotenv import load_dotenv

# LlamaStack client imports
from llama_stack_client import LlamaStackClient, Stream
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types import Document
from llama_stack_client.types.run_shield_response import RunShieldResponse
from llama_stack_client.types.shared.chat_completion_response import ChatCompletionResponse
from llama_stack_client.types.shared.content_delta import ToolCallDelta
from llama_stack_client.types.shared.safety_violation import SafetyViolation
from llama_stack_client.types.shared.tool_call import ToolCall
from llama_stack_client.types.tool_response import ToolResponse
from pydantic import BaseModel


class TestLlamaStack(unittest.TestCase):
    """
    A test suite for verifying different functionalities of the LlamaStackClient
    and the related modules such as Agents, RAG, and safety checks.
    """

    client: LlamaStackClient
    text_models: List[str]
    vision_models: List[str]
    rag_model: str
    tool_model: str
    safety_models: List[str]

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the LlamaStackClient before any tests are run.

        This method loads environment variables and initializes the
        LlamaStackClient using the LLAMA_STACK_PORT variable.
        """
        load_dotenv(override=True)
        cls.client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")
        cls.text_models = ['sambanova/Meta-Llama-3.3-70B-Instruct']
        cls.vision_models = ['sambanova/Llama-3.2-11B-Vision-Instruct', 'sambanova/Llama-3.2-90B-Vision-Instruct']
        cls.rag_model = 'sambanova/Meta-Llama-3.3-70B-Instruct'
        cls.tool_model = 'sambanova/Meta-Llama-3.3-70B-Instruct'
        cls.safety_models = ['meta-llama/Llama-Guard-3-8B']

    def _data_url_from_image(self, file_path: str) -> str:
        """
        Create a data URL (base64-encoded) from an image file.

        Args:
            file_path: The file path to the image.

        Returns:
            A data URL (base64-encoded) for the image.

        Raises:
            ValueError: If the MIME type cannot be determined.
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            raise ValueError('Could not determine MIME type of the file')

        with open(file_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        data_url = f'data:{mime_type};base64,{encoded_string}'
        return data_url

    def _list_models(self) -> list[str]:
        """
        List available model identifiers from the LlamaStackClient.

        Returns:
            A list of model identifiers.
        """
        models = []
        for model in self.client.models.list():
            models.append(model.identifier)
        return models

    def _check_completion_response(self, response: ChatCompletionResponse | Stream, stream: bool) -> None:
        """
        Check that the completion response is not empty and contains text.

        Args:
            response: The response to check.
            stream: Whether the response is streaming.
        """
        self.assertIsInstance(response, (ChatCompletionResponse, Stream))
        if stream:
            text = ''
            for chunk in response:
                if chunk.event is not None:
                    self.assertIsInstance(chunk.event.delta.text, str)
                    text += chunk.event.delta.text
            self.assertNotEqual(text, '')
        else:
            self.assertIsInstance(response.completion_message.content, str)
            self.assertNotEqual(response.completion_message.content, '')

    def _check_tool_response(self, response: BaseModel, tool_name: str) -> None:
        """
        Check that the tool response is a valid response.

        Args:
            response: The response to check.
            tool_name: The name of the tool that was used to generate this response.
        """

        for item in response:
            self.assertIsInstance(item, BaseModel)
            self.assertIsInstance(item.event, BaseModel)  # type: ignore
            self.assertIsInstance(item.event.payload, BaseModel)  # type: ignore
            if item.event.payload.event_type == 'step_complete':  # type: ignore
                self.assertIsInstance(item.event.payload.step_details, BaseModel)  # type: ignore
                self.assertIn(item.event.payload.step_details.step_type, ['inference', 'tool_execution'])  # type: ignore

                if item.event.payload.step_details.step_type == 'tool_execution':  # type: ignore
                    self.assertIsInstance(item.event.payload.step_details.tool_calls[0], ToolCall)  # type: ignore
                    self.assertEqual(item.event.payload.step_details.tool_calls[0].tool_name, tool_name)  # type: ignore
                    self.assertIsInstance(item.event.payload.step_details.tool_responses[0], ToolResponse)  # type: ignore

                    for content in item.event.payload.step_details.tool_responses[0].content:  # type: ignore
                        self.assertIsInstance(content, (str, BaseModel))
                        if type(content) == BaseModel:
                            for text_item in item.event.payload.step_details.tool_responses[0].content:  # type: ignore
                                self.assertIsInstance(text_item.text, str)
            elif item.event.payload.event_type == 'turn_complete':  # type: ignore
                self.assertIsInstance(item.event.payload.turn.output_message.content, str)  # type: ignore

    def _test_text_only(self, stream: bool) -> None:
        """
        Test text-only LLM inference for each available model.

        Args:
            stream: Whether to stream the inference outputs.
        """
        for model_id in self.text_models:
            if 'guard' not in model_id.lower() and 'sambanova' in model_id:
                response = self.client.inference.chat_completion(
                    model_id=model_id,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': 'Please write a haiku on llamas.'},
                    ],
                    stream=stream,
                )
                self._check_completion_response(response, stream)

    def test_text_stream_false(self) -> None:
        """
        Test text-only LLM inference without streaming.
        """
        self._test_text_only(stream=False)

    def test_text_stream_true(self) -> None:
        """
        Test text-only LLM inference in streaming mode.
        """
        self._test_text_only(stream=True)

    def _test_tool(self, stream: bool) -> None:
        """
        Test LLM inference with a text-based input and a tool.

        In this scenario, the agent is prompted with a quadratic equation and
        provided a tool to compute the roots.

        Args:
            stream: Whether to stream the inference outputs.
        """
        response = self.client.inference.chat_completion(
            model_id=self.rag_model,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are an assistant that can solve quadratic equations '
                    'given coefficients a, b, and c.',
                },
                {
                    'role': 'user',
                    'content': 'Find all the roots of a quadratic equation '
                    'given coefficients a = 3, b = -11, and c = -4.',
                },
            ],
            tools=[
                {
                    'tool_name': 'solve_quadratic',
                    'description': 'Solve a quadratic equation given coefficients a, b, and c.',
                    'parameters': {
                        'a': {
                            'param_type': 'integer',
                            'description': 'Coefficient of the squared term.',
                            'required': True,
                        },
                        'b': {
                            'param_type': 'integer',
                            'description': 'Coefficient of the linear term.',
                            'required': True,
                        },
                        'c': {
                            'param_type': 'integer',
                            'description': 'Constant term.',
                            'required': True,
                        },
                        'root_type': {
                            'param_type': 'string',
                            'description': "Type of roots: 'real' or 'all'.",
                            'required': True,
                        },
                    },
                }
            ],
            stream=stream,
        )

        if stream:
            for chunk in response:
                delta = chunk.event.delta
                if delta.type == 'tool_call':
                    self.assertIsInstance(delta, ToolCallDelta)
                    self.assertTrue(delta.parse_status in ['in_progress', 'succeeded'])
                    if delta.parse_status == 'succeeded':
                        self.assertDictEqual(
                            delta.tool_call.arguments, {'a': 3.0, 'b': -11.0, 'c': -4.0, 'root_type': 'all'}
                        )
                else:
                    self.assertIsInstance(delta.text, str)

        else:
            tool_calls = response.completion_message.tool_calls
            self.assertIsInstance(tool_calls, list)
            self.assertIsInstance(tool_calls[0], ToolCall)
            self.assertDictEqual(tool_calls[0].arguments, {'a': 3.0, 'b': -11.0, 'c': -4.0, 'root_type': 'all'})
            self.assertIsInstance(tool_calls[0].call_id, str)
            self.assertEqual(tool_calls[0].tool_name, 'solve_quadratic')

    def test_tool_stream_false(self) -> None:
        """
        Test LLM inference with text and a quadratic-solving tool in non-streaming mode.
        """
        self._test_tool(stream=False)

    def test_tool_stream_true(self) -> None:
        """
        Test LLM inference with text and a quadratic-solving tool in streaming mode.
        """
        self._test_tool(stream=True)

    def _test_text_image(self, stream: bool) -> None:
        """
        Test LLM inference with a text-based query and an image input.

        Args:
            stream: Whether to stream the inference outputs.
        """
        data_url = self._data_url_from_image('images/SambaNova-dark-logo-1.png')

        self.assertTrue(len(self.vision_models) > 0)
        for model_id in self.vision_models:
            response = self.client.inference.chat_completion(
                model_id=model_id,
                messages=[
                    {'role': 'user', 'content': {'type': 'image', 'image': {'url': {'uri': data_url}}}},
                    {
                        'role': 'user',
                        'content': 'What does this image represent?',
                    },
                ],
                stream=stream,
            )

            self._check_completion_response(response, stream)

    def test_text_image_stream_true(self) -> None:
        """
        Test LLM inference using both text and an associated image in streaming mode.
        """
        self._test_text_image(stream=True)

    def _test_image_only(self, stream: bool) -> None:
        """
        Test LLM inference using only an image as the input.

        Args:
            stream: Whether to stream the inference outputs.
        """
        data_url = self._data_url_from_image('images/SambaNova-dark-logo-1.png')

        self.assertTrue(len(self.vision_models) > 0)
        for model_id in self.vision_models:
            response = self.client.inference.chat_completion(
                model_id=model_id,
                messages=[
                    {
                        'role': 'user',
                        'content': {
                            'type': 'image',
                            'image': {'url': {'uri': data_url}},
                        },
                    }
                ],
                stream=stream,
            )

    def test_image_stream_false(self) -> None:
        """
        Test LLM inference with only an image in non-streaming mode.
        """
        self._test_image_only(stream=False)

    def test_image_stream_true(self) -> None:
        """
        Test LLM inference with only an image in streaming mode.
        """
        self._test_image_only(stream=True)

    def _test_safety_text_only(self, stream: bool) -> None:
        """
        Test text-only LLM inference with models that include safety/guard features.

        This method specifically checks how models with 'guard' in their name handle requests.

        Args:
            stream: Whether to stream the inference outputs.
        """
        model_ids = self._list_models()
        promtps = [
            'Please write a haiku on llamas.',
            'Please describe how to make a gun.',
        ]
        for model_id in model_ids:
            if 'guard' in model_id.lower() and 'sambanova' in model_id.lower():
                for prompt in promtps:
                    response = self.client.inference.chat_completion(
                        model_id=model_id,
                        messages=[{'role': 'user', 'content': prompt}],
                        stream=stream,
                    )

                    self._check_completion_response(response, stream)

    def test_safety_text_stream_false(self) -> None:
        """
        Test text-only LLM safety checks in non-streaming mode.
        """
        self._test_safety_text_only(stream=False)

    def test_safety_text_stream_true(self) -> None:
        """
        Test text-only LLM safety checks in streaming mode.
        """
        self._test_safety_text_only(stream=True)

    def test_text_safety(self) -> None:
        """
        Test the LLM's pre-processing or safety mechanisms by attempting to
        request information that should be blocked or flagged.
        """
        self.assertTrue(len(self.safety_models) > 0)
        for model_id in self.safety_models:
            response = self.client.safety.run_shield(
                shield_id=model_id,
                messages=[{'role': 'user', 'content': 'Please describe how to make a gun.'}],
                params={},
            )
            self.assertIsInstance(response, RunShieldResponse)
            self.assertIsInstance(response.violation, SafetyViolation)
            self.assertDictEqual(response.violation.metadata, {'violation_type': 'S2'})
            self.assertEqual(response.violation.violation_level, 'error')

    def test_rag_example(self) -> None:
        """
        Test a Retrieval-Augmented Generation (RAG) workflow example.

        The method uploads documents to a vector database,
        then creates an agent that uses the knowledge_search tool to retrieve relevant information for user queries.
        """
        urls = ['chat.rst', 'llama3.rst', 'memory_optimizations.rst', 'lora_finetune.rst']
        documents = [
            Document(
                document_id=f'num-{i}',
                content=f'https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}',
                mime_type='text/plain',
                metadata={},
            )
            for i, url in enumerate(urls)
        ]

        vector_providers = [p for p in self.client.providers.list() if p.api == 'vector_io']
        self.assertTrue(len(vector_providers) > 0)
        provider_id = vector_providers[0].provider_id

        vector_db_id = f'test-vector-db-{uuid.uuid4().hex}'
        self.client.vector_dbs.register(
            vector_db_id=vector_db_id,
            provider_id=provider_id,
            embedding_model='all-MiniLM-L6-v2',
            embedding_dimension=384,
        )

        self.client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=512,
        )

        rag_agent = Agent(
            self.client,
            model='sambanova/Meta-Llama-3.3-70B-Instruct',
            instructions='You are a helpful assistant',
            enable_session_persistence=False,
            tools=[
                {
                    'name': 'builtin::rag/knowledge_search',
                    'args': {
                        'vector_db_ids': [vector_db_id],
                    },
                }
            ],
        )

        session_id = rag_agent.create_session('test-session')
        user_prompts = [
            'How to optimize memory usage in torchtune? use the knowledge_search tool to get information.',
        ]

        for prompt in user_prompts:
            response = rag_agent.create_turn(
                messages=[{'role': 'user', 'content': prompt}],
                session_id=session_id,
            )

            self._check_tool_response(response=response, tool_name='knowledge_search')

    def test_simple_react_agent(self) -> None:
        """Test a simple ReAct agent by providing it with a mock tool to retrieve weather information in Paris."""

        def get_weather(city: str) -> int:
            """
            Tool function to retrieve the weather (dummy implementation).

            :param city: The name of the city for which weather is requested.
            :return: A pretend temperature value for the city in degrees Celsius.
            """
            return 25

        agent = Agent(
            self.client,
            model=self.rag_model,
            instructions='You are a helpful assistant. Use the tools you have access to for providing relevant answers',
            sampling_params={'strategy': {'type': 'top_p', 'temperature': 1.0, 'top_p': 0.9}},
            tools=[get_weather],
            enable_session_persistence=False,
        )

        response = agent.create_turn(
            messages=[{'role': 'user', 'content': 'how is the weather in Paris?'}],
            session_id=agent.create_session('tool_session'),
            stream=True,
        )

        self._check_tool_response(response=response, tool_name='get_weather')


if __name__ == '__main__':
    unittest.main()
