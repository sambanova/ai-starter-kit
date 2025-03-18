import base64
import mimetypes
import os
import unittest
import uuid
from typing import List

from dotenv import load_dotenv

# LlamaStack client imports
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import Document
from termcolor import cprint


class TestLlamaStack(unittest.TestCase):
    """
    A test suite for verifying different functionalities of the LlamaStackClient
    and the related modules such as Agents, RAG, and safety checks.
    """

    client: LlamaStackClient
    allowed_models: List[str]

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the LlamaStackClient before any tests are run.

        This method loads environment variables and initializes the
        LlamaStackClient using the LLAMA_STACK_PORT variable.
        """
        load_dotenv(override=True)
        cls.client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")
        cls.allowed_models = ['sambanova/Meta-Llama-3.2-3B-Instruct']

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
            list[str]: A list of model identifiers.
        """
        models = []
        for model in self.client.models.list():
            models.append(model.identifier)
        return [model for model in self.allowed_models]

    def _test_inference_llm_text_only(self, stream: bool) -> None:
        """
        Test text-only LLM inference for each available model.

        For each model, the method sends a Chat Completion request with a sample
        message asking for a haiku on llamas.

        Args:
            stream: Whether to stream the inference outputs.
        """
        model_ids = self._list_models()
        print('========== Inference: Text Only ==========')
        self.assertTrue(len(model_ids) > 0)
        for model_id in model_ids:
            if 'guard' not in model_id.lower() and 'sambanova' in model_id:
                print(f'>>>>> Sending request to {model_id}')
                iterator = self.client.inference.chat_completion(
                    model_id=model_id,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': 'Write a haiku on llamas'},
                    ],
                    stream=stream,
                )
                if stream:
                    print('<<<<< Streaming Response')
                    text = ''
                    for chunk in iterator:
                        print(f'{chunk.event.delta.text}', end='', flush=True)
                        text += chunk.event.delta.text
                    self.assertNotEqual(text, '')
                    print()
                else:
                    print('<<<<< Non-streaming Response')
                    print(
                        f'Type: {type(iterator.completion_message.content)}, '
                        'Value:{iterator.completion_message.content}'
                    )
                    self.assertNotEqual(iterator.completion_message.content, '')

    def test_inference_llm_text_only_stream_false(self) -> None:
        """
        Test text-only LLM inference without streaming.
        """
        self._test_inference_llm_text_only(stream=False)

    def test_inference_llm_text_only_stream_true(self) -> None:
        """
        Test text-only LLM inference in streaming mode.
        """
        self._test_inference_llm_text_only(stream=True)

    def _test_inference_llm_text_tool(self, stream: bool) -> None:
        """
        Test LLM inference with a text-based input and a tool.

        In this scenario, the agent is prompted with a quadratic equation and
        provided a tool to compute the roots.

        Args:
            stream: Whether to stream the inference outputs.
        """
        model_ids = [
            # 'sambanova/Meta-Llama-3.1-70B-Instruct',
            # 'sambanova/Meta-Llama-3.1-405B-Instruct',
            'sambanova/Meta-Llama-3.3-70B-Instruct',
        ]
        print('========== Inference: Text and Tool ==========')
        self.assertTrue(len(model_ids) > 0)
        for model_id in model_ids:
            print(f'>>>>> Sending request to {model_id}')
            iterator = self.client.inference.chat_completion(
                model_id=model_id,
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
                print('<<<<< Streaming Response')
                for chunk in iterator:
                    delta = chunk.event.delta
                    if delta.type == 'tool_call':
                        print(delta)
                    else:
                        print(delta.text)
            else:
                print('<<<<< Non-streaming Response')
                tool_calls = iterator.completion_message.tool_calls
                print(tool_calls)
            print()

    def test_inference_llm_text_tool_stream_false(self) -> None:
        """
        Test LLM inference with text and a quadratic-solving tool in non-streaming mode.
        """
        self._test_inference_llm_text_tool(stream=False)

    def test_inference_llm_text_tool_stream_true(self) -> None:
        """
        Test LLM inference with text and a quadratic-solving tool in streaming mode.
        """
        self._test_inference_llm_text_tool(stream=True)

    def _test_inference_llm_text_image(self, stream: bool) -> None:
        """
        Test LLM inference with a text-based query and an image input.

        Args:
            stream: Whether to stream the inference outputs.
        """
        model_ids = ['sambanova/Llama-3.2-11B-Vision-Instruct', 'sambanova/Llama-3.2-90B-Vision-Instruct']
        data_url = self._data_url_from_image('images/SambaNova-dark-logo-1.png')

        print('========== Inference: Text and Image ==========')
        self.assertTrue(len(model_ids) > 0)
        for model_id in model_ids:
            print(f'>>>>> Sending request to {model_id}')
            iterator = self.client.inference.chat_completion(
                model_id=model_id,
                messages=[
                    {'role': 'user', 'content': {'type': 'image', 'image': {'url': {'uri': data_url}}}},
                    {
                        'role': 'user',
                        'content': 'How many different colors are in this image?',
                    },
                ],
                stream=stream,
            )

            if stream:
                print('<<<<< Streaming Response')
                text = ''
                for chunk in iterator:
                    if chunk.event is not None:
                        print(f'{chunk.event.delta.text}', end='', flush=True)
                        text += chunk.event.delta.text
                # self.assertNotEqual(text, '')
                print()
            else:
                print('<<<<< Non-streaming Response')
                print(f'Type: {type(iterator.completion_message.content)}, Value:{iterator.completion_message.content}')
                self.assertNotEqual(iterator.completion_message.content, '')
            print()

    def test_inference_llm_text_image_stream_true(self) -> None:
        """
        Test LLM inference using both text and an associated image in streaming mode.
        """
        self._test_inference_llm_text_image(stream=True)

    def _test_inference_llm_image_only(self, stream: bool) -> None:
        """
        Test LLM inference using only an image as the input.

        Args:
            stream (bool): Whether to stream the inference outputs.
        """
        model_ids = ['sambanova/Llama-3.2-11B-Vision-Instruct', 'sambanova/Llama-3.2-11B-Vision-Instruct']
        data_url = self._data_url_from_image('images/SambaNova-dark-logo-1.png')

        print('========== Inference: Text and Image ==========')
        self.assertTrue(len(model_ids) > 0)
        for model_id in model_ids:
            print(f'>>>>> Sending request to {model_id}')
            iterator = self.client.inference.chat_completion(
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

            if stream:
                print('<<<<< Streaming Response')
                text = ''
                for chunk in iterator:
                    if chunk.event is not None:
                        print(f'{chunk.event.delta.text}', end='', flush=True)
                        text += chunk.event.delta.text
                self.assertNotEqual(text, '')
                print()
            else:
                print('<<<<< Non-streaming Response')
                print(f'Type: {type(iterator.completion_message.content)}, Value:{iterator.completion_message.content}')
                self.assertNotEqual(iterator.completion_message.content, '')
            print()

    def test_inference_llm_image_only_stream_false(self) -> None:
        """
        Test LLM inference with only an image in non-streaming mode.
        """
        self._test_inference_llm_image_only(stream=False)

    def test_inference_llm_image_only_stream_true(self) -> None:
        """
        Test LLM inference with only an image in streaming mode.
        """
        self._test_inference_llm_image_only(stream=True)

    def _test_inference_safety_text_only(self, stream: bool) -> None:
        """
        Test text-only LLM inference with models that include safety/guard features.

        This method specifically checks how models with 'guard' in their name
        handle requests such as writing a haiku on llamas.

        Args:
            stream (bool): Whether to stream the inference outputs.
        """
        model_ids = self._list_models()
        print('========== Safety on Inference: Text Only ==========')
        self.assertTrue(len(model_ids) > 0)
        for model_id in model_ids:
            if 'guard' in model_id.lower() and 'sambanova' in model_id.lower():
                print(f'>>>>> Sending request to {model_id}')
                iterator = self.client.inference.chat_completion(
                    model_id=model_id,
                    messages=[{'role': 'user', 'content': 'Write a haiku on llamas'}],
                    stream=stream,
                )

                if stream:
                    print('<<<<< Streaming Response')
                    text = ''
                    for chunk in iterator:
                        if chunk.event is not None:
                            print(f'{chunk.event.delta.text}', end='', flush=True)
                            text += chunk.event.delta.text
                    self.assertNotEqual(text, '')
                    print()
                else:
                    print('<<<<< Non-streaming Response')
                    print(
                        f'Type: {type(iterator.completion_message.content)}, '
                        'Value:{iterator.completion_message.content}'
                    )
                    self.assertNotEqual(iterator.completion_message.content, '')
                print()

    def test_inference_safety_text_only_stream_false(self) -> None:
        """
        Test text-only LLM safety checks in non-streaming mode.
        """
        self._test_inference_safety_text_only(stream=False)

    def test_inference_safety_text_only_stream_true(self) -> None:
        """
        Test text-only LLM safety checks in streaming mode.
        """
        self._test_inference_safety_text_only(stream=True)

    def test_text_safety(self) -> None:
        """
        Test the LLM's pre-processing or safety mechanisms by attempting to
        request information that should be blocked or flagged.
        """
        model_ids = ['sambanova/Meta-Llama-Guard-3-8B']
        print('========== Safety:Text Only ==========')
        self.assertTrue(len(model_ids) > 0)
        for model_id in model_ids:
            print(f'>>>>> Sending request to {model_id}')
            iterator = self.client.safety.run_shield(
                shield_id=model_id,
                messages=[{'role': 'user', 'content': 'how to make a gun'}],
                params={},
            )
            print(' Response')
            print(iterator)

    def test_rag_example(self) -> None:
        """
        Test a Retrieval-Augmented Generation (RAG) workflow example.

        The method uploads documents to a vector database, then creates an agent
        that uses the knowledge_search tool to retrieve relevant information for
        user queries.
        """
        print('========== RAG Example ==========')
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
        print(documents)

        vector_providers = [p for p in self.client.providers.list() if p.api == 'vector_io']
        print(vector_providers)
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
            cprint(f'User> {prompt}', 'green')
            response = rag_agent.create_turn(
                messages=[{'role': 'user', 'content': prompt}],
                session_id=session_id,
            )
            for log in EventLogger().log(response):
                log.print()

    def test_simple_react_agent(self) -> None:
        """
        Test a simple ReAct-styled agent by providing it with a tool to retrieve
        weather information, then prompting for the weather in Paris.
        """
        print('========== Simple React Agent ==========')

        def get_weather(city: str) -> int:
            """
            Tool function to retrieve the weather (dummy implementation).

            :param city: The name of the city for which weather is requested.
            :return: A pretend temperature value for the city in degrees Celsius.
            """
            return 26

        agent = Agent(
            self.client,
            model='sambanova/Meta-Llama-3.1-70B-Instruct',
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

        for event in response:
            print(event)
            if event.event.payload.event_type == 'turn_complete':
                print('#####')
                print(event.event.payload.turn.output_message.content)


if __name__ == '__main__':
    unittest.main()
