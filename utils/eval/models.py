import os
import sys
from typing import Any, Dict, Optional

import weave
from langchain_core.output_parsers import JsonOutputParser
from langchain_sambanova import ChatSambaNova
from weave import Model
from weave.flow.scorer import Scorer

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)

from utils.eval.eval_utils import calculate_cost
from utils.eval.prompts.judge_prompt import JUDGE_PROMPT
from utils.eval.prompts.system_prompt import SYSTEM_PROMPT
from utils.eval.rag import RAGChain
from utils.eval.schemas import EmbeddingsSchema, SNCloudSchema, VectorDBSchema


class CorrectnessLLMJudge(Scorer):
    """
    A judge class for evaluating the correctness of model outputs.

    This class is responsible for scoring the generated outputs of a model
    against expected answers using a chat model API. It encapsulates the
    configuration parameters needed to initialize the model.

    Attributes:
        model_name (str): The specific name of the model to be used.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens to generate.
        top_p (Optional[float]): Nucleus sampling parameter (default is 0.1).
        streaming (bool): Whether to use streaming (default is False).
        include_usage (Optional[bool]): Flag to include usage information (default is False).
        model_kwargs (Optional[Dict[str, Any]]): Additional model-specific parameters.
    """

    model_name: str
    temperature: float
    max_tokens: int
    top_p: Optional[float] = 0.1
    streaming: bool = False
    include_usage: Optional[bool] = False
    normalize_score: int = 1
    model_kwargs: Optional[Dict[str, Any]] = None

    @weave.op()
    async def score(
        self,
        model_output: Dict[str, Any],
        query: str,
        expected_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Score the model output against a query and an expected answer.

        This method generates a prompt based on the query and the model's generated answer,
        then invokes the judge model to evaluate the quality of the generated answer.

        Args:
            model_output (Dict[str, Any]): The output from the model containing the generated answer.
            query (str): The original query that was posed to the model.
            expected_answer (Optional[str]): The expected answer for comparison (if available).

        Returns:
            Dict[str, Any]: A dictionary containing the score and any additional information,
            such as the reason for the score.

        Raises:
            Exception: If there is an error during the invocation of the judge model.
        """
        generated_answer = model_output.get('completion', None)
        context = model_output.get('context', None)
        if generated_answer is None:
            return {'score': -1, 'reason': f'Completion not found:\n{model_output}'}

        judge_prompt = JUDGE_PROMPT.format(
            query=query, generated_answer=generated_answer, context=context, expected_answer=expected_answer
        )

        llm = ChatSambaNova(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            streaming=self.streaming,
            stream_options={'include_usage': True},
        )

        judge = llm | JsonOutputParser()

        try:
            result = judge.invoke([('system', judge_prompt)])
            result['answer_score'] = result['answer_score'] / self.normalize_score
            if result.get('context_score'):
                result['context_score'] = result['context_score'] / self.normalize_score
        except Exception as e:
            return {'score': -1, 'reason': f'Completion not completed:\n{e}'}

        if 'usage' in model_output and self.include_usage:
            result.update(model_output['usage'])
        return result


class WeaveChatModel(Model):
    """
    A class for generating chat responses.

    This class encapsulates the configuration required to generate responses
    from a chat model API. It provides methods to interact with the model and
    retrieve generated outputs.

    Attributes:
        model_name (str): The specific name of the model to be used.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens to generate.
        top_p (Optional[float]): Nucleus sampling parameter (default is 0.1).
        streaming (bool): Whether to use streaming (default is False).
        include_usage (Optional[bool]): Flag to include usage information (default is False).
        model_kwargs (Optional[Dict[str, Any]]): Additional model-specific parameters.
    """

    model_name: str
    temperature: float
    max_tokens: int
    top_p: Optional[float] = 0.1
    streaming: bool = False
    include_usage: Optional[bool] = False
    model_kwargs: Optional[Dict[str, Any]] = None

    @weave.op()
    async def predict(self, query: str, system_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response for a given query and system message.

        This method invokes the chat model to produce a response based on the
        provided query and system message. It handles any exceptions and
        returns the generated output along with usage metadata.

        Args:
            query (str): The user query to be processed.
            system_message (str): The system message providing context for the model.

        Returns:
            Dict[str, Any]: A dictionary containing the generated completion and
            any usage information.

        Raises:
            Exception: If there is an error during the invocation of the model.
        """
        client = ChatSambaNova(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            streaming=self.streaming,
            stream_options={'include_usage': True},
        )

        if system_message is None:
            system_message = SYSTEM_PROMPT

        try:
            messages = [
                ('system', system_message),
                ('user', query),
            ]

            response = client.invoke(messages)
            completion = response.content.strip()
            usage = response.response_metadata.get('usage', None)
            input_tokens, output_tokens = usage.get('prompt_tokens'), usage.get('completion_tokens')
            if self.model_kwargs:
                input_token_cost, ouput_token_cost = (
                    self.model_kwargs.get('input_token_cost'),
                    self.model_kwargs.get('ouput_token_cost'),
                )
                usage['cost'] = calculate_cost(input_tokens, output_tokens, input_token_cost, ouput_token_cost)
        except Exception as e:
            completion = f'<Error>: {type(e).__name__} - {str(e)}'
            usage = {}
        return {'completion': completion, 'usage': usage}


class WeaveRAGModel(Model):
    """
    A class representing a Weave RAG model.

    Attributes:
        type (str): The type of the model (e.g., 'sncloud').
        model (str): The specific name of the model to be used.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens to generate.
        top_p (Optional[float]): Nucleus sampling parameter (default is 0.1).
        streaming (bool): Whether to use streaming (default is False).
        include_usage (Optional[bool]): Flag to include usage information (default is False).
        model_kwargs (Optional[Dict[str, Any]]): Additional model-specific parameters.
    """

    llm_params: Optional[Dict[str, Any]] = None
    embeddings_params: Optional[Dict[str, Any]] = None
    rag_params: Optional[Dict[str, Any]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    rag_chain: Optional[RAGChain] = None

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self._initialize_rag()

    def _initialize_rag(self) -> None:
        self.rag_chain = RAGChain(
            SNCloudSchema(**self.llm_params),
            EmbeddingsSchema(**self.embeddings_params),
            VectorDBSchema(**self.rag_params),
        )

    def upload_docs(self, path: str) -> None:
        self.rag_chain.upload_docs(path)

    @weave.op()
    async def predict(self, query: str) -> Dict[str, Any]:
        """
        Logs predictions of a rag chain.

        Args:
        completion (str): The input completion.

        Returns:
        Dict[str, Any]: A dictionary containing the completion.

        Raises:
            Exception: If completion not found.
        """

        response = self.rag_chain.predict(query)
        context = [i.page_content for i in response['context']]
        completion = response['response']['content']
        usage = response['response']['metadata']['usage']
        input_tokens, output_tokens = usage.get('prompt_tokens'), usage.get('completion_tokens')
        if self.model_kwargs:
            input_token_cost, ouput_token_cost = (
                self.model_kwargs.get('input_token_cost'),
                self.model_kwargs.get('ouput_token_cost'),
            )
            usage['cost'] = calculate_cost(input_tokens, output_tokens, input_token_cost, ouput_token_cost)

        return {'completion': completion, 'context': context, 'usage': usage}


class WeaveDummyModel(Model):
    """
    A class representing a Weave Dummy model.

    Attributes:
        model_kwargs (Optional[Dict[str, Any]]): model-specific parameters.
    """

    model_kwargs: Optional[Dict[str, Any]] = None

    @weave.op()
    async def predict(self, completion: str, context: Optional[str]) -> Dict[str, Any]:
        """
        Logs predictions of a dummy model.

        Args:
        completion (str): The input completion.

        Returns:
        Dict[str, Any]: A dictionary containing the completion.

        Raises:
            Exception: If completion not found.
        """
        if context is not None:
            return {'completion': completion, 'context': context}
        return {'completion': completion}
