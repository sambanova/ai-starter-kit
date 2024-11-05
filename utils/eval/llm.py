import os
import sys

from langchain_core.output_parsers import JsonOutputParser

from utils.visual.env_utils import get_wandb_key

wandb_api_key = get_wandb_key()
if wandb_api_key:
    import weave
else:
    print('WANDB_API_KEY is not set. Weave initialization skipped.')
from typing import Any, Dict, Optional

from weave import Model
from weave.flow.scorer import Scorer

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)

from utils.eval.prompts.judge_prompt import JUDGE_PROMPT
from utils.model_wrappers.api_gateway import APIGateway


class CorrectnessLLMJudge(Scorer):
    """
    A judge class for evaluating the correctness of model outputs.

    This class is responsible for scoring the generated outputs of a model
    against expected answers using a chat model API. It encapsulates the
    configuration parameters needed to initialize the model.

    Attributes:
        model_type (str): The type of the model (e.g., 'sncloud').
        model_name (str): The specific name of the model to be used.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens to generate.
        top_k (Optional[int]): Number of top tokens to consider (default is 10).
        top_p (Optional[float]): Nucleus sampling parameter (default is 0.1).
        streaming (bool): Whether to use streaming (default is False).
        include_usage (Optional[bool]): Flag to include usage information (default is False).
        model_kwargs (Optional[Dict[str, Any]]): Additional model-specific parameters.
    """

    model_type: str
    model_name: str
    temperature: float
    max_tokens: int
    top_k: Optional[int] = 10
    top_p: Optional[float] = 0.1
    streaming: bool = False
    include_usage: Optional[bool] = False
    model_kwargs: Optional[Dict[str, Any]] = None

    @weave.op()
    async def score(
        self, model_output: Dict[str, Any], query: str, expected_answer: Optional[str] = None
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
        if generated_answer is None:
            return {'score': -1, 'reason': f'Completion not found:\n{model_output}'}

        judge_prompt = JUDGE_PROMPT.format(
            query=query, generated_answer=generated_answer, expected_answer=expected_answer
        )

        llm = APIGateway.load_chat(
            type=self.model_type,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            stream_options={'include_usage': True},
        )

        # fix_parser = OutputFixingParser.from_llm(parser=JsonOutputParser(), llm=llm)

        judge = (
            llm | JsonOutputParser()
            # | RunnableLambda(lambda x: fix_parser.parse(x))
        )

        try:
            result = judge.invoke([('system', judge_prompt)])
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
        model_type (str): The type of the model (e.g., 'sncloud').
        model_name (str): The specific name of the model to be used.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens to generate.
        top_k (Optional[int]): Number of top tokens to consider (default is 10).
        top_p (Optional[float]): Nucleus sampling parameter (default is 0.1).
        streaming (bool): Whether to use streaming (default is False).
        include_usage (Optional[bool]): Flag to include usage information (default is False).
        model_kwargs (Optional[Dict[str, Any]]): Additional model-specific parameters.
    """

    model_type: str
    model_name: str
    temperature: float
    max_tokens: int
    top_k: Optional[int] = 10
    top_p: Optional[float] = 0.1
    streaming: bool = False
    include_usage: Optional[bool] = False
    model_kwargs: Optional[Dict[str, Any]] = None

    @weave.op()
    async def predict(self, query: str, system_message: str) -> Dict[str, Any]:
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
        client = APIGateway.load_chat(
            type=self.model_type,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            stream_options={'include_usage': True},
        )

        try:
            messages = [
                ('system', system_message),
                ('user', query),
            ]

            response = client.invoke(messages)
            completion = response.content.strip()
            usage = response.response_metadata.get('usage', None)
        except Exception as e:
            completion = f'<Error>: {type(e).__name__} - {str(e)}'
            usage = {}
        return {'completion': completion, 'usage': usage}
