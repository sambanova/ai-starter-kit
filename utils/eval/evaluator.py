import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)

from typing import Any, Dict, List

import weave
import yaml

from utils.eval.models import ChatModel, CorrectnessLLMJudge
from utils.visual.env_utils import get_wandb_key

wandb_api_key = get_wandb_key()

if wandb_api_key:
    import weave
else:
    print('WANDB_API_KEY is not set. Weave initialization skipped.')


CONFIG_PATH = os.path.join(current_dir, 'config.yaml')


class BaseWeaveEvaluator:
    def __init__(self) -> None:
        self.config_info = self._get_config_info(CONFIG_PATH)
        self.judge = self._init_judge()

    async def evaluate(self, data: List[Dict[str, Any]]) -> None:
        """
        Evaluate a list of data using multiple LLM configurations.

        This asynchronous method iterates over the LLM configurations specified in
        `self.config_info['llms']`, creating a `ChatModel` for each configuration.
        It then sets up an evaluation using the provided data and a judge, executing
        the evaluation for each model configuration.

        Args:
            data (List[Dict[str, Any]]): A list of dictionaries representing the
            dataset to be evaluated. Each dictionary should contain the relevant
            information required for the evaluation.

        Returns:
            None: This method does not return any value. It performs the evaluation
            asynchronously and logs the results to wnb.

        Raises:
            KeyError: If 'llms' is not present in `self.config_info`.
            Exception: If an error occurs during model evaluation or if the
            parameters provided to `ChatModel` are invalid.
        """
        llm_info = self.config_info['llms']
        for params in llm_info:
            test_model = ChatModel(**params)
            evaluation = weave.Evaluation(
                name=' '.join(str(value) for value in params.values()), dataset=data, scorers=[self.judge]
            )
            with weave.attributes(params):
                await evaluation.evaluate(test_model)

    def _get_config_info(self, config_path: str) -> Any:
        """
        Load configuration information from a YAML file.

        This method reads a YAML file from the specified path and loads its contents
        into a configuration dictionary. The loaded configuration can then be used
        for further processing or to set application parameters.

        Args:
            config_path (str): The file path to the YAML configuration file to be read.

        Returns:
            None: This method does not return any value. It populates the configuration
            information into the class's internal state or properties (if applicable).

        Raises:
            FileNotFoundError: If the specified config_path does not exist.
            yaml.YAMLError: If the file contains invalid YAML syntax.

        Example:
            config_path = 'path/to/config.yaml'
            self._get_config_info(config_path)
        """
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)

        return config

    def _init_judge(
        self,
    ) -> CorrectnessLLMJudge:
        """
        Initialize and return an instance of CorrectnessLLMJudge.

        This method retrieves configuration information for the evaluation LLM
        from the internal state (specifically from `self.config_info`) and
        creates an instance of `CorrectnessLLMJudge` using the extracted parameters.

        Returns:
            CorrectnessLLMJudge: An initialized instance of the CorrectnessLLMJudge
            based on the configuration settings specified in `self.config_info['eval_llm']`.

        Raises:
            KeyError: If 'eval_llm' is not present in `self.config_info`.
            TypeError: If the parameters provided to `CorrectnessLLMJudge`
            do not match its constructor signature.

        Example:
            judge = self._init_judge()
        """
        judge_info = self.config_info['eval_llm']
        return CorrectnessLLMJudge(**judge_info)
