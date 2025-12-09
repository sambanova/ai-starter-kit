import os
import sys

from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import weave
import yaml
from weave import Dataset

from utils.eval.dataset import WeaveDatasetManager
from utils.eval.models import CorrectnessLLMJudge, WeaveChatModel, WeaveRAGModel
from utils.eval.rag import RAGChain

CONFIG_PATH = os.path.join(current_dir, 'config.yaml')

load_dotenv(os.path.join(repo_dir, '.env'), override=True)


class WeaveEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self, name: Optional[str] = None, filepath: Optional[str] = None, use_concurrency: bool = False
    ) -> None:
        pass


class BaseWeaveEvaluator(WeaveEvaluator):
    """
    Base class for evaluating LLM configurations using the Weave framework.

    This class provides a basic structure for evaluating multiple LLM configurations
    using the Weave framework. It initializes a judge, dataset manager, and loads
    configuration information from a specified path.

    Attributes:
        config_info (dict): Configuration information loaded from CONFIG_PATH.
        judge (object): Judge object used for evaluation.
        dataset_manager (object): Dataset manager object used for creating datasets.
    """

    def __init__(self) -> None:
        self.config_info = self._get_config_info(CONFIG_PATH)
        self.judge = self._init_judge()
        self.dataset_manager = self._init_dataset_manager()

    async def evaluate(
        self, name: Optional[str] = None, filepath: Optional[str] = None, use_concurrency: bool = False
    ) -> None:
        """
        Evaluate a list of data using multiple LLM configurations.

        This asynchronous method iterates over the LLM configurations specified in
        self.config_info['llms'], creating a WeaveChatModel for each configuration.
        It then sets up an evaluation using the provided data and a judge, executing
        the evaluation for each model configuration.

        Args:
            name (str, optional): Name of the dataset. Defaults to the value specified
                in self.config_info['eval_dataset']['name'].
            filepath (str, optional): Path to the dataset file. Defaults to the value
                specified in self.config_info['eval_dataset']['path'].

        Returns:
            None: This method does not return any value. It performs the evaluation
            asynchronously and logs the results to wnb.

        Raises:
            KeyError: If 'llms' is not present in self.config_info.
            Exception: If an error occurs during model evaluation or if the
            parameters provided to WeaveChatModel are invalid.
        """
        if name is None:
            name = self.config_info['eval_dataset']['name']
        if filepath is None:
            filepath = self.config_info['eval_dataset']['path']

        data = self.dataset_manager.create_dataset(name, filepath)

        llm_info = self.config_info['llms']

        if use_concurrency:
            await self._run_concurrently_with_threads(llm_info, data)
        else:
            await self._run_sequentially(llm_info, data)

    async def _run_sequentially(self, params: List[Dict[str, Any]], data: Dataset) -> None:
        """
        Run evaluations of models sequentially for a list of parameters.
        This method creates a `WeaveChatModel` for each parameter set, constructs an evaluation object,
        and then evaluates the model within the context of the given parameters.

        Args:
            params (List[Dict[str, Any]]): A list of dictionaries containing parameters for each model to evaluate.
            data (Dataset): The dataset to be used for evaluation.
        """

        for param in params:
            test_model = WeaveChatModel(**param)
            evaluation = weave.Evaluation(
                name=' '.join(str(value) for value in param.values()), dataset=data, scorers=[self.judge]
            )
            with weave.attributes(param):
                await evaluation.evaluate(test_model)

    async def _run_concurrently_with_threads(self, params: List[Dict[str, Any]], data: Dataset) -> None:
        """
        Run evaluations of models concurrently using threads for a list of parameters.
        This method utilizes a thread pool to run model evaluations concurrently.
        Each set of parameters is evaluated in a separate thread to optimize performance.

        Args:
            params (List[Dict[str, Any]]): A list of dictionaries containing parameters for each model to evaluate.
            data (Dataset): The dataset to be used for evaluation.
        """

        with ThreadPoolExecutor() as executor:
            evaluation_tasks = [
                asyncio.get_event_loop().run_in_executor(executor, self._evaluate_model, param, data)
                for param in params
            ]

            await asyncio.gather(*evaluation_tasks)

    def _evaluate_model(self, params: Dict[str, Any], data: Dataset) -> None:
        """
        Evaluate a model using the provided parameters in a separate thread.
        This method runs in a thread pool. It creates a `WeaveChatModel` using the
        specified parameters and evaluates it using the provided dataset and judges.

        Args:
            params (Dict[str, Any]): A dictionary containing parameters for the model to evaluate.
            data (Dataset): The dataset to be used for evaluation.
        """
        test_model = WeaveChatModel(**params)
        evaluation = weave.Evaluation(
            name=' '.join(str(value) for value in params.values()), dataset=data, scorers=[self.judge]
        )

        with weave.attributes(params):
            asyncio.run(evaluation.evaluate(test_model))

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

    def _init_dataset_manager(self) -> WeaveDatasetManager:
        """
        Initialize and return an instance of WeaveDatasetManager.

        Returns:
            WeaveDatasetManager: An initialized instance.
        """
        return WeaveDatasetManager()


class BaseWeaveRAGEvaluator(BaseWeaveEvaluator):
    """
    Base class for evaluating RAG models using the Weave framework.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rag_info = self.config_info['rag']
        self.rag_chain = self._init_chain()

    def populate_vectordb(self, path: str) -> None:
        """
        Populate the VectorDB with documents from the specified path.

        Args:
            path (str): The path to the directory containing the documents to be uploaded.
        """
        self.rag_chain.upload_docs(path)

    async def evaluate(
        self, name: Optional[str] = None, filepath: Optional[str] = None, use_concurrency: bool = False
    ) -> None:
        """
        Evaluate a list of data using multiple LLM configurations.

        This asynchronous method iterates over the LLM configurations specified in
        self.config_info['llms'], creating a WeaveChatModel for each configuration.
        It then sets up an evaluation using the provided data and a judge, executing
        the evaluation for each model configuration.

        Args:
            name (str, optional): Name of the dataset. Defaults to the value specified
                in self.config_info['eval_dataset']['name'].
            filepath (str, optional): Path to the dataset file. Defaults to the value
                specified in self.config_info['eval_dataset']['path'].

        Returns:
            None: This method does not return any value. It performs the evaluation
                asynchronously and logs the results to wnb.

        Raises:
            KeyError: If 'llms' is not present in self.config_info.
            Exception: If an error occurs during model evaluation or if the
                parameters provided to WeaveChatModel are invalid.
        """
        if name is None:
            name = self.config_info['eval_dataset']['name']
        if filepath is None:
            filepath = self.config_info['eval_dataset']['path']

        weave_data = self.dataset_manager.create_dataset(name, filepath)

        evaluation = weave.Evaluation(
            name=' '.join(str(value) for value in self.rag_info.values()), dataset=weave_data, scorers=[self.judge]
        )

        await evaluation.evaluate(self.rag_chain)

    def _init_chain(self) -> RAGChain:
        """
        Initialize and return an instance of RAGChain.

        This method retrieves configuration information for the RAG model
        from the internal state and creates an instance of `RAGChain`
        using the extracted parameters.

        Returns:
            RAGChain: An initialized instance of the RAGChain
                based on the configuration settings specified in `self.config_info['rag']`.

        Raises:
            KeyError: If 'rag' is not present in `self.config_info`.
            TypeError: If the parameters provided to `RAGChain`
                do not match its constructor signature.
        """
        rag_info = self.config_info['rag']
        llm_info = rag_info['llm']
        embeddings_info = rag_info['embeddings']
        vectordb_info = rag_info['vectordb']
        model_kwargs = rag_info.get('model_kwargs')

        return WeaveRAGModel(
            name=rag_info.get('name'),
            llm_params=llm_info,
            embeddings_params=embeddings_info,
            rag_params=vectordb_info,
            model_kwargs=model_kwargs,
        )
