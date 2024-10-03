import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import wandb
import yaml
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.llms.sambanova import SambaStudio
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Configure Utils Location
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))


class SambaStudioLLM(DeepEvalBaseLLM):
    """
    A wrapper class for the SambaStudio LLM, implementing DeepEvalBaseLLM interface.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SambaStudioLLM.

        Args:
            config (Dict[str, Any]): Configuration dictionary for SambaStudio.
        """
        self.config = config
        self.model: Optional[SambaStudio] = None

    def load_model(self) -> SambaStudio:
        """
        Load the SambaStudio model if not already loaded.

        Returns:
            SambaStudio: The loaded SambaStudio model.
        """
        if self.model is None:
            self.model = SambaStudio(**self.config)
        return self.model

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the SambaStudio model.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The generated response.
        """
        model = self.load_model()
        template = PromptTemplate.from_template(
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are JSON generator<|eot_id|>
            <|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>
            <|start_header_id|>machine-readable JSON<|end_header_id|>\n\n"""
        )
        chain = {'prompt': RunnablePassthrough()} | template | model
        return chain.invoke(prompt)

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response using the SambaStudio model.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The generated response.
        """
        return self.generate(prompt)

    def get_model_name(self) -> str:
        """
        Get the name of the SambaStudio model.

        Returns:
            str: The model name.
        """
        select_expert = self.config.get('model_kwargs', {}).get('select_expert', 'Unknown')
        return f'SambaStudio-{select_expert}'


class RAGEvalConfig:
    """
    Configuration class for RAG evaluation.
    """

    def __init__(self, config_yaml_path: str) -> None:
        """
        Initialize the RAGEvalConfig.

        Args:
            config_yaml_path (str): Path to the YAML configuration file.
        """
        with open(config_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_llm_config(self, llm_config: Tuple[str, Dict]) -> Tuple[str, Dict]:
        """
        Get the full configuration for an LLM.

        Args:
            llm_config (Tuple[str, Dict]): LLM name and partial configuration.

        Returns:
            Tuple[str, Dict]: LLM name and full configuration.
        """
        llm_name, config_dict = llm_config
        full_config_dict = {
            'sambastudio_base_url': os.getenv(f'{llm_name.upper()}_BASE_URL'),
            'sambastudio_project_id': os.getenv(f'{llm_name.upper()}_PROJECT_ID'),
            'sambastudio_endpoint_id': os.getenv(f'{llm_name.upper()}_ENDPOINT_ID'),
            'sambastudio_api_key': os.getenv(f'{llm_name.upper()}_API_KEY'),
            'streaming': True,
            'model_kwargs': config_dict.get('model_kwargs', {}),
        }
        return llm_name, full_config_dict

    @property
    def eval_dataset_path(self) -> str:
        """Get the evaluation dataset path."""
        return self.config['eval_dataset'].get('path')

    @property
    def eval_dataset_question_col(self) -> str:
        """Get the question column name for the evaluation dataset."""
        return self.config['eval_dataset']['question_col']

    @property
    def eval_dataset_answer_col(self) -> str:
        """Get the answer column name for the evaluation dataset."""
        return self.config['eval_dataset']['answer_col']

    @property
    def eval_dataset_ground_truth_col(self) -> str:
        """Get the ground truth column name for the evaluation dataset."""
        return self.config['eval_dataset']['ground_truth_col']

    @property
    def eval_dataset_context_col(self) -> Optional[str]:
        """Get the context column name for the evaluation dataset."""
        return self.config['eval_dataset'].get('context_col')

    @property
    def eval_llm_configs(self) -> List[Tuple[str, Dict]]:
        """Get the configurations for evaluation LLMs."""
        return [(llm['name'], llm) for llm in self.config['eval_llms']]


class RAGEvaluator:
    """
    Class for performing RAG evaluation.
    """

    def __init__(self, config_yaml_path: str) -> None:
        """
        Initialize the RAGEvaluator.

        Args:
            config_yaml_path (str): Path to the YAML configuration file.
        """
        self.config = RAGEvalConfig(config_yaml_path)
        self.eval_llms = []
        for llm_name, llm_config in self.config.eval_llm_configs:
            _, full_config = self.config.get_llm_config((llm_name, llm_config))
            self.eval_llms.append((llm_name, SambaStudioLLM(full_config)))

    def create_test_cases(self, df: pd.DataFrame) -> List[LLMTestCase]:
        """
        Create test cases from a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing evaluation data.

        Returns:
            List[LLMTestCase]: List of created test cases.
        """
        test_cases = []
        context_col = self.config.eval_dataset_context_col
        for _, row in df.iterrows():
            context = row.get(context_col) if context_col else None
            if isinstance(context, str):
                context = [context] if context else None
            elif isinstance(context, list):
                context = context if any(context) else None
            else:
                context = None

            test_case = LLMTestCase(
                input=row[self.config.eval_dataset_question_col],
                actual_output=row[self.config.eval_dataset_answer_col],
                expected_output=row[self.config.eval_dataset_ground_truth_col],
                retrieval_context=context,
            )
            test_cases.append(test_case)
        return test_cases

    def create_metrics(self, eval_llm: SambaStudioLLM) -> List:
        """
        Create evaluation metrics.

        Args:
            eval_llm (SambaStudioLLM): The LLM to use for evaluation.

        Returns:
            List: List of evaluation metrics.
        """
        metrics = [
            AnswerRelevancyMetric(threshold=0.7, model=eval_llm, verbose_mode=True),
            GEval(
                name='Correctness',
                criteria='Correctness - determine if the actual output is correct according to the expected output.',
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                evaluation_steps=[
                    """Compare the actual output directly with the expected output to verify factual accuracy.""",
                    """Check if all elements mentioned in the expected output are present and correctly represented in
                     the actual output.""",
                    """Assess if there are any discrepancies in details, values, or information between the actual and
                     expected outputs.""",
                ],
                model=eval_llm,
            ),
        ]

        if self.config.eval_dataset_context_col:
            context_metrics = [
                FaithfulnessMetric(threshold=0.7, model=eval_llm),
                ContextualRecallMetric(threshold=0.7, model=eval_llm),
                ContextualPrecisionMetric(threshold=0.7, model=eval_llm),
            ]
            metrics.extend(context_metrics)
        else:
            logger.warning('Context column not found in config. Skipping context-dependent metrics.')

        return metrics

    def evaluate(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform evaluation on the given DataFrame.

        Args:
            eval_df (pd.DataFrame): DataFrame containing evaluation data.

        Returns:
            Dict[str, Any]: Evaluation results for each LLM.
        """
        self.test_cases = self.create_test_cases(eval_df)
        dataset = EvaluationDataset(test_cases=self.test_cases)

        results = {}
        for llm_name, eval_llm in self.eval_llms:
            logger.info(f'Evaluating {llm_name}...')
            metrics = self.create_metrics(eval_llm)
            result = evaluate(dataset, metrics)
            results[llm_name] = result

        return results


def log_results_to_wandb(results_file: str = '.deepeval-cache.json') -> None:
    """
    Log evaluation results to Weights & Biases.

    Args:
        results_file (str): Path to the results file. Defaults to ".deepeval-cache.json".
    """
    # Read the JSON file
    with open(results_file, 'r') as f:
        results_data = json.load(f)

    # Extract the test cases and their metrics
    test_cases = results_data.get('test_cases_lookup_map', {})

    # Prepare data for the DataFrame
    data = []
    for test_case, metrics in test_cases.items():
        test_case_data = json.loads(test_case)
        for metric in metrics.get('cached_metrics_data', []):
            data.append(
                {
                    'input': test_case_data.get('input'),
                    'actual_output': test_case_data.get('actual_output'),
                    'expected_output': test_case_data.get('expected_output'),
                    'metric': metric['metric_metadata']['metric'],
                    'score': metric['metric_metadata']['score'],
                    'threshold': metric['metric_metadata']['threshold'],
                    'success': metric['metric_metadata']['success'],
                    'reason': metric['metric_metadata']['reason'],
                }
            )

    # Create DataFrame
    df = pd.DataFrame(data)

    # Initialize W&B run
    wandb.init(project='rag-evaluation')

    # Log the DataFrame as a table
    wandb_table = wandb.Table(dataframe=df)
    wandb.log({'results': wandb_table})

    # Log summary metrics
    summary_metrics = df.groupby('metric').agg({'score': 'mean', 'success': 'mean'}).reset_index()

    for _, row in summary_metrics.iterrows():
        wandb.log(
            {
                f"{row['metric']}_avg_score": row['score'],
                f"{row['metric']}_success_rate": row['success'],
            }
        )

    wandb.finish()


def main() -> None:
    """
    Main function to run the RAG evaluation.
    """

    config_path = os.path.abspath(os.path.join(current_dir, 'config.yaml'))
    eval_csv_path = os.path.abspath(os.path.join(current_dir, 'data/test.csv'))

    logger.info('Initializing RAG Evaluator...')
    evaluator = RAGEvaluator(config_path)

    logger.info(f'Loading evaluation data from {eval_csv_path}...')
    eval_df = pd.read_csv(eval_csv_path)

    logger.info('Starting evaluation...')
    results = evaluator.evaluate(eval_df)

    for llm_name, result in results.items():
        logger.info(f'Results for {llm_name}:')
        logger.info(result)

    logger.info('Logging results to Weights & Biases...')
    log_results_to_wandb()

    logger.info('Evaluation completed successfully.')


if __name__ == '__main__':
    main()
