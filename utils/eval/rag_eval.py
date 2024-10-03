import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import wandb
import yaml
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    context_relevancy,
    faithfulness,
)

load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


class RAGEvalConfig:
    """Configuration for RAG Evaluation"""

    def __init__(self, config_yaml_path: str) -> None:
        with open(config_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_llm_config(self, llm_config: Tuple[str, Dict]) -> Tuple[str, Dict]:
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
        return self.config['eval_dataset'].get('path')

    @property
    def eval_dataset_question_col(self) -> str:
        return self.config['eval_dataset']['question_col']

    @property
    def eval_dataset_answer_col(self) -> str:
        return self.config['eval_dataset']['answer_col']

    @property
    def eval_dataset_ground_truth_col(self) -> str:
        return self.config['eval_dataset']['ground_truth_col']

    @property
    def eval_dataset_context_col(self) -> str:
        return self.config['eval_dataset'].get('context_col')

    @property
    def embedding_model_name(self) -> str:
        return self.config['embeddings']['model_name']

    @property
    def llm_configs(self) -> List[Tuple[str, Dict]]:
        return [(llm['name'], llm) for llm in self.config['llms']]

    @property
    def eval_llm_configs(self) -> List[Tuple[str, Dict]]:
        return [(llm['name'], llm) for llm in self.config['eval_llms']]

    @property
    def save_eval_table_csv(self) -> bool:
        return self.config['evaluation'].get('save_eval_table_csv', True)

    def print_config_keys(self) -> None:
        logging.info('Configuration Keys:')
        for key, value in self.config.items():
            logging.info(f'{key}:')
            if isinstance(value, list):
                for item in value:
                    logging.info(f'  - {item}')
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    logging.info(f'  {sub_key}: {sub_value}')
            else:
                logging.info(f'  {value}')

    @property
    def vector_db_location(self) -> str:
        return self.config['pipeline']['kwargs'].get('vector_db_location')

    @property
    def num_eval_samples(self) -> Optional[int]:
        return self.config['evaluation'].get('num_samples')

    @property
    def log_wandb(self) -> bool:
        return self.config['evaluation']['log_wandb']

    @property
    def wandb_project(self) -> str:
        return self.config['evaluation']['project_name']

    @property
    def wandb_eval_name(self) -> Optional[str]:
        return self.config['evaluation'].get('eval_name')

    @property
    def eval_methodology(self) -> str:
        return self.config['evaluation']['methodology']

    @property
    def user_provided_answers_path(self) -> str:
        return self.config.get('user_provided_answers', None)

    @property
    def hf_dataset_name(self) -> str:
        return self.config['eval_dataset'].get('hf_dataset_name')

    def get_flattened_config(self) -> Dict:
        """Return flattened config dict for logging"""
        flat_config = {}
        for k, v in self.config.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat_config[f'{k}.{kk}'] = vv
            else:
                flat_config[k] = v
        return flat_config


class RAGEvaluator:
    """RAG Evaluation Module"""

    def __init__(
        self,
        eval_llms: List[Tuple[str, BaseLLM]],
        eval_embeddings: Embeddings,
        config_yaml_path: str,
        csv_filename: str = 'eval_table.csv',
    ) -> None:
        self.eval_llms = eval_llms
        self.eval_embeddings = eval_embeddings
        self.config = RAGEvalConfig(config_yaml_path)
        self.csv_filename = csv_filename

    def create_ragas_dataset(
        self,
        eval_df: pd.DataFrame,
        llm_name: Optional[str] = None,
        answers_df: Optional[pd.DataFrame] = None,
    ) -> Dataset:
        """Create RAGAS eval dataset from question, answer, context dataframe"""
        if answers_df is not None:
            eval_df = pd.merge(eval_df, answers_df, on=self.config.eval_dataset_question_col)

        ragas_data = []
        for _, row in eval_df.iterrows():
            answer_col = (
                f'answer_{llm_name}'
                if llm_name and f'answer_{llm_name}' in eval_df.columns
                else self.config.eval_dataset_answer_col
            )
            context_col = None
            if llm_name:
                context_col = f'context_{llm_name}'
                if context_col not in eval_df.columns:
                    context_col = None
            else:
                context_col = self.config.eval_dataset_context_col

            ragas_data.append(
                {
                    'question': row[self.config.eval_dataset_question_col],
                    'answer': row[answer_col],
                    'ground_truth': row[self.config.eval_dataset_ground_truth_col],
                    'contexts': ([row[context_col]] if context_col and not pd.isna(row[context_col]) else ['']),
                }
            )

        ragas_dataset = Dataset.from_list(ragas_data)
        return ragas_dataset

    def evaluate(
        self,
        eval_df: pd.DataFrame,
        answer_generation_pipelines: Optional[List] = None,
    ) -> Dict:
        """Run RAG evaluation"""

        gen_llm_names = []
        if answer_generation_pipelines:
            num_samples = self.config.num_eval_samples
            if num_samples is None or num_samples > len(eval_df):
                num_samples = len(eval_df)

            for llm_name, pipeline in answer_generation_pipelines:
                answers = []
                contexts = []
                for _, row in eval_df.iloc[:num_samples].iterrows():
                    query = row[self.config.eval_dataset_question_col]
                    logging.info(f'Generating answer for query: {query}')
                    result = pipeline.generate(query)

                    if isinstance(result['answer'], dict):
                        answer = result['answer']['result']
                        source_documents = result['answer'].get('source_documents', [])
                        if source_documents:
                            context = '\n\n=====\n\n'.join(
                                [f'Context {i+1}:\n{doc.page_content}' for i, doc in enumerate(source_documents)]
                            )
                        else:
                            context = ''
                    else:
                        answer = result['answer']
                        context = ''

                    answers.append(answer)
                    if context:
                        contexts.append(context)
                    else:
                        contexts.append(None)

                eval_df.loc[: num_samples - 1, f'answer_{llm_name}'] = answers
                if contexts and any(ctx is not None for ctx in contexts):
                    eval_df.loc[: num_samples - 1, f'context_{llm_name}'] = contexts
                gen_llm_names.append(llm_name)

        metrics = [
            answer_relevancy,
            answer_correctness,
            answer_similarity,
        ]

        # Check if context columns exist for any of the generation LLMs or if eval_dataset_context_col is provided
        context_cols_exist = any(f'context_{llm_name}' in eval_df.columns for llm_name in gen_llm_names) or (
            self.config.eval_dataset_context_col and self.config.eval_dataset_context_col in eval_df.columns
        )

        # Extend the metrics if context columns exist or if eval_dataset_context_col is provided
        if context_cols_exist:
            metrics.extend(
                [
                    context_precision,
                    context_recall,
                    faithfulness,
                    context_relevancy,
                ]
            )

        results = {}
        # logging.info(f"Evaluation dataframe:\n{eval_df}")

        num_samples = self.config.num_eval_samples
        if num_samples is None or num_samples > len(eval_df):
            num_samples = len(eval_df)

        if gen_llm_names:
            for gen_llm_name in gen_llm_names:
                for eval_llm_name, eval_llm in self.eval_llms:
                    ragas_dataset = self.create_ragas_dataset(eval_df.iloc[:num_samples], gen_llm_name)

                    logging.info(f'Evaluating metrics for {gen_llm_name} and {eval_llm_name}')
                    result = evaluate(
                        ragas_dataset,
                        metrics=metrics,
                        llm=eval_llm,
                        embeddings=self.eval_embeddings,
                    )
                    results[f'{gen_llm_name}_{eval_llm_name}'] = result.to_pandas()  # Use result.to_pandas()
        else:
            for eval_llm_name, eval_llm in self.eval_llms:
                ragas_dataset = self.create_ragas_dataset(eval_df.iloc[:num_samples], None)

                logging.info(f'Evaluating metrics for {eval_llm_name}')
                result = evaluate(
                    ragas_dataset,
                    metrics=metrics,
                    llm=eval_llm,
                    embeddings=self.eval_embeddings,
                )
                # logging.info(f"Results dataframe for {eval_llm_name}:\n{result.to_pandas()}")
                results[eval_llm_name] = result.to_pandas()  # Use result.to_pandas()

        if self.config.log_wandb:
            logging.info('Logging results to Weights & Biases')
            self._log_wandb(eval_df.iloc[:num_samples], results)

        if self.config.save_eval_table_csv:
            pass
            # self.l.to_csv(self.csv_filename, index=False)

        return results

    def create_wandb_table(self, eval_df: pd.DataFrame, results: Dict) -> pd.DataFrame:
        table_data = []
        flattened_config = self.config.get_flattened_config()

        for _, row in eval_df.iterrows():
            base_row_data = {
                'question': row[self.config.eval_dataset_question_col],
                'ground_truth': row[self.config.eval_dataset_ground_truth_col],
                'user_answer': (
                    row[self.config.eval_dataset_answer_col] if self.config.eval_dataset_answer_col else None
                ),
                'user_context': (
                    row[self.config.eval_dataset_context_col]
                    if self.config.eval_dataset_context_col and self.config.eval_dataset_context_col in eval_df.columns
                    else None
                ),
                **flattened_config,
            }

            for eval_llm_name, eval_results_df in results.items():
                row_data = {
                    **base_row_data,
                    'gen_llm_name': None,
                    'generated_answer': None,
                    'context': None,
                }

                eval_results_row = eval_results_df[
                    eval_results_df['question'] == row[self.config.eval_dataset_question_col]
                ]

                for metric_name in eval_results_row.columns:
                    if metric_name in [
                        'question',
                        'answer',
                        'ground_truth',
                        'contexts',
                    ]:
                        continue
                    row_data['eval_llm_name'] = eval_llm_name
                    row_data['metric_name'] = metric_name
                    row_data['metric_value'] = eval_results_row[metric_name].values[0]
                    table_data.append(row_data.copy())

        columns = [
            'question',
            'ground_truth',
            'user_answer',
            'user_context',
            *flattened_config.keys(),
            'gen_llm_name',
            'generated_answer',
            'context',
            'eval_llm_name',
            'metric_name',
            'metric_value',
        ]

        df = pd.DataFrame(table_data, columns=columns)

        return df

    def _log_wandb(self, eval_df: pd.DataFrame, results: Dict) -> None:
        """Log eval results and config to Weights & Biases"""
        run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_eval_name if self.config.wandb_eval_name else None,
            config=self.config.get_flattened_config(),
        )

        table_df = self.create_wandb_table(eval_df, results)
        self.logging_table = table_df
        run.log({'eval_table': wandb.Table(dataframe=table_df)})

        # Extract data from the DataFrame
        eval_llm_names = self.logging_table['eval_llm_name'].unique()
        metrics = ['answer_relevancy', 'answer_correctness', 'answer_similarity']

        data = []
        for metric in metrics:
            # Gather metric values for each evaluation LLM name
            metric_values = [
                self.logging_table[
                    (self.logging_table['eval_llm_name'] == llm_name) & (self.logging_table['metric_name'] == metric)
                ]['metric_value'].mean()  # Use mean() to handle NaN values
                for llm_name in eval_llm_names
            ]

            # Prepare data for the current metric
            data = [[llm_name, value] for llm_name, value in zip(eval_llm_names, metric_values)]
            table = wandb.Table(data=data, columns=['Evaluation Model', 'Value'])

            # Log the bar chart for the current metric
            run.log(
                {
                    f'Metric_Value_by_Evaluation Model_{metric}': wandb.plot.bar(
                        table,
                        'Evaluation Model',
                        'Value',
                        title=f'Metric_Value_by_Evaluation Model_{metric}',
                    )
                }
            )

        run.finish()


def load_pipeline(llm: Tuple[str, BaseLLM], config: RAGEvalConfig) -> Tuple[str, Any]:
    """Dynamically load answer generation pipeline from config"""
    llm_name, llm_instance = llm

    pipeline_class = config.config['pipeline']['class']
    pipeline_kwargs = config.config['pipeline'].get('kwargs', {}) or {}

    module_name, class_name = pipeline_class.split('.')
    module = __import__(module_name, fromlist=[class_name])

    PipelineClass = getattr(module, class_name)
    pipeline_kwargs['llm'] = llm_instance

    if 'vector_db_location' in pipeline_kwargs:
        pipeline_kwargs['embeddings'] = HuggingFaceInstructEmbeddings(model_name=config.embedding_model_name)
        pipeline_kwargs['vector_db_location'] = config.vector_db_location

    logging.info(f'Pipeline: {PipelineClass(**pipeline_kwargs)}')

    return llm_name, PipelineClass(**pipeline_kwargs)


def load_eval_dataframe(config: RAGEvalConfig) -> Any:
    if config.hf_dataset_name:
        dataset = load_dataset(config.hf_dataset_name)
        eval_df = dataset['test'].to_pandas()
    else:
        eval_df = pd.read_csv(config.eval_dataset_path)

    return eval_df
