import sys

sys.path.append("../")

import pandas as pd
import yaml
from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain_community.llms.sambanova import Sambaverse
from typing import List, Dict, Optional
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    answer_correctness,
    answer_similarity,
    context_recall,
    context_relevancy,
)
from ragas import evaluate
from ragas.run_config import RunConfig
from abc import ABC, abstractmethod
import os
from rag.RAG_pipeline import *
from rag import utils
from datasets import Dataset
from dotenv import load_dotenv
import wandb
from openai import OpenAI

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

load_dotenv()


class RAGEvalBase(ABC):
    """Base class for RAG Evaluation"""

    def __init__(self, llm: BaseLLM, config_yaml_path: Optional[str] = None):
        """
        Initialize RAGEvalBase

        Args:
            llm (BaseLLM): The language model to use for evaluation.
            config_yaml_path (str, optional): Path to a config YAML file. Defaults to None.
        """
        self.llm = llm

        if config_yaml_path:
            with open(config_yaml_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

    @abstractmethod
    def evaluate(
        self, eval_dataset: pd.DataFrame, num_samples: int, log_wandb: bool
    ) -> Dict:
        """
        Evaluate the RAG pipeline.

        Args:
            eval_dataset (pd.DataFrame): The evaluation dataset.
            num_samples (int): Number of samples to evaluate on.
            log_wandb (bool): Whether to log results to Weights & Biases.

        Returns:
            Dict: Evaluation results
        """
        pass

    def _log_wandb(self, results: Dict):
        """
        Log results to Weights & Biases

        Args:
            results (Dict): Evaluation results to log
        """

        # Log config
        wandb.config.update(self.config)

        # Log metrics
        wandb.log(results)

        print("Logged results to Weights & Biases")


class RAGASEval(RAGEvalBase):
    """RAGAS Evaluation Module"""

    def __init__(
        self,
        llm: BaseLLM,
        rag_pipeline: RAGPipeline,
        rag_config_path: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        eval_llm: Optional[BaseLLM] = None,
    ):
        """
        Initialize RAGASEval

        Args:
            llm (BaseLLM): The language model to use for evaluation.
            rag_pipeline (RAGPipeline): The RAG pipeline to evaluate.
            config_yaml_path (str, optional): Path to a config YAML file. Defaults to None.
        """
        super().__init__(llm, rag_config_path)
        self.rag_pipeline = rag_pipeline
        self.llm = llm
        self.embeddings = embeddings
        self.eval_llm = eval_llm

    def create_ragas_dataset(
        self, eval_dataset: pd.DataFrame, num_samples: int
    ) -> List[Dict]:
        """
        Create a RAGAS evaluation dataset.

        Args:
            eval_dataset (pd.DataFrame): The evaluation dataset.
            num_samples (int): Number of samples to evaluate on.

        Returns:
            List[Dict]: RAGAS evaluation dataset
        """

        # Load ReRankers
        rerank_tokenizer, rerank_model = load_reranker_model()

        rag_dataset = []
        for _, row in eval_dataset.sample(n=num_samples).iterrows():
            device_name = utils.get_device_name(row["question"])[0]
            cur_filter = {"device_name": device_name.lower()}

            response = self.rag_pipeline.qa_chain(
                {
                    "question": row["question"],
                    "filter": cur_filter,
                    "reranker": rerank_model,
                    "tokenizer": rerank_tokenizer,
                    "final_k": 3,
                }
            )

            final_answer = final_answer = self.rag_pipeline.OutputParser(
                response["answer"]
            )
            rag_dataset.append(
                {
                    "question": row["question"],
                    "contexts": [
                        context.page_content for context in response["source_documents"]
                    ],
                    "answer": final_answer,
                    "ground_truth": row["ground_truth"],
                }
            )

        rag_df = pd.DataFrame(rag_dataset)

        # rag_df.to_csv("../../data/eval/datasets/rag_dataset.csv")

        rag_eval_dataset = Dataset.from_pandas(rag_df)
        return rag_eval_dataset

    def evaluate(
        self, eval_dataset: pd.DataFrame, num_samples: int = 100, log_wandb: bool = True
    ) -> Dict:
        """
        Evaluate the RAG pipeline using RAGAS.

        Args:
            eval_dataset (pd.DataFrame): The evaluation dataset. Must contain columns 'question' and 'ground_truth'.
            num_samples (int): Number of samples to evaluate on. Defaults to 100.
            log_wandb (bool): Whether to log results to Weights & Biases. Defaults to True.

        Returns:
            Dict: Evaluation results
        """

        num_samples = (
            num_samples
            if len(eval_dataset.index) >= num_samples
            else len(eval_dataset.index)
        )

        logging.info('Creating RAGAS dataset')
        ragas_dataset = self.create_ragas_dataset(eval_dataset, num_samples)
        
        logging.info('Evaluating RAGAS dataset')
        run_config = RunConfig(timeout=1800, max_wait=1800)
        # print(f'run_config: {run_config}')
        
        results = evaluate(
            ragas_dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
                answer_correctness,
            ],
            llm=self.eval_llm,
            embeddings=self.embeddings,
            is_async = False,
            run_config = run_config
        )
        
        if log_wandb:
            logging.info('Logging RAGAS results')
            self._log_wandb(results)

        return results


if __name__ == "__main__":
    
    # Load eval config
    eval_config_file = "./eval/config.yaml"
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_config_path = os.path.join(package_dir, eval_config_file)
    with open(eval_config_path, 'r') as yaml_file:
            eval_config = yaml.safe_load(yaml_file)

    # Load rag config
    rag_config_file = "./rag/config.yaml"
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rag_config_path = os.path.join(package_dir, rag_config_file)
    with open(rag_config_path, 'r') as yaml_file:
            rag_config = yaml.safe_load(yaml_file)

    # Load RAG pipeline
    rag_pipeline = RAGPipeline(
        llm=load_llama2(),
        vector_db_location=rag_config['vector_db']['vector_db_location'],
        embedding_model=get_embeddings(),
        k=rag_config['vector_db']['k']
    )

    # Set Up Eval LLM To Be Used In Judging

    # eval_llm = Sambaverse(
    #     sambaverse_model_name=eval_config['eval_llm']['sambaverse_model_name'],
    #     model_kwargs=eval_config['eval_llm']['model_kwargs']
    # )
    
    # print(f'eval_llm: {eval_llm("greet me as a helpful assistant")}')
    
    # Load evaluation dataset
    eval_data_path = eval_config['evaluation']['data_path']
    eval_df = pd.read_csv(eval_data_path)

    # Initialize RAGASEval
    rag_eval = RAGASEval(
        llm=rag_pipeline.llm,
        rag_pipeline=rag_pipeline,
        rag_config_path=rag_config_path,
        embeddings=rag_pipeline.embedding_model,
        eval_llm=None,
    )

    # Run evaluation
    logging.info('RAGAS evaluation started')
    results = rag_eval.evaluate(eval_dataset=eval_df, num_samples=30, log_wandb=False)

    results_df = results.to_pandas()
    results_file_name = eval_data_path.split('/')[-1]
    results_df.to_csv(f"../../data/eval/results/adi_qna_sample/results_{results_file_name}")
