import re
import sys
import os
import logging

sys.path.append("../")
sys.path.append("../../../")

import shutil
import json
from typing import Any, Dict, List, Optional, Callable

from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

from utils.sambanova_endpoint import SambaNovaEndpoint
from rag.rerank_retriever import (
    RetrievalQAReranker,
    VectorStoreRetrieverReranker,
)
from rag import utils

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
import pandas as pd
from datasets import Dataset

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

B_INST, E_INST = "[INST]", "[/INST]"


def parse(input_string: str) -> Dict:
    """Parse JSON strings to dict."""

    print(input_string)
    if not input_string.strip():
        return {"score": "no"}

    start_index = input_string.find("{")
    end_index = input_string.rfind("}") + 1

    if start_index == -1 or end_index == 0:
        return {"score": "no"}

    json_string = input_string[start_index:end_index]
    addn_details = input_string[:start_index].strip() + input_string[end_index:].strip()

    formatted_json_string = re.sub(r"(\w+):", r'"\1":', json_string)
    formatted_json_string = re.sub(r": (\w+)", r': "\1"', formatted_json_string)

    try:
        json_data = json.loads(formatted_json_string)
        if addn_details:
            json_data["addn_details"] = addn_details
        return json_data

    except json.JSONDecodeError:
        return {"score": "no"}


class RAGASEval:
    """RAGAS Evaluation Module"""

    def __init__(
        self,
        llm: BaseLLM,
        rag_pipeline: SelfCorrectingRAG,
        embeddings: Optional[Embeddings] = None,
        eval_llm: Optional[BaseLLM] = None,
    ):
        """
        Initialize RAGASEval

        Args:
            llm: The language model to use for evaluation.
            rag_pipeline: The RAG pipeline to evaluate.
            embeddings: The embedding model used in RAG.
        """
        self.llm = llm
        self.rag_pipeline = rag_pipeline
        self.embeddings = embeddings
        self.eval_llm = eval_llm

    def create_ragas_dataset(
        self, eval_dataset: pd.DataFrame, num_samples: int
    ) -> Dataset:
        """
        Create a RAGAS evaluation dataset.

        Args:
            eval_dataset: The evaluation dataset.
            num_samples: Number of samples to evaluate on.

        Returns:
            RAGAS evaluation dataset
        """
        logger.info(f"Creating RAGAS dataset with {num_samples} samples")
        rag_dataset = []
        for _, row in eval_dataset.sample(n=num_samples).iterrows():
            logger.debug(f"Processing question: {row['question']}")
            response = self.rag_pipeline.graph_run({"question": row["question"]})

            print(f"This is response {response}")

            rag_dataset.append(
                {
                    "question": row["question"],
                    "contexts": [
                        context.page_content for context in response["documents"]
                    ],
                    "answer": response["generation"],
                    "ground_truth": row["ground_truth"],
                }
            )

        rag_eval_dataset = Dataset.from_pandas(pd.DataFrame(rag_dataset))
        logger.info("RAGAS dataset created")
        return rag_eval_dataset

    def evaluate(
        self, eval_dataset: pd.DataFrame, num_samples: int = 100, log_wandb: bool = True
    ) -> Dict:
        """
        Evaluate the RAG pipeline using RAGAS.

        Args:
            eval_dataset: The evaluation dataset with 'question' and 'ground_truth'.
            num_samples: Number of samples to evaluate on.
            log_wandb: Whether to log results to Weights & Biases.

        Returns:
            Evaluation results
        """

        num_samples = (
            num_samples
            if len(eval_dataset.index) >= num_samples
            else len(eval_dataset.index)
        )
        logger.info(f"Evaluating on {num_samples} samples")

        ragas_dataset = self.create_ragas_dataset(eval_dataset, num_samples)

        results = evaluate(
            ragas_dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
                answer_correctness,
                answer_similarity,
                context_relevancy,
            ],
            llm=self.eval_llm,
            embeddings=self.embeddings,
        )
        logger.info(f"Evaluation results: {results}")

        if log_wandb:
            self._log_wandb(results)

        return results

    def _log_wandb(self, results: Dict):
        """Log results to Weights & Biases"""
        logger.info("Logging results to Weights & Biases")
        import wandb

        # Log metrics
        wandb.log(results)
        logger.info("Results logged to Weights & Biases")
