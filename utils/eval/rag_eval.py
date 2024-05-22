import os
import yaml
import pandas as pd
from typing import List, Dict, Optional
from datasets import Dataset, load_dataset
from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
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
import wandb
import argparse
from langchain_community.llms.sambanova import SambaStudio, Sambaverse
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


class RAGEvalConfig:
    """Configuration for RAG Evaluation"""

    def __init__(self, config_yaml_path: str):
        with open(config_yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_llm_config(self, llm_config: Dict) -> Dict:
        print(llm_config)
        llm_name = llm_config["name"]
        return {
            "sambastudio_base_url": os.getenv(f"{llm_name.upper()}_BASE_URL"),
            "sambastudio_project_id": os.getenv(f"{llm_name.upper()}_PROJECT_ID"),
            "sambastudio_endpoint_id": os.getenv(f"{llm_name.upper()}_ENDPOINT_ID"),
            "sambastudio_api_key": os.getenv(f"{llm_name.upper()}_API_KEY"),
            **llm_config.get("model_kwargs", {}),
        }

    @property
    def eval_dataset_path(self) -> str:
        return self.config["eval_dataset"].get("path")

    @property
    def eval_dataset_question_col(self) -> str:
        return self.config["eval_dataset"]["question_col"]

    @property
    def eval_dataset_answer_col(self) -> str:
        return self.config["eval_dataset"]["answer_col"]

    @property
    def eval_dataset_context_col(self) -> str:
        return self.config["eval_dataset"].get("context_col")

    @property
    def embedding_model_name(self) -> str:
        return self.config["embeddings"]["model_name"]

    @property
    def llm_configs(self) -> List[Dict]:
        return [self.get_llm_config(llm) for llm in self.config["llms"]]

    @property
    def eval_llm_configs(self) -> List[Dict]:
        return [self.get_llm_config(llm) for llm in self.config["eval_llms"]]

    def print_config_keys(self):
        print("Configuration Keys:")
        for key, value in self.config.items():
            print(f"{key}:")
            if isinstance(value, list):
                for item in value:
                    print(f"  - {item}")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"  {value}")

    @property
    def vector_db_location(self) -> str:
        return self.config["vector_db"]["location"]

    @property
    def num_eval_samples(self) -> int:
        return self.config["evaluation"]["num_samples"]

    @property
    def log_wandb(self) -> bool:
        return self.config["evaluation"]["log_wandb"]

    @property
    def wandb_project(self) -> str:
        return self.config["evaluation"]["project_name"]

    @property
    def wandb_eval_name(self) -> str:
        return self.config["evaluation"]["eval_name"]

    @property
    def eval_methodology(self) -> str:
        return self.config["evaluation"]["methodology"]

    @property
    def user_provided_answers_path(self) -> str:
        return self.config.get("user_provided_answers", None)

    @property
    def hf_dataset_name(self) -> str:
        return self.config["eval_dataset"].get("hf_dataset_name")

    def get_flattened_config(self) -> Dict:
        """Return flattened config dict for logging"""
        flat_config = {}
        for k, v in self.config.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat_config[f"{k}.{kk}"] = vv
            else:
                flat_config[k] = v
        return flat_config


class RAGEvaluator:
    """RAG Evaluation Module"""

    def __init__(
        self,
        eval_llms: List[BaseLLM],
        eval_embeddings: Embeddings,
        config_yaml_path: str,
    ):
        self.eval_llms = eval_llms
        self.eval_embeddings = eval_embeddings
        self.config = RAGEvalConfig(config_yaml_path)

    def create_ragas_dataset(
        self,
        eval_df: pd.DataFrame,
        answers_df: Optional[pd.DataFrame] = None,
    ) -> Dataset:
        """Create RAGAS eval dataset from question, answer, context dataframe"""
        if answers_df is not None:
            eval_df = pd.merge(
                eval_df, answers_df, on=self.config.eval_dataset_question_col
            )

        ragas_data = []
        for _, row in eval_df.iterrows():
            ragas_data.append(
                {
                    "question": row[self.config.eval_dataset_question_col],
                    "answer": row["answer"],
                    "ground_truth": row[self.config.eval_dataset_answer_col],
                    "contexts": (
                        [row[self.config.eval_dataset_context_col]]
                        if self.config.eval_dataset_context_col
                        else []
                    ),
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

        if answer_generation_pipelines:
            for pipeline in answer_generation_pipelines:
                answers = []
                for _, row in eval_df.iterrows():
                    query = row[self.config.eval_dataset_question_col]
                    answer = pipeline.generate(query)
                    answers.append(answer["answer"])

                eval_df[f"answer_{pipeline.llm.model_name}"] = answers

        metrics = [
            answer_relevancy,
            answer_correctness,
            answer_similarity,
        ]

        if self.config.eval_dataset_context_col:
            metrics.extend(
                [
                    context_precision,
                    context_recall,
                    faithfulness,
                    context_relevancy,
                ]
            )

        results = {}
        for eval_llm in self.eval_llms:
            ragas_dataset = self.create_ragas_dataset(eval_df)

            result = evaluate(
                ragas_dataset.select(range(self.config.num_eval_samples)),
                metrics=metrics,
                llm=eval_llm,
                embeddings=self.eval_embeddings,
            )
            results[eval_llm.model_name] = result

        if self.config.log_wandb:
            self._log_wandb(results)

        return results

    def _log_wandb(self, results: Dict):
        """Log eval results and config to Weights & Biases"""
        run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_eval_name,
            config=self.config.get_flattened_config(),
        )
        wandb.log(results)
        run.finish()


def load_pipeline(llm: BaseLLM, config: RAGEvalConfig):
    """Dynamically load answer generation pipeline from config"""
    if not config.config.get("pipeline"):
        return None

    pipeline_class = config.config["pipeline"]["class"]
    pipeline_kwargs = config.config["pipeline"].get("kwargs", {})

    module_name, class_name = pipeline_class.split(".")
    module = __import__(module_name, fromlist=[class_name])
    PipelineClass = getattr(module, class_name)

    pipeline_kwargs["llm"] = llm
    if config.vector_db_location:
        pipeline_kwargs["embeddings"] = HuggingFaceInstructEmbeddings(
            model_name=config.embedding_model_name
        )
        pipeline_kwargs["vector_db_location"] = config.vector_db_location

    return PipelineClass(**pipeline_kwargs)


def load_eval_dataframe(config: RAGEvalConfig):
    if config.hf_dataset_name:
        dataset = load_dataset(config.hf_dataset_name)
        eval_df = dataset["test"].to_pandas()
    else:
        eval_df = pd.read_csv(config.eval_dataset_path)

    return eval_df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = RAGEvalConfig(args.config)

    eval_llms = [
        SambaStudio(**config.get_llm_config(conf)) for conf in config.eval_llm_configs
    ]

    eval_embeddings = HuggingFaceInstructEmbeddings(
        model_name=config.embedding_model_name
    )

    evaluator = RAGEvaluator(
        eval_llms=eval_llms,
        eval_embeddings=eval_embeddings,
        config_yaml_path=args.config,
    )

    eval_df = load_eval_dataframe(config)

    answer_generation_pipelines = []
    for llm_config in config.llm_configs:
        llm = SambaStudio(**config.get_llm_config(llm_config))
        pipeline = load_pipeline(llm, config)
        answer_generation_pipelines.append(pipeline)

    results = evaluator.evaluate(eval_df, answer_generation_pipelines)
    print(results)


if __name__ == "__main__":
    main()
