import os
import yaml
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
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

    def get_llm_config(self, llm_config: Tuple[str, Dict]) -> Tuple[str, Dict]:
        llm_name, config_dict = llm_config
        full_config_dict = {
            "sambastudio_base_url": os.getenv(f"{llm_name.upper()}_BASE_URL"),
            "sambastudio_project_id": os.getenv(f"{llm_name.upper()}_PROJECT_ID"),
            "sambastudio_endpoint_id": os.getenv(f"{llm_name.upper()}_ENDPOINT_ID"),
            "sambastudio_api_key": os.getenv(f"{llm_name.upper()}_API_KEY"),
            "model_kwargs": config_dict.get("model_kwargs", {}),
        }
        return llm_name, full_config_dict

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
    def eval_dataset_ground_truth_col(self) -> str:
        return self.config["eval_dataset"]["ground_truth_col"]

    @property
    def eval_dataset_context_col(self) -> str:
        return self.config["eval_dataset"].get("context_col")

    @property
    def embedding_model_name(self) -> str:
        return self.config["embeddings"]["model_name"]

    @property
    def llm_configs(self) -> List[Tuple[str, Dict]]:
        return [(llm["name"], llm) for llm in self.config["llms"]]

    @property
    def eval_llm_configs(self) -> List[Tuple[str, Dict]]:
        return [(llm["name"], llm) for llm in self.config["eval_llms"]]

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
        eval_llms: List[Tuple[str, BaseLLM]],
        eval_embeddings: Embeddings,
        config_yaml_path: str,
    ):
        self.eval_llms = eval_llms
        self.eval_embeddings = eval_embeddings
        self.config = RAGEvalConfig(config_yaml_path)

    def create_ragas_dataset(
        self,
        eval_df: pd.DataFrame,
        llm_name: Optional[str] = None,
        answers_df: Optional[pd.DataFrame] = None,
    ) -> Dataset:
        """Create RAGAS eval dataset from question, answer, context dataframe"""
        if answers_df is not None:
            eval_df = pd.merge(
                eval_df, answers_df, on=self.config.eval_dataset_question_col
            )

        ragas_data = []
        for _, row in eval_df.iterrows():
            answer_col = (
                f"answer_{llm_name}"
                if llm_name and f"answer_{llm_name}" in eval_df.columns
                else self.config.eval_dataset_answer_col
            )
            context_col = (
                f"context_{llm_name}"
                if llm_name and f"context_{llm_name}" in eval_df.columns
                else self.config.eval_dataset_context_col
            )
            ragas_data.append(
                {
                    "question": row[self.config.eval_dataset_question_col],
                    "answer": row[answer_col],
                    "ground_truth": row[self.config.eval_dataset_ground_truth_col],
                    "contexts": (
                        [row[context_col]]
                        if context_col and not pd.isna(row[context_col])
                        else [""]
                    ),
                }
            )
        print(ragas_data)
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
            for llm_name, pipeline in answer_generation_pipelines:
                answers = []
                contexts = []
                for _, row in eval_df.iterrows():
                    query = row[self.config.eval_dataset_question_col]
                    result = pipeline.generate(query)

                    if isinstance(result["answer"], dict):
                        answer = result["answer"]["result"]
                        source_documents = result["answer"].get("source_documents", [])
                        if source_documents:
                            context = "\n\n=====\n\n".join(
                                [
                                    f"Context {i+1}:\n{doc.page_content}"
                                    for i, doc in enumerate(source_documents)
                                ]
                            )
                        else:
                            context = ""
                    else:
                        answer = result["answer"]
                        context = ""

                    answers.append(answer)
                    contexts.append(context)

                eval_df[f"answer_{llm_name}"] = answers
                eval_df[f"context_{llm_name}"] = contexts
                gen_llm_names.append(llm_name)

        metrics = [
            answer_relevancy,
            answer_correctness,
            answer_similarity,
        ]

        # Check if context columns exist for any of the generation LLMs
        context_cols_exist = any(
            f"context_{llm_name}" in eval_df.columns for llm_name in gen_llm_names
        )

        # Extend the metrics if context columns exist or if eval_dataset_context_col is provided
        if context_cols_exist or self.config.eval_dataset_context_col:
            metrics.extend(
                [
                    context_precision,
                    context_recall,
                    faithfulness,
                    context_relevancy,
                ]
            )

        results = {}
        print(eval_df)

        if gen_llm_names:
            for gen_llm_name in gen_llm_names:
                for eval_llm_name, eval_llm in self.eval_llms:
                    ragas_dataset = self.create_ragas_dataset(eval_df, gen_llm_name)

                    result = evaluate(
                        ragas_dataset.select(range(self.config.num_eval_samples)),
                        metrics=metrics,
                        llm=eval_llm,
                        embeddings=self.eval_embeddings,
                    )
                    results[f"{gen_llm_name}_{eval_llm_name}"] = result
        else:
            for eval_llm_name, eval_llm in self.eval_llms:
                ragas_dataset = self.create_ragas_dataset(eval_df, None)

                result = evaluate(
                    ragas_dataset.select(range(self.config.num_eval_samples)),
                    metrics=metrics,
                    llm=eval_llm,
                    embeddings=self.eval_embeddings,
                )
                results[eval_llm_name] = result

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


def load_pipeline(llm: Tuple[str, BaseLLM], config: RAGEvalConfig) -> Tuple[str, Any]:
    """Dynamically load answer generation pipeline from config"""
    llm_name, llm_instance = llm

    print(f"This is llm instance {llm_instance}")
    pipeline_class = config.config["pipeline"]["class"]
    pipeline_kwargs = config.config["pipeline"].get("kwargs", {})

    module_name, class_name = pipeline_class.split(".")
    module = __import__(module_name, fromlist=[class_name])
    PipelineClass = getattr(module, class_name)

    pipeline_kwargs["llm"] = llm_instance
    if "vector_db_location" in pipeline_kwargs:
        pipeline_kwargs["embeddings"] = HuggingFaceInstructEmbeddings(
            model_name=config.embedding_model_name
        )
        pipeline_kwargs["vector_db_location"] = config.vector_db_location

    print(PipelineClass(**pipeline_kwargs))
    return llm_name, PipelineClass(**pipeline_kwargs)


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
