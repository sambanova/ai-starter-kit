import os
import yaml
import pandas as pd
from typing import List, Dict, Optional, Tuple
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_community.llms.sambanova import SambaStudio
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval import evaluate
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

class SambaStudioLLM(DeepEvalBaseLLM):
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = SambaStudio(**self.config)
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        template =  PromptTemplate.from_template("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are JSON generator<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>machine-readable JSON<|end_header_id|>\n\n")
        chain = (
        {"prompt": RunnablePassthrough()}    
        | template
        | model)
        return chain.invoke(prompt)


    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


    def get_model_name(self):
        select_expert = self.config.get('model_kwargs', {}).get('select_expert', 'Unknown')
        return f"SambaStudio-{select_expert}"

class RAGEvalConfig:
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
            "streaming": True,
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
    def eval_dataset_context_col(self) -> Optional[str]:
        return self.config["eval_dataset"].get("context_col")

    @property
    def eval_llm_configs(self) -> List[Tuple[str, Dict]]:
        return [(llm["name"], llm) for llm in self.config["eval_llms"]]

class RAGEvaluator:
    def __init__(self, config_yaml_path: str):
        self.config = RAGEvalConfig(config_yaml_path)
        self.eval_llms = []
        for llm_name, llm_config in self.config.eval_llm_configs:
            _, full_config = self.config.get_llm_config((llm_name, llm_config))
            self.eval_llms.append((llm_name, SambaStudioLLM(full_config)))

    def create_test_cases(self, df: pd.DataFrame) -> List[LLMTestCase]:
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
                retrieval_context=context
            )
            test_cases.append(test_case)
        return test_cases

    def create_metrics(self, eval_llm: SambaStudioLLM) -> List:
        metrics = [AnswerRelevancyMetric(threshold=0.7, model=eval_llm,verbose_mode=True)]
        
        # Only add context-dependent metrics if context is present
        if self.config.eval_dataset_context_col:
            context_metrics = [
                FaithfulnessMetric(threshold=0.7, model=eval_llm),
                ContextualRecallMetric(threshold=0.7, model=eval_llm),
                ContextualPrecisionMetric(threshold=0.7, model=eval_llm)
            ]
            metrics.extend(context_metrics)
        else:
            logging.warning("Context column not found in config. Skipping context-dependent metrics.")
        
        return metrics

    def evaluate(self, eval_df: pd.DataFrame):
        self.test_cases = self.create_test_cases(eval_df)
        dataset = EvaluationDataset(test_cases=self.test_cases)
        
        results = {}
        for llm_name, eval_llm in self.eval_llms:
            metrics = self.create_metrics(eval_llm)
            result = evaluate(dataset, metrics)
            results[llm_name] = result
        
        return results

def main():
    config_path = "config.yaml"
    eval_csv_path = "data/test.csv"
    
    evaluator = RAGEvaluator(config_path)
    eval_df = pd.read_csv(eval_csv_path)
    
    results = evaluator.evaluate(eval_df)
    
    for llm_name, result in results.items():
        logging.info(f"Results for {llm_name}:")
        logging.info(result)

if __name__ == "__main__":
    main()