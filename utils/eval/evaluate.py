import argparse
import logging

import pandas as pd
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms.sambanova import SambaStudio
from rag_eval import RAGEvalConfig, RAGEvaluator, load_pipeline


def run_evaluation(config_path: str, eval_csv: str, use_generation: bool) -> None:
    config = RAGEvalConfig(config_path)

    eval_llms = [
        (llm_name, SambaStudio(**llm_config))
        for conf in config.eval_llm_configs
        for llm_name, llm_config in [config.get_llm_config(conf)]
    ]
    # eval_llms = [
    #     ('open_ai',ChatOpenAI())
    # ]

    eval_embeddings = HuggingFaceInstructEmbeddings(model_name=config.embedding_model_name)

    evaluator = RAGEvaluator(
        eval_llms=eval_llms,
        eval_embeddings=eval_embeddings,
        config_yaml_path=config_path,
    )

    eval_df = pd.read_csv(eval_csv)

    if use_generation:
        logging.info('Running evaluation with generation pipeline')
        pipelines = [
            load_pipeline((llm_name, SambaStudio(**llm_config)), config)
            for llm_name, llm_config in [config.get_llm_config(conf) for conf in config.llm_configs]
        ]
        results = evaluator.evaluate(eval_df, pipelines)
    else:
        logging.info('Running evaluation without generation pipeline')
        results = evaluator.evaluate(eval_df)

    logging.info(f'Evaluation results: {results}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--eval_csv', required=True, help='Path to the evaluation CSV file')
    parser.add_argument('--generation', action='store_true', help='Use generation pipeline')
    args = parser.parse_args()

    run_evaluation(args.config, args.eval_csv, args.generation)
