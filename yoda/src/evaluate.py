import argparse
import json
import os

import numpy as np
import tqdm
import yaml
from dotenv import load_dotenv
from src.search import RetrieverWithBM25
from transformers import AutoTokenizer

from utils.sambanova_endpoint import SambaNovaEndpoint

load_dotenv('.env')

llm = SambaNovaEndpoint(
    base_url=os.getenv('YODA_BASE_URL'),
    project_id=os.getenv('YODA_PROJECT_ID'),
    endpoint_id=os.getenv('FINETUNED_ENDPOINT_ID'),
    api_key=os.getenv('FINETUNED_API_KEY'),
    model_kwargs={
                'do_sample': False,
                'max_tokens_to_generate': 256
            },
            )

# Prompt prefix and postfix for llama chat model
prompt_prefix = "[INST] "
prompt_postfix = " [\INST]"


RAG_CONTEXT_TOP_K = 2


RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE_PATH = os.path.join(
    RESULTS_DIR, "generations.json")

# utils


def format_rag_prompt(question, candidate_contexts):
    prompt = 'Here is some relevant context that might assist in answering the SambaNova-related question.\n\n'
    for candidate_context, filename in candidate_contexts:
        prompt += f"Content: {candidate_context}\n\n"
    prompt += f'Answer the following SambaNova-related question: {question}\n\nAnswer:'
    return prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sn_expert_conf.yaml",
                        type=str, help="Path to config file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()


    # read config from yaml file
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    base_dir =  config['src_folder']
    data_dir = config['dest_folder']
    txt_subfolders = config["src_subfolders"]
    # collect all context articles in one folder

    LLAMA_2_70B_CHAT_PATH = config["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_70B_CHAT_PATH)

    # read in evaluation samples
    qa_eval_file = os.path.join(
        data_dir, "processed_data", "synthetic_qa_eval.jsonl")

    eval_data = []
    with open(qa_eval_file) as reader:
        for obj in reader:
            eval_data.append(eval(obj))

    context_retriever = RetrieverWithBM25(
        base_dir, tokenizer, top_k=RAG_CONTEXT_TOP_K)
    results = []
    golden_retrieved_list = []

    for i, example in tqdm.tqdm(enumerate(eval_data)):
        golden_context_filepath = example["context_filepath"]
        question = example["question"]

        # Ignorant baseline with out-of-the-box llama-7b
        prompt = example["question"]
        ignorant_pred_answer = llm(prompt=prompt_prefix + prompt + prompt_postfix)


        # Generate answer with question and golden context
        golden_context_prompt = example["question_with_context"]
        prompt_length = len(tokenizer.encode(golden_context_prompt))
        if prompt_length > 4096 - 256:
            golden_context_answer = f"golden_context_prompt too long: {prompt_length} tokens"
        else:
            golden_context_answer =llm(prompt=prompt_prefix + golden_context_prompt + prompt_postfix)

        # Generate answer with only question
        prompt = example["question"]
        pred_answer = llm(prompt=prompt_prefix + prompt + prompt_postfix)

        # Generate answer with RAG
        candidate_contexts = context_retriever(question)
        retrieved_files = [c[1] for c in candidate_contexts]
        golden_is_retrieved = False
        for f in retrieved_files:
            if f in golden_context_filepath:
                golden_is_retrieved = True
                break
        rag_prompt = format_rag_prompt(question, candidate_contexts)
        prompt_length = len(tokenizer.encode(rag_prompt))
        if prompt_length > 4096 - 256:
            rag_answer = f"rag_prompt too long: {prompt_length} tokens"
        else:
            rag_answer = llm(prompt=prompt_prefix + rag_prompt + prompt_postfix)

        result = {
            "question": question,
            "ignorant_pred_answer": ignorant_pred_answer,
            "pred_answer": pred_answer,
            "pred_answer_with_rag": rag_answer,
            "pred_answer_with_golden_context": golden_context_answer,
            "ground_truth_answer": example["answer"],
            "retrieved_files": retrieved_files,
            "golden_context_filepath": golden_context_filepath,
            "golden_context_is_retrieved": str(golden_is_retrieved)
        }
        results.append(result)
        golden_retrieved_list.append(golden_is_retrieved)

    results.append(
        {"percentage of golden context being retrieved": np.mean(golden_retrieved_list)})
    with open(RESULTS_FILE_PATH, 'w') as file:
        json.dump(results, file, indent=4)
