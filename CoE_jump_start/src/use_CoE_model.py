import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from utils.sambanova_endpoint import SambaStudio, SambaStudioEmbeddings, Sambaverse
import yaml
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# Use embeddings As Part of Langchain
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(kit_dir, "config.yaml")

with open(CONFIG_PATH, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)
api_info = config["api"]
llm_info = config["llm"]
retrieval_info = config["retrieval"]

# Load environment variables from .env file
load_dotenv(os.path.join(repo_dir, ".env"))

def get_expert_val(res: Union[Dict[str, Any], str]) -> str:
    """
    Extract the expert value from the API response.

    Args:
        res (Union[Dict[str, Any], str]): The API response, either as a dictionary or a string.

    Returns:
        str: The expert value or "Generalist" if not found.
    """
    supported_experts_map = config["supported_experts_map"]
    supported_experts = list(supported_experts_map.keys())

    if isinstance(res, str):
        data = res.strip().lower()
    elif isinstance(res, dict):
        if not res or not res.get("data") or not res["data"]:
            return "Generalist"
        data = (res["data"][0].get("completion", "") or "").strip().lower()
    else:
        return "Generalist"

    expert = next((x for x in supported_experts if x in data), "Generalist")
    return supported_experts_map.get(expert, "Generalist")

def get_expert(
    input_text: str,
    do_sample: bool = False,
    max_tokens_to_generate: int = 500,
    repetition_penalty: float = 1.0,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 1.0,
    select_expert: str = "Meta-Llama-3-70B-Instruct",
    use_requests: bool = False,
    use_wrapper: bool = False,
) -> Dict[str, Any]:
    """
    Classifies the given input text into one of the predefined categories.

    Args:
        input_text (str): The input text to classify.
        do_sample (bool): Whether to sample from the model's output distribution. Default is False.
        max_tokens_to_generate (int): The maximum number of tokens to generate. Default is 500.
        repetition_penalty (float): The penalty for repeating tokens. Default is 1.0.
        temperature (float): The temperature for sampling. Default is 0.7.
        top_k (int): The number of top most likely tokens to consider. Default is 50.
        top_p (float): The cumulative probability threshold for top-p sampling. Default is 1.0.
        select_expert (str): The name of the expert model to use. Default is "Meta-Llama-3-70B-Instruct".
        use_requests (bool): Whether to use the requests library instead of SNSDK. Default is False.
        use_wrapper (bool): Whether to use the SambaStudio wrapper with langchain. Default is False.

    Returns:
        Dict[str, Any]: The response from the model.
    """
    select_expert = config["llm"]["samabaverse_select_expert"]
    prompt = config["expert_prompt"].format(input=input_text)

    inputs = json.dumps(
        {
            "conversation_id": "sambaverse-conversation-id",
            "messages": [{"message_id": 0, "role": "user", "content": input_text}],
            "prompt": prompt,
            "streaming": True
        }
    )


    tuning_params = {
        "do_sample": {"type": "bool", "value": str(do_sample).lower()},
        "max_tokens_to_generate": {"type": "int", "value": str(max_tokens_to_generate)},
        "repetition_penalty": {"type": "float", "value": str(repetition_penalty)},
        "temperature": {"type": "float", "value": str(temperature)},
        "top_k": {"type": "int", "value": str(top_k)},
        "top_p": {"type": "float", "value": str(top_p)},
        "select_expert": {"type": "str", "value": select_expert},
        "process_prompt": {"type": "bool", "value": "false"},
    }

    if use_wrapper:
        llm = SambaStudio(
            streaming=True,
            model_kwargs={
                "do_sample": do_sample,
                "temperature": temperature,
                "max_tokens_to_generate": max_tokens_to_generate,
                "select_expert": select_expert,
                "process_prompt": False,
            }
        )
        chat_prompt = ChatPromptTemplate.from_template(config["expert_prompt"])
        return llm.invoke(chat_prompt.format_prompt(input=input_text).to_string())
    elif use_requests:
        url = "{}/api/predict/generic/{}/{}"
        headers = {"Content-Type": "application/json", "key": os.getenv("SAMBASTUDIO_API_KEY")}
        data = {
            "instances": [inputs],
            "params": tuning_params,
        }

        response = requests.post(
            url.format(os.getenv("SAMBASTUDIO_BASE_URL"), os.getenv("SAMBASTUDIO_PROJECT_ID"), os.getenv("SAMBASTUDIO_ENDPOINT_ID")),
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        return response.json()
    else:
        from snsdk import SnSdk

        sdk = SnSdk(os.getenv("SAMBASTUDIO_BASE_URL"), "endpoint_secret")
        return sdk.nlp_predict(
            os.getenv("SAMBASTUDIO_PROJECT_ID"),
            os.getenv("SAMBASTUDIO_ENDPOINT_ID"),
            os.getenv("SAMBASTUDIO_API_KEY"),
            inputs,
            json.dumps(tuning_params),
        )

def run_e2e_vector_database(user_query: str, documents):
    """Run the end-to-end vector database example."""
    snsdk_model = SambaStudioEmbeddings()
    embeddings = snsdk_model

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=retrieval_info["chunk_size"],
        chunk_overlap=retrieval_info["chunk_overlap"],
    )
    split_documents = text_splitter.split_documents(documents)

    vector = Chroma.from_documents(split_documents, embeddings)

    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}"""
    )

    # Get expert
    expert_response = get_expert(user_query, use_wrapper=True)
    logger.info(f"Expert response: {expert_response}")

    expert = get_expert_val(expert_response)
    logger.info(f"Expert: {expert}")

    named_expert = config["coe_name_map"][expert]
    logger.info(f"Named expert: {named_expert}")

    llm = get_llm(named_expert)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": user_query})
    logger.info(f"Response: {response['answer']}")

    return expert, response['answer']

def run_simple_llm_invoke(user_query: str):
    """Run a simple LLM invoke with routing."""
    expert_response = get_expert(user_query, use_wrapper=True)
    logger.info(f"Expert response: {expert_response}")

    expert = get_expert_val(expert_response)
    logger.info(f"Expert: {expert}")

    named_expert = config["coe_name_map"][expert]
    logger.info(f"Named expert: {named_expert}") 

    llm = get_llm(named_expert)

    response = llm.invoke(user_query)
    logger.info(f"Response: {response}")

    return expert, response

def get_expert_only(user_query: str):
    """Get only the expert name for the given query."""
    expert_response = get_expert(user_query, use_wrapper=True)
    expert = get_expert_val(expert_response)
    logger.info(f"Expert for query '{user_query}': {expert}")

def get_llm(expert: Optional[str] = None) -> Union[Sambaverse, SambaStudio]:
    """Get the appropriate LLM based on the API configuration and expert."""
    if api_info == "sambaverse":
        return Sambaverse(
            sambaverse_model_name=llm_info["sambaverse_model_name"],
            sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
            model_kwargs={
                "do_sample": False,
                "max_tokens_to_generate": llm_info["max_tokens_to_generate"],
                "temperature": llm_info["temperature"],
                "process_prompt": True,
                "select_expert": expert or llm_info["samabaverse_select_expert"],
            },
        )
    elif api_info == "sambastudio":
        return SambaStudio(
            streaming=True,
            model_kwargs={
                "do_sample": True,
                "temperature": llm_info["temperature"],
                "max_tokens_to_generate": llm_info["max_tokens_to_generate"],
                "select_expert": expert or llm_info["samabaverse_select_expert"],
                "process_prompt": False,
            }
        )
    else:
        raise ValueError("Invalid API configuration")
    

def run_bulk_routing_eval(dataset_path: str, num_examples: int = None):
    """Run bulk routing evaluation on the given dataset."""
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    df = pd.DataFrame(data)
    categories = list(config["supported_experts_map"].keys())

    # Sample the data if num_examples is specified and less than the total dataset size
    if num_examples and num_examples < len(df):
        sampled_df = df.groupby('router_label', group_keys=False).apply(lambda x: x.sample(min(len(x), num_examples // len(categories))))
        if len(sampled_df) < num_examples:
            additional_samples = df[~df.index.isin(sampled_df.index)].sample(num_examples - len(sampled_df))
            sampled_df = pd.concat([sampled_df, additional_samples])
    else:
        sampled_df = df

    results = []
    confusion_matrix = pd.DataFrame(0, index=categories, columns=categories)
    correct_count = 0

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(kit_dir, "results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    logging.info(f"Starting evaluation of {len(sampled_df)} samples...")

    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Processing samples"):
        query = row['prompt']
        true_category = row['router_label']
        
        expert_response = get_expert(query, use_wrapper=True)
        predicted_category = get_expert_val(expert_response)
        
        # Map the predicted category to the key in supported_experts_map
        predicted_expert_key = next((k for k, v in config["supported_experts_map"].items() if v == predicted_category), "None of the above")
        
        is_correct = predicted_expert_key == true_category
        if is_correct:
            correct_count += 1
        
        result = {
            'prompt': query,
            'router_label': true_category,
            'predicted_label': predicted_expert_key,
            'is_correct': int(is_correct)
        }
        results.append(result)

        confusion_matrix.loc[true_category, predicted_expert_key] += 1

        # Log info in the terminal
        logger.info(f"Predicted: {predicted_expert_key} | True: {true_category} | {'✓' if is_correct else '✗'}")

        current_accuracy = correct_count / (len(results))
        logger.info(f"Current accuracy: {current_accuracy:.2f}")

    results_df = pd.DataFrame(results)
    
    accuracies = results_df.groupby('router_label')['is_correct'].mean().to_dict()
    logger.info(f"Final accuracies: {accuracies}")

    # Save results
    results_jsonl_path = os.path.join(results_dir, "results.jsonl")
    with open(results_jsonl_path, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')
    logger.info(f"Results saved to {results_jsonl_path}")

   # Generate and save visualizations
    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.title("Accuracy by Category")
    plt.xlabel("Category")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    accuracy_plot_path = os.path.join(results_dir, "accuracy_plot.png")
    plt.savefig(accuracy_plot_path)
    logger.info(f"Accuracy plot saved to {accuracy_plot_path}")

    # Confusion matrix with raw counts
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues')
    plt.title("Confusion Matrix (Raw Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Add total counts for each category
    for i, total in enumerate(confusion_matrix.sum(axis=1)):
        plt.text(len(categories) + 0.5, i + 0.5, f'Total: {total:.0f}', ha='left', va='center')

    confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    logger.info(f"Confusion matrix saved to {confusion_matrix_path}")

    logger.info("Evaluation complete.")
    return results_df, accuracies, confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Run SambaStudio script in different modes.")
    parser.add_argument("mode", choices=["e2e", "simple", "expert", "bulk"], default="expert",
                        help="Mode to run the script in (default: expert)")
    parser.add_argument("--query", type=str, default="What are the interest rates?",
                        help="User query to process (default: 'What are the interest rates?')")
    parser.add_argument("--dataset", type=str, help="Path to the dataset JSONL file for bulk evaluation")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to run in bulk mode (default: all)")

    args = parser.parse_args()

    if args.mode == "e2e":
        run_e2e_vector_database(args.query)
    elif args.mode == "simple":
        run_simple_llm_invoke(args.query)
    elif args.mode == "expert":
        get_expert_only(args.query)
    elif args.mode == "bulk":
        if not args.dataset:
            parser.error("The bulk mode requires a --dataset argument")
        num_examples = args.num_examples if args.num_examples is not None else float('inf')
        run_bulk_routing_eval(args.dataset, num_examples)

if __name__ == "__main__":
    main()