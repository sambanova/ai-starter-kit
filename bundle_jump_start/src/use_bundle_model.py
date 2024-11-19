import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import yaml
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.llms import LLM
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)
from utils.model_wrappers.api_gateway import APIGateway

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
api_info = config['api']
llm_info = config['llm']
embedding_model_info = config['embedding_model']
retrieval_info = config['retrieval']

load_dotenv(os.path.join(repo_dir, '.env'))


def get_expert_val(res: Union[Dict[str, Any], str]) -> Any:
    supported_experts_map = config['supported_experts_map']
    supported_experts = list(supported_experts_map.keys())

    if isinstance(res, str):
        data = res.strip().lower()
    elif isinstance(res, dict):
        if not res or not res.get('data') or not res['data']:
            return 'Generalist'
        data = (res['data'][0].get('completion', '') or '').strip().lower()
    else:
        return 'Generalist'

    expert = next((x for x in supported_experts if x in data), 'Generalist')
    return supported_experts_map.get(expert, 'Generalist')


def get_expert(
    input_text: str,
    do_sample: bool = False,
    max_tokens_to_generate: int = 500,
    repetition_penalty: float = 1.0,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 1.0,
    select_expert: str = 'Meta-Llama-3-70B-Instruct',
    use_requests: bool = False,
    use_wrapper: bool = False,
) -> Any:
    select_expert = config['llm']['select_expert']
    prompt = config['expert_prompt'].format(input=input_text)

    inputs = json.dumps(
        {
            'conversation_id': 'sambacloud-conversation-id',
            'messages': [{'message_id': 0, 'role': 'user', 'content': input_text}],
            'prompt': prompt,
            'streaming': True,
        }
    )

    tuning_params = {
        'do_sample': {'type': 'bool', 'value': str(do_sample).lower()},
        'max_tokens_to_generate': {'type': 'int', 'value': str(max_tokens_to_generate)},
        'repetition_penalty': {'type': 'float', 'value': str(repetition_penalty)},
        'temperature': {'type': 'float', 'value': str(temperature)},
        'top_k': {'type': 'int', 'value': str(top_k)},
        'top_p': {'type': 'float', 'value': str(top_p)},
        'select_expert': {'type': 'str', 'value': select_expert},
        'process_prompt': {'type': 'bool', 'value': 'false'},
    }

    if use_wrapper:
        llm = get_llm()
        chat_prompt = ChatPromptTemplate.from_template(config['expert_prompt'])
        return llm.invoke(chat_prompt.format_prompt(input=input_text).to_string())
    elif use_requests:
        url = '{}/api/predict/generic/{}/{}'
        headers = {
            'Content-Type': 'application/json',
            'key': os.getenv('SAMBASTUDIO_API_KEY'),
        }
        data = {
            'instances': [inputs],
            'params': tuning_params,
        }

        response = requests.post(
            url.format(
                os.getenv('SAMBASTUDIO_BASE_URL'),
                os.getenv('SAMBASTUDIO_PROJECT_ID'),
                os.getenv('SAMBASTUDIO_ENDPOINT_ID'),
            ),
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        return response.json()
    else:
        from snsdk import SnSdk  # type: ignore

        sdk = SnSdk(os.getenv('SAMBASTUDIO_BASE_URL'), 'endpoint_secret')
        return sdk.nlp_predict(
            os.getenv('SAMBASTUDIO_PROJECT_ID'),
            os.getenv('SAMBASTUDIO_ENDPOINT_ID'),
            os.getenv('SAMBASTUDIO_API_KEY'),
            inputs,
            json.dumps(tuning_params),
        )


def run_e2e_vector_database(user_query: str, documents: Any) -> Tuple[str, Any]:
    embeddings = APIGateway.load_embedding_model(
        type=embedding_model_info['type'],
        batch_size=embedding_model_info['batch_size'],
        bundle=embedding_model_info['bundle'],
        select_expert=embedding_model_info['select_expert'],
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=retrieval_info['chunk_size'],
        chunk_overlap=retrieval_info['chunk_overlap'],
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

    expert_response = get_expert(user_query, use_wrapper=True)
    logger.info(f'Expert response: {expert_response}')

    expert = get_expert_val(expert_response)
    logger.info(f'Expert: {expert}')

    named_expert = config['bundle_name_map'][expert]
    logger.info(f'Named expert: {named_expert}')

    llm = get_llm(named_expert)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': user_query})
    logger.info(f"Response: {response['answer']}")

    return expert, response['answer']


def run_simple_llm_invoke(user_query: str) -> Tuple[str, str]:
    router_response = get_expert(user_query, use_wrapper=True)
    expert = get_expert_val(router_response)
    logger.info(f'Expert: {expert}')

    named_expert = config['bundle_name_map'][expert]
    logger.info(f'Named expert: {named_expert}')

    llm = get_llm(named_expert)

    response = llm.invoke(user_query)
    logger.info(f'Response: {response}')

    return expert, response


def get_expert_only(user_query: str) -> Any:
    expert_response = get_expert(user_query, use_wrapper=True)
    expert = get_expert_val(expert_response)
    logger.info(f"Expert for query '{user_query}': {expert}")
    return expert_response


def get_llm(expert: Optional[str] = None) -> LLM:
    return APIGateway.load_llm(
        type=api_info,
        streaming=True,
        bundle=llm_info['bundle'],
        do_sample=llm_info['do_sample'],
        max_tokens_to_generate=llm_info['max_tokens_to_generate'],
        temperature=llm_info['temperature'],
        select_expert=expert or llm_info['select_expert'],
        process_prompt=False,
    )


def run_bulk_routing_eval(
    dataset_path: str, num_examples: Optional[int | float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    categories = list(config['supported_experts_map'].keys())

    if num_examples and num_examples < len(df):
        sampled_df = df.groupby('router_label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), num_examples // len(categories)))
        )
        if len(sampled_df) < num_examples:
            additional_samples = df[~df.index.isin(sampled_df.index)].sample(num_examples - len(sampled_df))
            sampled_df = pd.concat([sampled_df, additional_samples])
    else:
        sampled_df = df

    results = []
    confusion_matrix = pd.DataFrame(0, index=categories, columns=categories)
    correct_count = 0

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(kit_dir, 'results', timestamp)
    os.makedirs(results_dir, exist_ok=True)

    logging.info(f'Starting evaluation of {len(sampled_df)} samples...')

    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc='Processing samples'):
        query = row['prompt']
        true_category = row['router_label']

        expert_response = get_expert(query, use_wrapper=True)
        predicted_category = get_expert_val(expert_response)

        predicted_expert_key = next(
            (k for k, v in config['supported_experts_map'].items() if v == predicted_category),
            'None of the above',
        )

        is_correct = predicted_expert_key == true_category
        if is_correct:
            correct_count += 1

        result = {
            'prompt': query,
            'router_label': true_category,
            'predicted_label': predicted_expert_key,
            'is_correct': int(is_correct),
        }
        results.append(result)

        confusion_matrix.loc[true_category, predicted_expert_key] += 1

        logger.info(f"Predicted: {predicted_expert_key} | True: {true_category} | {'✓' if is_correct else '✗'}")

        current_accuracy = correct_count / (len(results))
        logger.info(f'Current accuracy: {current_accuracy:.2f}')

    results_df = pd.DataFrame(results)

    accuracies = results_df.groupby('router_label')['is_correct'].mean().to_dict()
    logger.info(f'Final accuracies: {accuracies}')

    results_jsonl_path = os.path.join(results_dir, 'results.jsonl')
    with open(results_jsonl_path, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')
    logger.info(f'Results saved to {results_jsonl_path}')

    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.title('Accuracy by Category')
    plt.xlabel('Category')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    accuracy_plot_path = os.path.join(results_dir, 'accuracy_plot.png')
    plt.savefig(accuracy_plot_path)
    logger.info(f'Accuracy plot saved to {accuracy_plot_path}')

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues')
    plt.title('Confusion Matrix (Raw Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    for i, total in enumerate(confusion_matrix.sum(axis=1)):
        plt.text(
            len(categories) + 0.5,
            i + 0.5,
            f'Total: {total:.0f}',
            ha='left',
            va='center',
        )

    confusion_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    logger.info(f'Confusion matrix saved to {confusion_matrix_path}')

    logger.info('Evaluation complete.')
    return results_df, accuracies, confusion_matrix


def load_config(config_path: str) -> Any:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_expert_datasets(eval_data_dir: str, num_examples: int, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    all_examples = []
    files = [f for f in os.listdir(eval_data_dir) if f.endswith('.jsonl')]
    samples_per_file = num_examples // len(files)
    extra_samples = num_examples % len(files)

    for i, filename in enumerate(files):
        expert_name = filename.split('-')[1]
        filepath = os.path.join(eval_data_dir, filename)
        with open(filepath, 'r') as f:
            dataset = [json.loads(line) for line in f]

        samples_to_take = samples_per_file + (1 if i < extra_samples else 0)
        if samples_to_take < len(dataset):
            dataset = random.sample(dataset, samples_to_take)

        for item in dataset:
            item['true_expert'] = expert_name

        all_examples.extend(dataset)

    random.shuffle(all_examples)
    return all_examples


def calculate_similarity(str1: str, str2: str) -> float:
    return SequenceMatcher(None, str1, str2).ratio()


def visualize_similarity(similarity: float) -> str:
    if similarity < 0.3:
        return '🔴 Poor'
    elif similarity < 0.7:
        return '🟠 Medium'
    else:
        return '🟢 Good'


def evaluate_experts(examples: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    for item in tqdm(examples, desc='Evaluating examples'):
        prompt = item['prompt']
        ground_truth = item['completion']
        true_expert = item['true_expert']

        # Get the router response (expert only)
        predicted_expert = get_expert_val(get_expert(prompt, use_wrapper=True))

        # Get the actual response from the expert
        named_expert = config['bundle_name_map'][f'{predicted_expert}']
        _, expert_response = run_simple_llm_invoke(prompt)

        # Calculate similarity
        similarity = calculate_similarity(ground_truth, expert_response)

        # Determine if the category prediction is correct
        is_correct = predicted_expert == f'{true_expert.capitalize()} expert'

        # Log the result
        logger.info(f"Predicted: {predicted_expert} | True: {true_expert} | {'✓' if is_correct else '✗'}")
        logger.info(f'Similarity score: {similarity:.2f} {visualize_similarity(similarity)}')
        logger.info(f'Ground truth: {ground_truth[:100]}...')
        logger.info(f'Model output: {expert_response[:100]}...')
        logger.info('-' * 50)

        result = {
            'prompt': prompt,
            'true_expert': true_expert,
            'predicted_expert': predicted_expert,
            'is_correct_category': is_correct,
            'ground_truth': ground_truth,
            'expert_response': expert_response,
            'similarity': similarity,
        }
        results.append(result)

    return results


def save_evaluation_results(results: List[Dict[str, Any]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results as JSONL
    output_file = os.path.join(output_dir, 'evaluation_results.jsonl')
    with open(output_file, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')

    # Create a summary DataFrame
    df = pd.DataFrame(results)
    summary = df.groupby('true_expert').agg({'is_correct_category': 'mean', 'similarity': 'mean'}).reset_index()
    summary.columns = ['Expert', 'Category Accuracy', 'Average Similarity']

    # Save summary as CSV
    summary_file = os.path.join(output_dir, 'evaluation_summary.csv')
    summary.to_csv(summary_file, index=False)

    # Print detailed results
    print('\nDetailed Evaluation Results:')
    for result in results:
        print(
            f"""Expert: {result['true_expert']} | Predicted:
             {result['predicted_expert']} | {'✓' if result['is_correct_category'] else '✗'}"""
        )
        print(f"Similarity: {result['similarity']:.2f} {visualize_similarity(result['similarity'])}")
        print(f"Prompt: {result['prompt'][:50]}...")
        print(f"Ground truth: {result['ground_truth'][:50]}...")
        print(f"Model output: {result['expert_response'][:50]}...")
        print('-' * 50)

    # Print summary
    print('\nEvaluation Results Summary:')
    print(summary.to_string(index=False))
    print(f'\nDetailed results saved to {output_file}')
    print(f'Summary saved to {summary_file}')

    # Visualize results
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Expert', y='Category Accuracy', data=summary)
    plt.title('Category Prediction Accuracy by Expert')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    accuracy_plot_path = os.path.join(output_dir, 'category_accuracy_plot.png')
    plt.savefig(accuracy_plot_path)
    print(f'Category accuracy plot saved to {accuracy_plot_path}')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Expert', y='Average Similarity', data=summary)
    plt.title('Average Response Similarity by Expert')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    similarity_plot_path = os.path.join(output_dir, 'similarity_plot.png')
    plt.savefig(similarity_plot_path)
    print(f'Similarity plot saved to {similarity_plot_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run SambaStudio script in different modes.')
    parser.add_argument(
        'mode',
        choices=['e2e', 'simple', 'expert', 'bulk', 'evaluate_experts'],
        default='expert',
        help='Mode to run the script in (default: expert)',
    )
    parser.add_argument(
        '--query',
        type=str,
        default='What are the interest rates?',
        help="User query to process (default: 'What are the interest rates?')",
    )
    parser.add_argument('--dataset', type=str, help='Path to the dataset JSONL file for bulk evaluation')
    parser.add_argument(
        '--eval_data_dir',
        type=str,
        help='Directory containing expert evaluation datasets',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results',
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=None,
        help='Total number of examples to evaluate across all experts (default: all)',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the configuration file',
    )

    args = parser.parse_args()

    if args.mode == 'e2e':
        documents: List[Any] = []  # You need to implement document loading here
        expert, response = run_e2e_vector_database(args.query, documents)
        print(f'Expert: {expert}')
        print(f'Response: {response}')
    elif args.mode == 'simple':
        expert, response = run_simple_llm_invoke(args.query)
        print(f'Expert: {expert}')
        print(f'Response: {response}')
    elif args.mode == 'expert':
        expert_response = get_expert_only(args.query)
        print(f'Expert response: {expert_response}')
    elif args.mode == 'bulk':
        if not args.dataset:
            parser.error('The bulk mode requires a --dataset argument')
        num_examples = args.num_examples if args.num_examples is not None else float('inf')
        results_df, accuracies, confusion_matrix = run_bulk_routing_eval(args.dataset, num_examples)
        print('Accuracies by category:')
        print(accuracies)
        print('\nConfusion Matrix:')
        print(confusion_matrix)
    elif args.mode == 'evaluate_experts':
        if not args.eval_data_dir:
            parser.error('The evaluate_experts mode requires an --eval_data_dir argument')
        expert_datasets = load_expert_datasets(args.eval_data_dir, args.num_examples, config)
        results = evaluate_experts(expert_datasets, config)
        save_evaluation_results(results, args.output_dir)


if __name__ == '__main__':
    main()
