import argparse
import csv
import glob
import logging
import os
import pickle  # Import for saving and loading .pkl files
import sys
from pathlib import Path
from random import shuffle
from typing import Any, List, Tuple, Union

import pandas as pd
from dotenv import load_dotenv
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings import OpenAIEmbedding
from llama_index.finetuning import (
    EmbeddingQAFinetuneDataset,
    SentenceTransformersFinetuneEngine,
    generate_qa_embedding_pairs,
)
from llama_index.llms import LangChainLLM
from llama_index.node_parser import SentenceSplitter
from llama_index.schema import TextNode
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from tqdm.auto import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = current_dir
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import yaml
from langchain_community.llms.sambanova import SambaStudio

CONFIG_PATH = os.path.join(current_dir, 'config.yaml')

with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
api_info = config['api']
llm_info = config['llm']

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Process Environment Variables
load_dotenv(os.path.join(kit_dir, '.env'))


def split_files_into_datasets(
    input_data_directory: str,
    output_data_directory: str,
    file_extension: str = 'pdf',
    split_ratio: float = 0.8,
) -> Tuple[str, str]:
    """
    Split files in the given directory into training and validation datasets based on the split ratio and save them to
     the specified output directory.

    Args:
    - input_data_directory: The directory path to search for files.
    - file_extension: The type of files to search for, defaults to 'pdf'.
    - split_ratio: The ratio to split the files for training, with the rest for validation.
    - output_data_directory: The directory where processed files are stored.

    Returns:
    - Tuple containing paths to the generated CSV files for training and validation datasets.
    """

    files = glob.glob(f'{input_data_directory}/**/*.{file_extension}', recursive=True)
    shuffle(files)

    print(files)

    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]

    train_csv_path = os.path.join(output_data_directory, 'train_files.csv')
    val_csv_path = os.path.join(output_data_directory, 'val_files.csv')

    # Make output dir
    Path(output_data_directory).mkdir(parents=True, exist_ok=True)

    for dataset, path in [(train_files, train_csv_path), (val_files, val_csv_path)]:
        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['path', 'filename'])
            for file in dataset:
                writer.writerow([file, os.path.basename(file)])

    logging.info(f'Datasets generated: Train - {train_csv_path}, Val - {val_csv_path}')
    return train_csv_path, val_csv_path


# Function to instantiate SambaNova LLM
def instantiate_llm() -> LangChainLLM:
    # Initialize LLM to be used in generating queries for fine tuning.
    # Example LLM instantiation:
    # For a Sambanova LLM:

    if api_info == 'sambastudio':
        llm = SambaStudio(
            model_kwargs={
                'do_sample': True,
                'temperature': llm_info['temperature'],
                'max_tokens_to_generate': llm_info['max_tokens_to_generate'],
            }
        )

    # # Convert SN Endpoint to LangChain LLM As The Wrapper Is In Langchain
    llm = LangChainLLM(llm=llm)

    return llm


def save_nodes_to_pkl(nodes: Any, file_path: str) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(nodes, f)


def load_nodes_from_pkl(file_path: str) -> Any:
    with open(file_path, 'rb') as f:
        nodes = pickle.load(f)
    return nodes


def generate_corpus(
    input_data_directory: str,
    output_data_directory: str,
    file_extension: str,
    split_ratio: float,
) -> Tuple[Any, Any]:
    # Existing implementation to generate train_csv and val_csv
    # After generating CSVs, load the corpus and save nodes as .pkl

    if not os.path.exists(input_data_directory) or not len(os.listdir(input_data_directory)) > 0:
        raise FileNotFoundError(
            f"Directory {input_data_directory} is empty or doesn't exist. Add files to the directory and try again."
        )

    train_csv, val_csv = split_files_into_datasets(
        input_data_directory, output_data_directory, file_extension, split_ratio
    )

    # Load corpus from CSV files
    train_loader = CorpusLoader(train_csv, verbose=True)
    val_loader = CorpusLoader(val_csv, verbose=True)
    train_nodes = train_loader.load_corpus()
    val_nodes = val_loader.load_corpus()

    # Save nodes as .pkl for both train and validation datasets
    train_pkl_path = os.path.join(output_data_directory, 'train_nodes.pkl')
    val_pkl_path = os.path.join(output_data_directory, 'val_nodes.pkl')
    save_nodes_to_pkl(train_nodes, train_pkl_path)
    save_nodes_to_pkl(val_nodes, val_pkl_path)

    return train_pkl_path, val_pkl_path


def finetune(
    train_dataset_path: str,
    val_dataset_path: str,
    model_id: str,
    model_output_path: str,
    force_retrain: bool,
) -> None:
    # Check if model output path exists and is not empty
    if not force_retrain and os.path.exists(model_output_path) and os.listdir(model_output_path):
        logging.info(f'Found finetuned model at {model_output_path}. Will use this model without retraining.')
        return
    else:
        if not force_retrain:
            logging.info('No finetuned model found or directory is empty. Proceeding with finetuning.')

        train_dataset = EmbeddingQAFinetuneDataset.from_json(train_dataset_path)
        val_dataset = EmbeddingQAFinetuneDataset.from_json(val_dataset_path)
        finetune_wrapper = FinetuneEngineWrapper(train_dataset, val_dataset, model_id, model_output_path)
        finetune_wrapper.finetune()


def evaluate_all(
    val_dataset_path: str,
    model_ids: List[Union[str, OpenAIEmbedding]],
    baseline_model_id: str,
    finetuned_model_path: str,
) -> None:
    val_dataset = EmbeddingQAFinetuneDataset.from_json(val_dataset_path)
    results = []
    st_results = []
    hit_rate_results = []

    for model in model_ids:
        if isinstance(model, str):
            # Use string IDs for evaluate_st, explicitly checking for baseline and finetuned models
            if model == baseline_model_id:
                name_prefix = 'baseline'
            elif model == finetuned_model_path.strip('./'):
                name_prefix = 'finetuned'
            else:
                name_prefix = 'other'  # This can be adjusted as needed

            # Prefix with "local:" for string IDs when using evaluate
            embed_model = 'local:' + model
            eval_results = evaluate(val_dataset, embed_model, verbose=True)
            # add column for model_id + name_prefix and add it to each row
            df_eval_results = pd.DataFrame(eval_results)
            df_eval_results['model_name'] = model + '_' + name_prefix
            results.append(df_eval_results)

            # create a mew dataframe for the mean of the hit column
            hit_rate = df_eval_results['is_hit'].mean()
            # create a new dataframe for the hit rate and model name
            hit_rate_df = pd.DataFrame(
                {
                    'model_id': [model],
                    'hit_rate': [hit_rate],
                    'name_prefix': [name_prefix],
                }
            )
            hit_rate_results.append(hit_rate_df)

            eval_st_result = evaluate_st(val_dataset, model, name=name_prefix)
            st_results.append(pd.DataFrame({'model_id': [model], 'evaluation_result': [eval_st_result]}))

        elif isinstance(model, OpenAIEmbedding):
            # Use OpenAIEmbedding object directly with evaluate
            eval_results = evaluate(val_dataset, model, verbose=True)
            eval_results['model_name'] = 'openai_embedding'
            results.append(pd.DataFrame(eval_results))

    # Concatenate and save results from evaluate
    concat_results_df = pd.concat(results, ignore_index=True)
    concat_results_df.to_csv(os.path.join(kit_dir, 'results/eval_1_results.csv'), index=False)

    # Concatenate and save results from evaluate_st
    concat_st_results_df = pd.concat(st_results, ignore_index=True)
    concat_st_results_df.to_csv(
        os.path.join(kit_dir, 'results/Information-Retrieval_evaluation__results_concat.csv'), index=False
    )

    # Concatenate and save hit rate results
    concat_hit_rate_results_df = pd.concat(hit_rate_results, ignore_index=True)
    concat_hit_rate_results_df.to_csv(os.path.join(kit_dir, 'results/hit_rate_results.csv'), index=False)


class CorpusLoader:
    def __init__(self, csv_file: str, verbose: bool = False) -> None:
        self.csv_file = csv_file
        self.verbose = verbose

    def load_corpus(self) -> Any:
        files = []
        with open(self.csv_file, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                files.append(row['path'])

        reader = SimpleDirectoryReader(input_files=files)
        assert hasattr(reader, 'load_data')
        docs = reader.load_data()
        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(docs, show_progress=self.verbose)
        return nodes


class DatasetGenerator:
    def __init__(self, llm: Any, nodes: Any) -> None:
        self.llm_model = llm
        self.nodes = nodes  # Now loads nodes from a .pkl file

    def generate_dataset(self) -> Any:
        dataset = generate_qa_embedding_pairs(llm=self.llm_model, nodes=self.nodes)
        return dataset


class FinetuneEngineWrapper:
    def __init__(
        self,
        train_dataset: EmbeddingQAFinetuneDataset,
        val_dataset: EmbeddingQAFinetuneDataset,
        model_id: str,
        model_output_path: str,
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_id = model_id
        self.model_output_path = model_output_path

    def finetune(self) -> None:
        finetune_engine = SentenceTransformersFinetuneEngine(
            self.train_dataset,
            model_id=self.model_id,
            model_output_path=self.model_output_path,
            val_dataset=self.val_dataset,
        )
        finetune_engine.finetune()


def evaluate(dataset: Any, embed_model: Any, top_k: int = 5, verbose: bool = False) -> List[Any]:
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    nodes = [TextNode(id_=id_, text=text) for id_, text in dataset.corpus.items()]
    index = VectorStoreIndex(nodes, service_context=service_context, show_progress=verbose)
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(dataset.queries.items(), desc='Evaluating'):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = dataset.relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids
        eval_result = {
            'is_hit': is_hit,
            'retrieved': retrieved_ids,
            'expected': expected_id,
            'query': query_id,
        }
        eval_results.append(eval_result)
    return eval_results


def evaluate_st(dataset: Any, model_id: str, name: str) -> Any:
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, name=name)
    model = SentenceTransformer(model_id)
    output_path = os.path.join(kit_dir, 'results/')
    Path(output_path).mkdir(exist_ok=True, parents=True)
    return evaluator(model, output_path=output_path)


def parse_arguments() -> Any:
    parser = argparse.ArgumentParser(description='Script to finetune embeddings.')
    parser.add_argument(
        '--input_data_directory',
        default=os.path.join(kit_dir, 'sample_data'),
        help='Directory containing the raw files for dataset creation.',
    )
    parser.add_argument(
        '--output_data_directory',
        default=os.path.join(kit_dir, 'processed_data'),
        help='Directory where the processed files will be stored.',
    )
    parser.add_argument(
        '--train_dataset_path',
        default=None,
        help='Path to the already generated train dataset, skips dataset generation if provided.',
    )
    parser.add_argument(
        '--val_dataset_path',
        default=None,
        help='Path to the already generated validation dataset, skips dataset generation if provided.',
    )
    parser.add_argument(
        '--train_nodes_pkl_file',
        default=None,
        help='Path to the already loaded train nodes .pkl file, skips node loading if provided.',
    )
    parser.add_argument(
        '--val_nodes_pkl_file',
        default=None,
        help='Path to the already loaded validation nodes .pkl file, skips node loading if provided.',
    )
    parser.add_argument(
        '--file_extension',
        default='pdf',
        help="File extension to filter by, defaults to 'pdf'.",
    )
    parser.add_argument(
        '--split_ratio',
        type=float,
        default=0.8,
        help='Ratio for splitting data into train and validation sets.',
    )
    parser.add_argument(
        '--model_id',
        default='BAAI/bge-small-en',
        help='Model identifier for finetuning.',
    )
    parser.add_argument(
        '--model_output_path',
        default='finetuned_model',
        help='Path to save the finetuned model.',
    )
    parser.add_argument(
        '--force_retrain',
        action='store_true',
        help='Force retraining even if a finetuned model already exists.',
    )
    return parser.parse_args()


def generate_data(args: Any) -> Tuple[Any, Any]:
    # elif args.output_data_directory + train_nodes.pkl and val_nodes.pkl exists, load them:
    # be sure to use path output_data_directory + train_nodes.pkl and val_nodes.pkl to check if the files exist
    if os.path.exists(args.output_data_directory + '/' + 'train_nodes.pkl') and os.path.exists(
        args.output_data_directory + '/' + 'val_nodes.pkl'
    ):
        logging.info('Loading nodes from .pkl files.')
        train_nodes_pkl = load_nodes_from_pkl(args.output_data_directory + '/' + 'train_nodes.pkl')
        val_nodes_pkl = load_nodes_from_pkl(args.output_data_directory + '/' + 'val_nodes.pkl')
    elif args.train_nodes_pkl_file is None or args.val_nodes_pkl_file is None:
        train_nodes_pkl, val_nodes_pkl = generate_corpus(
            args.input_data_directory,
            args.output_data_directory,
            args.file_extension,
            args.split_ratio,
        )

    else:
        train_nodes_pkl = load_nodes_from_pkl(args.train_nodes_pkl_file)
        val_nodes_pkl = load_nodes_from_pkl(args.val_nodes_pkl_file)

    # Instantiate LLM
    llm = instantiate_llm()

    # check if the train_dataset_path and val_dataset_path exists, if they do, load them

    if os.path.exists(args.output_data_directory + '/' + 'train_dataset.json') and os.path.exists(
        args.output_data_directory + '/' + 'val_dataset.json'
    ):
        logging.info('Loading datasets from .json files.')
        train_dataset_path = args.output_data_directory + '/' + 'train_dataset.json'
        val_dataset_path = args.output_data_directory + '/' + 'val_dataset.json'
        logging.info(f'Train dataset path: {train_dataset_path}, Val dataset path: {val_dataset_path}')

    elif args.train_dataset_path is None or args.val_dataset_path is None:
        # Generate datasets using loaded nodes
        train_dataset_generator = DatasetGenerator(llm=llm, nodes=train_nodes_pkl)
        val_dataset_generator = DatasetGenerator(llm=llm, nodes=val_nodes_pkl)
        train_dataset = train_dataset_generator.generate_dataset()
        val_dataset = val_dataset_generator.generate_dataset()

        # Save datasets
        train_dataset_path = os.path.join(args.output_data_directory, 'train_dataset.json')
        val_dataset_path = os.path.join(args.output_data_directory, 'val_dataset.json')
        train_dataset.save_json(train_dataset_path)
        val_dataset.save_json(val_dataset_path)
    else:
        train_dataset_path = args.train_dataset_path
        val_dataset_path = args.val_dataset_path

    return train_dataset_path, val_dataset_path


def main() -> None:
    args = parse_arguments()

    train_csv, val_csv = generate_data(args)

    # Pass force_retrain flag to the finetune function
    finetune(train_csv, val_csv, args.model_id, args.model_output_path, args.force_retrain)

    model_ids = [
        args.model_id,  # Baseline model ID
        args.model_output_path,  # Finetuned model path, stripped for consistency
    ]

    evaluate_all(val_csv, model_ids, args.model_id, args.model_output_path)

    logging.info('Script finished successfully.')


if __name__ == '__main__':
    main()
