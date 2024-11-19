import argparse
import glob
import json
import logging
import os
import random
import re
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

from llama_index import SimpleDirectoryReader
from llama_index.llms import LangChainLLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import MetadataMode
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.model_wrappers.api_gateway import APIGateway

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CorpusLoader:
    """
    Loads and splits a corpus of documents from a given directory into training and validation sets.

    Attributes:
        directory (str): Directory path where the documents are located.
        val_ratio (float): Ratio of documents to be used for validation.
        train_files (List[str]): List of training file paths.
        val_files (List[str]): List of validation file paths.
    """

    def __init__(self, directory: str, val_ratio: float = 0.2) -> None:
        """
        Initializes the CorpusLoader with a directory and validation ratio.
        """
        self.directory = directory
        self.val_ratio = val_ratio
        self.train_files, self.val_files = self.split_train_val()

    def split_train_val(self) -> Tuple[List[str], List[str]]:
        """
        Splits the documents into training and validation sets based on the validation ratio.

        Returns:
            Tuple containing lists of training and validation file paths.
        """
        pdf_files = glob.glob(f'{self.directory}/*.pdf')
        random.shuffle(pdf_files)
        split_index = int(len(pdf_files) * (1 - self.val_ratio))
        return pdf_files[:split_index], pdf_files[split_index:]

    def load_corpus(self, files: List[str]) -> Dict[str, str]:
        """
        Loads the corpus from the specified files.

        Args:
            files (List[str]): List of file paths to load.

        Returns:
            Dictionary with node IDs as keys and document content as values.
        """
        logging.info(f'Loading {len(files)} documents...')
        reader = SimpleDirectoryReader(input_files=files)
        docs = reader.load_data()
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(docs, show_progress=False)
        return {node.node_id: node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes}

    def save_corpus(self, corpus: Dict[str, str], file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_dict_safely(corpus, file_path)


class QueryGenerator:
    """
    Generates synthetic queries and relevant documents from a corpus using a Language Model.

    Attributes:
        llm (LangChainLLM): Language Model for generating queries.
    """

    def __init__(self, llm: Any, model_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the QueryGenerator with a language model and optional model arguments.
        """
        self.llm = llm

    def generate_queries(
        self, corpus: Dict[str, str], num_questions_per_chunk: int = 2, verbose: bool = False
    ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Generates queries based on the corpus using the language model.

        Args:
            corpus (Dict[str, str]): Corpus with node IDs as keys and document content as values.
            num_questions_per_chunk (int): Number of questions to generate per document chunk.
            verbose (bool): Whether to output progress information.

        Returns:
            Tuple containing dictionaries for queries and relevant documents.
        """
        logging.info('Generating queries...')
        prompt_template = """\
        Context information is below.

        ---------------------
        {context_str}
        ---------------------

        Given the context information and not prior knowledge.
        generate only questions based on the below query.

        You are a Teacher/ Professor. Your task is to setup \
        {num_questions_per_chunk} questions for an upcoming \
        quiz/examination. The questions should be diverse in nature \
        across the document. Restrict the questions to the \
        context information provided."
        """
        queries, relevant_docs = {}, {}
        for node_id, text in tqdm(corpus.items(), disable=not verbose, desc='Processing corpus'):
            query = prompt_template.format(context_str=text, num_questions_per_chunk=num_questions_per_chunk)
            response = self.llm.complete(query)
            result = str(response).strip().split('\n')
            questions = [re.sub(r'^\d+[\).\s]', '', question).strip() for question in result if len(question) > 0]
            for question in questions:
                question_id = str(uuid.uuid4())
                queries[question_id] = question
                relevant_docs[question_id] = [node_id]
        return queries, relevant_docs


def save_dict_safely(data: Dict[str, Any], file_path: str) -> None:
    """
    Saves a dictionary to a file in a way that avoids memory issues with large datasets.

    Args:
        data (Dict): Dictionary to save.
        file_path (str): Path to the file where the dictionary will be saved.
    """
    logging.info(f'Saving data to {file_path}...')
    with open(file_path, 'w') as f:
        f.write('{')
        for i, (key, value) in enumerate(tqdm(data.items(), desc='Saving data')):
            if i > 0:
                f.write(',')
            f.write(json.dumps(key) + ':' + json.dumps(value))
        f.write('}')


def main() -> None:
    """
    Main function to run the script. Parses command-line arguments and generates a synthetic dataset.
    """
    parser = argparse.ArgumentParser(description='Generate a synthetic dataset of (query, relevant document) pairs.')
    parser.add_argument(
        '--data_directory',
        type=str,
        default='./sample_data/embed_tuning/datasheets',
        help='Directory containing PDF files.',
    )
    parser.add_argument(
        '--train_corpus_output_path',
        type=str,
        default='./data/train_corpus.json',
        help='Output path for the training corpus.',
    )
    parser.add_argument(
        '--val_corpus_output_path',
        type=str,
        default='./data/val_corpus.json',
        help='Output path for the validation corpus.',
    )
    parser.add_argument(
        '--train_output_path',
        type=str,
        default='./data/train_dataset.json',
        help='Output path for the training dataset.',
    )
    parser.add_argument(
        '--val_output_path', type=str, default='./data/val_dataset.json', help='Output path for the validation dataset.'
    )
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of data to be used for validation.')
    args = parser.parse_args()

    corpus_loader = CorpusLoader(directory=args.data_directory, val_ratio=args.val_ratio)

    # Load Corpus
    train_corpus = corpus_loader.load_corpus(corpus_loader.train_files)
    val_corpus = corpus_loader.load_corpus(corpus_loader.val_files)

    # Save Corpus to Disk
    corpus_loader.save_corpus(train_corpus, args.train_corpus_output_path)
    corpus_loader.save_corpus(val_corpus, args.val_corpus_output_path)

    # Example LLM instantiation:
    # For a Sambanova LLM:
    base_url = 'YOUR_BASE_URL'
    project_id = 'YOUR_PROJECT_ID'
    endpoint_id = 'YOUR_ENDPOINT_ID'
    api_key = 'YOUR_API_KEY'

    llm = APIGateway.load_llm(
        type='sambastudio',
        streaming=True,
        bundle=True,
        do_sample=True,
        max_tokens_to_generate=512,
        temperature=0.01,
        select_expert='Meta-Llama-3-70B-Instruct-4096',
        process_prompt=False,
        sambanova_api_key=api_key,
    )

    # Convert SN Endpoint to LangChain LLM As The Wrapper Is In Langchain
    llm = LangChainLLM(llm=llm)

    # For OpenAI:
    # llm = OpenAI(model='gpt-3.5-turbo')  # This line remains commented in the script for instructional purposes

    # Use The LLM to Generate the Queries for the Corpus
    query_generator = QueryGenerator(llm=llm)

    train_queries, train_relevant_docs = query_generator.generate_queries(train_corpus, verbose=True)
    val_queries, val_relevant_docs = query_generator.generate_queries(val_corpus, verbose=True)

    train_dataset = {'queries': train_queries, 'corpus': train_corpus, 'relevant_docs': train_relevant_docs}
    val_dataset = {'queries': val_queries, 'corpus': val_corpus, 'relevant_docs': val_relevant_docs}

    save_dict_safely(train_dataset, args.train_output_path)
    save_dict_safely(val_dataset, args.val_output_path)


if __name__ == '__main__':
    main()
