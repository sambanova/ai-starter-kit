import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv
from langchain.prompts import load_prompt
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms.sambanova import SambaStudio
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '../..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)

from utils.model_wrappers.api_gateway import APIGateway
from utils.model_wrappers.langchain_embeddings import SambaStudioEmbeddings
from utils.model_wrappers.langchain_llms import SambaNovaCloud

load_dotenv(os.path.join(repo_dir, '.env'))


class SyntheticDatum(BaseModel):
    """Model of a synthetic generated datum"""

    question: str = Field(description='generated question')
    answer: str = Field(description='generated answer')
    references: list[str] = Field(description='references for generated answer')
    thought: str = Field(description='thought for answer generation')


class SyntheticData(BaseModel):
    """Model of a synthetic data generation"""

    data: List[SyntheticDatum] = Field(description='synthetic data pairs')


class SyntheticDataGen:
    """Class for generating synthetic data"""

    def __init__(self, config_file: str = None) -> None:
        """
        Initialize SyntheticDataGen class with configuration parameters.

        Parameters:
        config_file (str): Path to the configuration YAML file. If not provided, a default path is used.

        Returns:
        None
        """
        if config_file is None:
            config_file = os.path.join(utils_dir, 'synthetic_data_gen', 'config.yaml')
        # Load configuration parameters from YAML file
        config = self.load_config(config_file)
        self.llm_info = config['llm']
        # Set LLM given llm configuration in config file
        self.llm = self.set_llm()
        self.embedding_model_info = config['embedding_model']
        # Set embedding model given the embedding model configuration in config file
        self.embedding_model = self.set_embedding_model()
        self.prompts = config['prompts']
        self.generation_config = config['generation']
        self.splitting_config = config['splitting']

    def load_config(self, config_file: str) -> None:
        """
        Load configuration parameters from a YAML file.

        Parameters:
        config_file (str): Path to the YAML configuration file.

        Returns:
        None
        """
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            return config

    def set_llm(self) -> Union[SambaStudio, SambaNovaCloud]:
        """
        Set the LLM to use for generation.

        Parameters:
        None

        Returns:
        SambaStudio, or SambaNovaCloud instance
        """
        llm = APIGateway.load_llm(
            type=self.llm_info['api'],
            streaming=True,
            coe=self.llm_info['coe'],
            do_sample=self.llm_info['do_sample'],
            max_tokens_to_generate=self.llm_info['max_tokens_to_generate'],
            temperature=self.llm_info['temperature'],
            select_expert=self.llm_info['select_expert'],
            process_prompt=False,
        )
        return llm

    def set_embedding_model(self) -> Union[SambaStudioEmbeddings, HuggingFaceInstructEmbeddings]:
        embedding_model = APIGateway.load_embedding_model(
            type=self.embedding_model_info['type'],
            batch_size=self.embedding_model_info['batch_size'],
            coe=self.embedding_model_info['coe'],
            select_expert=self.embedding_model_info['select_expert'],
        )
        return embedding_model

    def split_documents(
        self,
        documents: Union[list, str],
        breakpoint_threshold_amount: int = 95,
        min_doc_length: Optional[int] = None,
    ) -> List[Document]:
        """
        Split large documents into smaller chunks based on semantic similarity.

        Parameters:
        documents (Union[list, str]): A single string document or a list of string documents to be split.
        breakpoint_threshold_amount (int, optional): The threshold for determining the breakpoint in the document.
            Defaults to 95.
        min_doc_length (Optional[int], optional): The minimum length for a document chunk.
            If a chunk is shorter than this, it will be discarded. Defaults to None.

        Returns:
        List[Document]: A list of Document objects representing the splitted documents.
        """
        # TODO add recursive character splitting first
        if isinstance(documents, str):
            documents = [documents]
        text_splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            sentence_split_regex=r'(?<=[.?!])\s+',
        )
        logging.info('splitting documents')
        new_docs = text_splitter.create_documents(documents)
        if min_doc_length is not None:
            new_docs = [doc for doc in new_docs if len(doc.page_content) >= min_doc_length]
        return new_docs

    def generate_synthetic_data(
        self,
        documents: Union[list, str],
        amount: Optional[int] = None,
        include_context: Optional[bool] = None,
        include_thoughts: Optional[bool] = None,
        include_references: Optional[bool] = None,
        out_file: Optional[str] = None,
        breakpoint_threshold_amount: Optional[int] = None,
        min_doc_length: Optional[int] = None,
    ) -> None:
        """
        Generate synthetic dataset in jsonl file for a given list of documents for LLM fine-tuning.

        Parameters:
        documents (Union[list, str]): A single string document or a list of string documents.
        amount (Optional[int], optional): The number of question answer pairs to generate per document.
        include_context (Optional[bool], optional): Whether to include the context in the question.
        include_thoughts (Optional[bool], optional): Whether to include the rezoning thought in the answer.
        include_references (Optional[bool], optional): Whether to include the references in the answer.
        out_file (Optional[str], optional): The path to save the generated question answer pairs in JSONL format.
        breakpoint_threshold_amount (Optional[int], optional): The threshold for determining the breakpoint
            for splitting the original document.
        min_doc_length (Optional[int], optional): The minimum length for a document after splitting.

        Returns:
            None
        """
        if out_file is None:
            out_file = self.generation_config['output_path']
        if amount is None:
            amount = self.generation_config['amount_per_document']
        if include_context is None:
            include_context = self.generation_config['include_context']
        if include_thoughts is None:
            include_thoughts = self.generation_config['include_thoughts']
        if include_references is None:
            include_references = self.generation_config['include_references']
        if breakpoint_threshold_amount is None:
            breakpoint_threshold_amount = self.splitting_config['breakpoint_threshold_amount']
        if min_doc_length is None:
            min_doc_length = self.splitting_config['min_doc_length']

        if isinstance(documents, str):
            documents = [documents]

        documents = self.split_documents(
            documents=documents, breakpoint_threshold_amount=breakpoint_threshold_amount, min_doc_length=min_doc_length
        )
        for document in documents:
            try:
                qa_pairs = self.generate_qa_pairs(
                    context=document.page_content,
                    amount=amount,
                    include_context=include_context,
                    include_thoughts=include_thoughts,
                    include_references=include_references,
                )
                lines = self.qa_pairs_to_prompt_completion(qa_pairs)
                self.update_jsonl(out_file, lines)
                logging.info(f'Added {amount} qa pairs to {out_file}')
            except Exception as e:
                logging.warning(f'Failed to generate qa pairs, error: \n {e} \n for document: "{document}", \nskipping')

        self.remove_repeated_lines_in_place(out_file)

    def generate_qa_pairs(
        self,
        context: str,
        amount: int = 5,
        include_context: bool = True,
        include_thoughts: bool = True,
        include_references: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate question answer (context, thought, references) pairs for a given context using the LLM.

        Parameters:
        context (str): The context to generate question answer pairs from.
        amount (int, optional): The number of question answer pairs to generate. Defaults to 5.
        include_context (bool, optional): Whether to include the context in the result. Defaults to True.
        include_thoughts (bool, optional): Whether to include the rezoning thought in result. Defaults to True.
        include_references (bool, optional): Whether to include the references in the result. Defaults to True.

        Returns:
        dict: A dictionary containing the generated question answer pairs.
        """
        prompt = load_prompt(os.path.join(utils_dir, 'synthetic_data_gen', self.prompts['generate_qa_prompt']))
        synthetic_datum_parser = JsonOutputParser(pydantic_object=SyntheticData)
        qa_generate_chain = prompt | self.llm | synthetic_datum_parser
        qa_pairs = []
        generation = qa_generate_chain.invoke({'document': context, 'amount': amount})
        for datum in generation:
            qa_pair = {
                'question': datum['question'],
                'context': context if include_context else None,
                'answer': datum['answer'],
                'thought': datum['thought'] if include_thoughts else None,
                'references': datum['references'] if include_references else None,
            }
            qa_pair = {k: v for k, v in qa_pair.items() if v is not None}
            qa_pairs.append(qa_pair)
        return qa_pairs

    def update_jsonl(self, file_path: str, new_lines: str) -> None:
        """
        Update an existing jsonl file with new lines

        Parameters:
        file_path (str): Path to the JSON Lines file.
        new_lines (list): List of new lines to be added to the file.

        Returns:
        None
        """

        if not os.path.exists(os.path.dirname(file_path)):
            raise FileNotFoundError(f'Folder {os.path.dirname(file_path)} does not exist.')
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write('')
        with open(file_path, 'a') as f:
            f.write('\n'.join(new_lines) + '\n')
        logging.info(f'Updated {file_path} with new lines.')

    def qa_pairs_to_prompt_completion(self, qa_pairs: Union[List, Dict]) -> list:
        """
        Convert a list of question answer (context, thought, references) pairs into a
            list of prompt completion lines for jsonl file.

        Parameters:
        qa_pairs (Union[List, Dict]): A single question answer (context, thought, references) pair
            or a list of question answer (context, thought, references) pairs.

        Returns:
        list: A list of prompt-completion lines for the given question answer pairs.
        """
        if isinstance(qa_pairs, dict):
            qa_pairs = [qa_pairs]
        lines = []
        for pair in qa_pairs:
            line = {'prompt': f'{self.generation_config["system_prompt"]}{pair["question"]}', 'completion': ''}
            if pair.get('context'):
                line['prompt'] += f'\nContext: {pair["context"]}\n'
            if pair.get('thought'):
                line['completion'] += f'Thought: {pair["thought"]}\n'
            line['completion'] += f'Answer: {pair["answer"]}\n'
            if pair.get('references'):
                line['completion'] += f'References: {pair["references"]}\n'

            lines.append(json.dumps(line))
        return lines

    def remove_repeated_lines_in_place(self, file_path: str) -> None:
        """
        Remove repeated lines from a JSON Lines file and overwrite the original file.

        Parameters:
        file_path (str): Path to the JSON Lines file.

        Returns:
        None
        """
        unique_lines = set()

        # Read the input file and collect unique lines
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    json_object = json.loads(line)
                    unique_lines.add(json.dumps(json_object, sort_keys=True))
                except json.JSONDecodeError:
                    logging.info(f'Invalid JSON line skipped: {line.strip()}')

        # Write the unique lines back to the same file
        with open(file_path, 'w') as outfile:
            for unique_line in unique_lines:
                outfile.write(unique_line + '\n')

        logging.info(f'removed repeated lines, out file: {file_path}.')
