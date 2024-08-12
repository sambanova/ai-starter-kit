import os
import sys
import logging
import yaml
import json 
from dotenv import load_dotenv

from typing import Dict, List, Union, Optional
from langchain_community.llms.sambanova import SambaStudio, Sambaverse
from langchain_community.embeddings import SambaStudioEmbeddings, HuggingFaceInstructEmbeddings
from langchain.prompts import load_prompt
from langchain_core.output_parsers import JsonOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, "../.."))
repo_dir = os.path.abspath(os.path.join(utils_dir, ".."))
sys.path.append(utils_dir)
sys.path.append(repo_dir)

from utils.model_wrappers.api_gateway import APIGateway 
from utils.model_wrappers.langchain_llms import SambaNovaFastAPI

load_dotenv(os.path.join(repo_dir, ".env"))

class SyntheticDatum(BaseModel):
    question: str = Field(description="generated question")
    answer: str = Field(description="generated answer")
    references: list[str] = Field(description="references for generated answer")
    thought: str = Field(description="thought for answer generation")

class SyntheticData(BaseModel):
    data: List[SyntheticDatum] = Field(description="synthetic data pairs")

class SyntheticDataGen():
    def __init__(self, config_file: str = None) -> None:
        if config_file is None:
            config_file = os.path.join(utils_dir, "synthetic_data_gen", "config.yaml")
        config = self.load_config(config_file) 
        self.llm_info = config["llm"]
        self.llm = self.set_llm()
        self.embedding_model_info = config["embedding_model"]
        self.embedding_model = self.set_embedding_model()
        self.prompts = config["prompts"]
        self.generation_config = config["generation"]
        self.splitting_config = config["splitting"]

    def load_config(self, config_file) -> None:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            return config
        
    def set_llm(self) -> Union[SambaStudio, Sambaverse, SambaNovaFastAPI]:
        """
        Set the LLM to use.
        sambaverse, sambastudio and fastapi endpoints implemented.
        """
        llm = APIGateway.load_llm(
            type=self.llm_info['api'],
            streaming=True,
            coe=self.llm_info["coe"],
            do_sample=self.llm_info["do_sample"],
            max_tokens_to_generate=self.llm_info["max_tokens_to_generate"],
            temperature=self.llm_info["temperature"],
            select_expert=self.llm_info["select_expert"],
            process_prompt=False,
            sambaverse_model_name=self.llm_info["sambaverse_model_name"],
        )           
        return llm
    
    def set_embedding_model(self) -> Union[SambaStudioEmbeddings, HuggingFaceInstructEmbeddings] :
        embedding_model = APIGateway.load_embedding_model(
            type=self.embedding_model_info["type"],
            batch_size=self.embedding_model_info["batch_size"],
            coe=self.embedding_model_info["coe"],
            select_expert=self.embedding_model_info["select_expert"],
        )
        return embedding_model
    
    def split_documents(
        self,
        documents: Union[list, str],
        breakpoint_threshold_amount: int = 95,
        min_doc_length: Optional[int] = None,
        ) -> List[Document]:
        if isinstance(documents, str):
            documents = [documents]        
        text_splitter = SemanticChunker(
            embeddings = self.embedding_model,
            breakpoint_threshold_type = "percentile",
            breakpoint_threshold_amount = breakpoint_threshold_amount,
            sentence_split_regex = r"(?<=[.?!])\s+",
            )
        logging.info("splitting documents")
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
        min_doc_length: Optional[int] = None
        ) -> None:
        
        if out_file is None:
            out_file = self.generation_config["output_path"]
        if amount is None:
            amount = self.generation_config["amount_per_document"]
        if include_context is None:
            include_context = self.generation_config["include_context"]
        if include_thoughts is None:
            include_thoughts = self.generation_config["include_thoughts"]
        if include_references is None:
            include_references = self.generation_config["include_references"]
        if breakpoint_threshold_amount is None:
            breakpoint_threshold_amount = self.splitting_config["breakpoint_threshold_amount"]
        if min_doc_length is None:
            min_doc_length = self.splitting_config["min_doc_length"]
        
        if isinstance(documents, str):
            documents = [documents]
          
        documents = self.split_documents(
            documents = documents,
            breakpoint_threshold_amount = breakpoint_threshold_amount,
            min_doc_length = min_doc_length
            )
        for document in documents:
            try:
                qa_pairs = self.generate_qa_pairs(
                    context = document.page_content,
                    amount=amount,
                    include_context=include_context,
                    include_thoughts=include_thoughts,
                    include_references=include_references,
                    )
                lines=self.qa_pairs_to_prompt_completion(qa_pairs) 
                self.update_jsonl(out_file, lines)
                logging.info(f'Added {amount} qa pairs to {out_file}')
            except:
                logging.warning(f'Failed to generate qa pairs for document: "{document}", skipping')
                
        self.remove_repeated_lines_in_place(out_file)
             
    def generate_qa_pairs(self, 
                          context: str,
                          amount: int = 5,
                          include_context=True,
                          include_thoughts=True,
                          include_references= True
                          ): 
        prompt = load_prompt(os.path.join(utils_dir, "synthetic_data_gen", self.prompts["generate_qa_prompt"]))
        synthetic_datum_parser=JsonOutputParser(pydantic_object=SyntheticData)
        qa_generate_chain = prompt | self.llm | synthetic_datum_parser
        qa_pairs=[]
        generation=qa_generate_chain.invoke({"document":context, "amount":amount})
        for datum in generation:
            qa_pair = {
                    "question":datum["question"],
                    "context": context if include_context else None,
                    "answer":datum["answer"],
                    "thought": datum["thought"] if include_thoughts else None,
                    "references": datum["references"] if include_references else None,
                }
            qa_pair={k:v for k, v in qa_pair.items() if v is not None}
            qa_pairs.append(qa_pair)
        return qa_pairs
  
    def update_jsonl(self, file_path, new_lines) -> None:    
        if not os.path.exists(os.path.dirname(file_path)):
            raise FileNotFoundError(f"Folder {os.path.dirname(file_path)} does not exist.")
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("")
        with open(file_path, "a") as f:
            f.write("\n".join(new_lines)+"\n")
        logging.info(f"Updated {file_path} with new lines.")
        
    def qa_pairs_to_prompt_completion(
        self,
        qa_pairs: Union[List, Dict]
        ) -> list:
        if isinstance(qa_pairs, dict):
            qa_pairs = [qa_pairs]
        lines=[]
        for pair in qa_pairs:
            line = {
                "prompt":f'{self.generation_config["system_prompt"]}{pair["question"]}',
                "completion":""
            }
            if pair.get("context"):
                line["prompt"] += f'\nContext: {pair["context"]}\n'            
            if pair.get("thought"):
                line["completion"] += f'Thought: {pair["thought"]}\n'
            line["completion"] +=  f'Answer: {pair["answer"]}\n'
            if pair.get("references"):
                line["completion"] += f'References: {pair["references"]}\n'
                
            lines.append(json.dumps(line))
        return lines
        
    def remove_repeated_lines_in_place(self, file_path: str) -> None:
        """
        Remove repeated lines from a JSON Lines file and overwrite the original file.

        Parameters:
        file_path (str): Path to the JSON Lines file.
        """
        unique_lines = set()

        # Read the input file and collect unique lines
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    json_object = json.loads(line)
                    unique_lines.add(json.dumps(json_object, sort_keys=True))
                except json.JSONDecodeError:
                    logging.info(f"Invalid JSON line skipped: {line.strip()}")

        # Write the unique lines back to the same file
        with open(file_path, 'w') as outfile:
            for unique_line in unique_lines:
                outfile.write(unique_line + '\n')
                
        logging.info(f"removed repeated lines, out file: {file_path}.")
                