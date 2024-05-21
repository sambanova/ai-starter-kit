import os
import sys
import yaml
import fitz
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from vectordb.vector_db import VectorDb
from data_extraction.src.multi_column import column_boxes
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, load_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader
from langchain_community.llms.sambanova import SambaStudio, Sambaverse
from langchain.docstore.document import Document
import shutil
from typing import List, Dict, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir,"data/my-vector-db")

load_dotenv(os.path.join(repo_dir,'.env'))

from utils.parsing.sambaparse import SambaParse, parse_doc_universal


class DocumentRetrieval():
    def __init__(self):
        self.vectordb = VectorDb()
        config_info = self.get_config_info()
        self.api_info =config_info[0] 
        self.llm_info =config_info[1] 
        self.embedding_model_info =config_info[2] 
        self.retrieval_info =config_info[3] 
        self.loaders = config_info[4] 

    def get_config_info(self):
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config["api"]
        llm_info =  config["llm"]
        embedding_model_info = config["embedding_model"]
        retrieval_info = config["retrieval"]
        loaders = config["loaders"]
        
        return api_info, llm_info, embedding_model_info, retrieval_info, loaders
    

    def parse_doc(self, docs: List, additional_metadata: Optional[Dict] = None) -> List[Document]:
        """
        Parse the uploaded documents and return a list of LangChain documents.

        Args:
            docs (List[UploadFile]): A list of uploaded files.
            additional_metadata (Optional[Dict], optional): Additional metadata to include in the processed documents.
                Defaults to an empty dictionary.

        Returns:
            List[Document]: A list of LangChain documents.
        """
        if additional_metadata is None:
            additional_metadata = {}

        # Create the data/tmp folder if it doesn't exist
        temp_folder = os.path.join(kit_dir, "data/tmp")
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        else:
            # If there are already files there, delete them
            for filename in os.listdir(temp_folder):
                file_path = os.path.join(temp_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

        # Save all selected files to the tmp dir with their file names
        for doc in docs:
            temp_file = os.path.join(temp_folder, doc.name)
            with open(temp_file, "wb") as f:
                f.write(doc.getvalue())

        # Pass in the temp folder for processing into the parse_doc_universal function
        _, _, langchain_docs = parse_doc_universal(doc=temp_folder, additional_metadata=additional_metadata)
        return langchain_docs


    def load_embedding_model(self):
        embeddings = self.vectordb.load_embedding_model(type=self.embedding_model_info) 
        return embeddings  

    def create_vector_store(self, text_chunks, embeddings, output_db=None):
        vectorstore = self.vectordb.create_vector_store(text_chunks, embeddings, output_db=output_db, db_type=self.retrieval_info["db_type"])
        return vectorstore
    
    def load_vdb(self, db_path, embeddings):
        vectorstore = self.vectordb.load_vdb(db_path, embeddings, db_type=self.retrieval_info["db_type"])
        return vectorstore
    
    def get_qa_retrieval_chain(self, vectorstore):
        """
        Generate a qa_retrieval chain using a language model.

        This function uses a language model, specifically a SambaNova LLM, to generate a qa_retrieval chain
        based on the input vector store of text chunks.

        Parameters:
        vectorstore (Chroma): A Vector Store containing embeddings of text chunks used as context
                            for generating the conversation chain.

        Returns:
        RetrievalQA: A chain ready for QA without memory
        """
        
        if self.api_info == "sambaverse":
            llm = Sambaverse(
                    sambaverse_model_name=self.llm_info["sambaverse_model_name"],
                    sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
                    model_kwargs={
                        "do_sample": False, 
                        "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                        "temperature": self.llm_info["temperature"],
                        "process_prompt": True,
                        "select_expert": self.llm_info["sambaverse_select_expert"]
                        #"stop_sequences": { "type":"str", "value":""},
                        # "repetition_penalty": {"type": "float", "value": "1"},
                        # "top_k": {"type": "int", "value": "50"},
                        # "top_p": {"type": "float", "value": "1"}
                    }
                )
            
        elif self.api_info == "sambastudio":
            llm = SambaStudio(
                model_kwargs={
                    "do_sample": False,
                    "temperature": self.llm_info["temperature"],
                    "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                    #"stop_sequences": { "type":"str", "value":""},
                    # "repetition_penalty": {"type": "float", "value": "1"},
                    # "top_k": {"type": "int", "value": "50"},
                    # "top_p": {"type": "float", "value": "1"}
                }
            )
            
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": self.retrieval_info["score_threshold"], "k": self.retrieval_info["k_retrieved_documents"]},
        )
        qa_chain = RetrievalQA.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            input_key="question",
            output_key="answer",
        )
        
        customprompt = load_prompt(os.path.join(kit_dir, "prompts/llama7b-knowledge_retriever-custom_qa_prompt.yaml"))

        ## Inject custom prompt
        qa_chain.combine_documents_chain.llm_chain.prompt = customprompt
        return qa_chain


    def get_conversational_qa_retrieval_chain(self, vectorstore):
        """
        Generate a conversational retrieval qa chain using a language model.

        This function uses a language model, specifically a SambaNova LLM, to generate a conversational_qa_retrieval chain
        based on the chat history and the relevant retrieved content from the input vector store of text chunks.

        Parameters:
        vectorstore (Chroma): A Vector Store containing embeddings of text chunks used as context
                                        for generating the conversation chain.

        Returns:
        RetrievalQA: A chain ready for QA with memory
        """




