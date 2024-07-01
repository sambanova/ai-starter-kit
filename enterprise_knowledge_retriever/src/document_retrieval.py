import os
import sys
import yaml
import fitz
import torch
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.chains.base import Chain
from langchain.prompts import BasePromptTemplate,  load_prompt
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_community.llms.sambanova import SambaStudio, Sambaverse
from langchain.docstore.document import Document
import shutil
from typing import List, Dict, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from vectordb.vector_db import VectorDb
from data_extraction.src.multi_column import column_boxes

CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir,"data/my-vector-db")

load_dotenv(os.path.join(repo_dir,'.env'))


from utils.parsing.sambaparse import SambaParse, parse_doc_universal


class RetrievalQAChain(Chain):
    """class for question-answering."""
    retriever : BaseRetriever 
    rerank : bool = True
    llm : BaseLanguageModel
    qa_prompt : BasePromptTemplate
    final_k_retrieved_documents : int = 3
    
    @property
    def input_keys(self) -> List[str]:
        """Input keys.
        :meta private:
        """
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys.
        :meta private:
        """
        return ["answer", "source_documents"]  
    
    def _format_docs(self, docs):
        
        return "\n\n".join(doc.page_content for doc in docs)
    
    def rerank_docs(self, query, docs, final_k):
        
        # Lazy hardcoding for now
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
        reranker = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
        pairs = []
        for d in docs:
            pairs.append([query, d.page_content])

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            scores = (
                reranker(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )

        scores_list = scores.tolist()
        scores_sorted_idx = sorted(
            range(len(scores_list)), key=lambda k: scores_list[k], reverse=True
        )

        docs_sorted = [docs[k] for k in scores_sorted_idx]
        # docs_sorted = [docs[k] for k in scores_sorted_idx if scores_list[k]>0]
        docs_sorted = docs_sorted[:final_k]

        return docs_sorted
    
    def _call(self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        qa_chain = self.qa_prompt | self.llm | StrOutputParser()
        response = {}
        documents = self.retriever.invoke(inputs["question"])
        if self.rerank:
            documents = self.rerank_docs(inputs["question"], documents, self.final_k_retrieved_documents)
        docs = self._format_docs(documents)
        response["answer"] = qa_chain.invoke({"question": inputs["question"], "context": docs})
        response["source_documents"] = documents
        return response

class DocumentRetrieval():
    def __init__(self):
        self.vectordb = VectorDb()
        config_info = self.get_config_info()
        self.api_info =config_info[0] 
        self.llm_info =config_info[1] 
        self.embedding_model_info =config_info[2] 
        self.retrieval_info =config_info[3] 
        self.prompts = config_info[4]
        self.retriever = None
        self.llm=self.set_llm()

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
        prompts = config["prompts"]
        
        return api_info, llm_info, embedding_model_info, retrieval_info, prompts
    

    def set_llm(self):
        if self.api_info == "sambaverse":
            llm = Sambaverse(
                sambaverse_model_name=self.llm_info["sambaverse_model_name"],
                sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
                model_kwargs={
                    "do_sample": False, 
                    "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                    "temperature": self.llm_info["temperature"],
                    "select_expert": self.llm_info["select_expert"]
                }
            )
        elif self.api_info == "sambastudio":
            if self.llm_info["coe"]:
                llm = SambaStudio(
                    streaming=True,
                    model_kwargs={
                        "do_sample": False,
                        "temperature": self.llm_info["temperature"],
                        "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                        "select_expert": self.llm_info["select_expert"]
                    }
                )
            else:
                llm = SambaStudio(
                    streaming=True,
                    model_kwargs={
                        "do_sample": False,
                        "temperature": self.llm_info["temperature"],
                        "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"]
                    }
                )
        return llm
            

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
        embeddings = self.vectordb.load_embedding_model(
            type=self.embedding_model_info["type"],
            batch_size=self.embedding_model_info["batch_size"],
            coe=self.embedding_model_info["coe"],
            select_expert=self.embedding_model_info["select_expert"]
            ) 
        return embeddings  

    def create_vector_store(self, text_chunks, embeddings, output_db=None):
        vectorstore = self.vectordb.create_vector_store(text_chunks, embeddings, output_db=output_db, db_type="chroma")
        return vectorstore
    
    def load_vdb(self, db_path, embeddings):
        vectorstore = self.vectordb.load_vdb(db_path, embeddings, db_type="chroma")
        return vectorstore
    
    def init_retriever(self, vectorstore):
        if self.retrieval_info["rerank"]:
            self.retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": self.retrieval_info["score_threshold"], "k": self.retrieval_info["k_retrieved_documents"]},
            )
        else:
            self.retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": self.retrieval_info["score_threshold"], "k": self.retrieval_info["final_k_retrieved_documents"]},
            )
    
    def get_qa_retrieval_chain(self):
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
        # customprompt = load_prompt(os.path.join(kit_dir, self.prompts["qa_prompt"]))   
        # qa_chain = customprompt | self.llm | StrOutputParser()   
        
        # response = {}
        # documents = self.retriever.invoke(question)
        # if self.retrieval_info["rerank"]:
        #     documents = self.rerank_docs(question, documents, self.retrieval_info["final_k_retrieved_documents"])
        # docs = self._format_docs(documents)
        
        # response["answer"] = qa_chain.invoke({"question": question, "context": docs})
        # response["source_documents"] = documents

        retrievalQAChain = RetrievalQAChain(
            retriever=self.retriever,
            llm=self.llm,
            qa_prompt = load_prompt(os.path.join(kit_dir, self.prompts["qa_prompt"])),
            rerank = self.retrieval_info["rerank"],
            final_k_retrieved_documents = self.retrieval_info["final_k_retrieved_documents"]
        )
        return retrievalQAChain        

    def get_conversational_qa_retrieval_chain(self):
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




