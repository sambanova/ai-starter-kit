import os
import sys
import yaml
import fitz
import torch
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import Any, Dict, List, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.chains.base import Chain
from langchain.prompts import BasePromptTemplate,  load_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader
from langchain_community.llms.sambanova import SambaStudio, Sambaverse

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, "../../../"))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from vectordb.vector_db import VectorDb
from data_extraction.src.multi_column import column_boxes

CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir,"data/my-vector-db")

load_dotenv(os.path.join(repo_dir,'.env'))

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
        self.loaders = config_info[4]
        self.prompts = config_info[5]
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
        loaders = config["loaders"]
        prompts = config["prompts"]
        
        return api_info, llm_info, embedding_model_info, retrieval_info, loaders, prompts
    
    def set_llm(self):
        if self.api_info == "sambaverse":
            llm = Sambaverse(
                    sambaverse_model_name=self.llm_info["sambaverse_model_name"],
                    sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"), 
                    model_kwargs={
                        "do_sample": False, 
                        "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                        "temperature": self.llm_info["temperature"],
                        "process_prompt": True,
                        "select_expert": self.llm_info["select_expert"]
                        #"stop_sequences": { "type":"str", "value":""},
                        # "repetition_penalty": {"type": "float", "value": "1"},
                        # "top_k": {"type": "int", "value": "50"},
                        # "top_p": {"type": "float", "value": "1"}
                    }
                )
            
        elif self.api_info == "sambastudio":
            if self.llm_info["sambastudio_coe"] == True:
                llm = SambaStudio(
                    streaming=True,
                    model_kwargs={
                        "do_sample": False,
                        "temperature": self.llm_info["temperature"],
                        "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                        "select_expert": self.llm_info["select_expert"],
                        "process_prompt": False
                        #"stop_sequences": { "type":"str", "value":""},
                        # "repetition_penalty": {"type": "float", "value": "1"},
                        # "top_k": {"type": "int", "value": "50"},
                        # "top_p": {"type": "float", "value": "1"}
                    }
                )
            else:
                 llm = SambaStudio(
                    streaming=True,
                    model_kwargs={
                        "do_sample": False,
                        "temperature": self.llm_info["temperature"],
                        "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                        "process_prompt": False
                        #"stop_sequences": { "type":"str", "value":""},
                        # "repetition_penalty": {"type": "float", "value": "1"},
                        # "top_k": {"type": "int", "value": "50"},
                        # "top_p": {"type": "float", "value": "1"}
                    }
                )               
        
        return llm
            
    def get_pdf_text_and_metadata_pypdf2(self, pdf_doc, extra_tags=None):
        """Extract text and metadata from pdf document with pypdf2 loader

        Args:
            pdf_doc: path to pdf document

        Returns:
            list, list: list of extracted text and metadata per page
        """
        text = []
        metadata = []
        pdf_reader = PdfReader(pdf_doc)
        doc_name = pdf_doc.name
        for page in pdf_reader.pages:
            #page_number =pdf_reader.get_page_number(page)+1
            text.append(page.extract_text())
            metadata.append({"filename": doc_name})#, "page": page_number})
        return text, metadata


    def get_pdf_text_and_metadata_fitz(self, pdf_doc):    
        """Extract text and metadata from pdf document with fitz loader

        Args:
            pdf_doc: path to pdf document

        Returns:
            list, list: list of extracted text and metadata per page
        """
        text = []
        metadata = []
        temp_folder = os.path.join(kit_dir,"data/tmp")
        temp_file = os.path.join(temp_folder,"file.pdf")
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        with open(temp_file, "wb") as f:
            f.write(pdf_doc.getvalue())
        docs = fitz.open(temp_file)  
        for page, page in enumerate(docs):
            full_text = ''
            bboxes = column_boxes(page, footer_margin=100, no_image_text=True)
            for rect in bboxes:
                full_text += page.get_text(clip=rect, sort=True)
            text.append(full_text)
            metadata.append({"filename": pdf_doc.name})
        return text, metadata

    def get_pdf_text_and_metadata_unstructured(self, pdf_doc):
        """Extract text and metadata from pdf document with unstructured loader

        Args:
            pdf_doc: path to pdf document

        Returns:
            list, list: list of extracted text and metadata per page
        """
        text = []
        metadata = []
        temp_folder = os.path.join(kit_dir,"data/tmp")
        temp_file = os.path.join(temp_folder,"file.pdf")
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        with open(temp_file, "wb") as f:
            f.write(pdf_doc.getvalue())
        loader = UnstructuredPDFLoader(temp_file)
        docs_unstructured = loader.load()
        for doc in docs_unstructured:
            text.append(doc.page_content)
            metadata.append({"filename": pdf_doc.name})
        return text, metadata
    
    def get_txt_text_and_metadata(self, txt_doc):
        """Extract text and metadata from txt document with txt loader

        Args:
            txt_doc: path to txt document

        Returns:
            list, list: list of extracted text and metadata per page
            
        """
        text = []
        metadata = []
        temp_folder = os.path.join(kit_dir,"data/tmp")
        temp_file = os.path.join(temp_folder,"file.txt")
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        with open(temp_file, "wb") as f:
            f.write(txt_doc.getvalue())
            
        loader = TextLoader(temp_file)
        docs_text_loader = loader.load()
        
        for doc in docs_text_loader:
            text.append(doc.page_content)
            metadata.append({"filename": txt_doc.name})
        return text, metadata

    def get_data_for_splitting(self, docs):
        """Extract text and metadata from all the pdf files

        Args:
            pdf_docs (list): list of pdf files

        Returns:
            list, list: list of extracted text and metadata per file
        """
        files_data = []
        files_metadatas = []
        for i in range(len(docs)):
            if docs[i].name.endswith(".pdf"):
                if self.loaders["pdf"] == "unstructured":
                    text, meta = self.get_pdf_text_and_metadata_unstructured(docs[i])
                elif self.loaders["pdf"] == "pypdf2":
                    text, meta = self.get_pdf_text_and_metadata_pypdf2(docs[i])
                elif self.loaders["pdf"] == "fitz":
                    text, meta = self.get_pdf_text_and_metadata_fitz(docs[i])
                else:
                    raise ValueError(f"{self.loaders['pdf']} is not a valid pdf loader")
            elif docs[i].name.endswith(".txt"):
                if self.loaders["txt"] == "text_loader":
                    text, meta = self.get_txt_text_and_metadata(docs[i])
                else:
                    raise ValueError(f"{self.loaders['txt']} is not a valid txt loader")
            files_data.extend(text)
            files_metadatas.extend(meta)
        return files_data, files_metadatas      
            
    def get_text_chunks_with_metadata(self, docs, meta_data):
        """Gets text chunks. .

        Args:
        doc (list): list of strings with text to split 
        chunk_size (int): chunk size in number of tokens
        chunk_overlap (int): chunk overlap in number of tokens
        metadata (list, optional): list of metadata in dictionary format.

        Returns:
            list: list of documents 
        """
        chunk_size = self.retrieval_info["chunk_size"]
        chunk_overlap = self.retrieval_info["chunk_overlap"]
        chunks_list = []
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
            )
        for doc, meta in zip(docs, meta_data):
            chunks = text_splitter.create_documents([doc], [meta])
            for chunk in chunks:
                chunk.page_content = f"Source: {meta['filename'].split('.')[0]}, Text: \n{chunk.page_content}\n"
                chunks_list.append(chunk)
        return chunks_list

    def load_embedding_model(self):
        embeddings = self.vectordb.load_embedding_model(type=self.embedding_model_info) 
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


