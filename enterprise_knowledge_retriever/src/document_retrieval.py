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

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir,"data/my-vector-db")

load_dotenv(os.path.join(repo_dir,'.env'))

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
        _info =  config["llm"]
        embedding_model_info = config["embedding_model"]
        retrieval_info = config["retrieval"]
        loaders = config["loaders"]
        
        return api_info, llm_info, embedding_model_info, retrieval_info, loaders
    
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
            search_kwargs={"score_threshold": self.retrieval_info["score_treshold"], "k": self.retrieval_info["k_retrieved_documents"]},
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


