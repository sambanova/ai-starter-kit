import os
import sys
import yaml
import fitz
from data_extraction.src.multi_column import column_boxes
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from utils.sambanova_endpoint import SambaNovaEndpoint, SambaverseEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, load_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir,"data/my-vector-db")

load_dotenv(os.path.join(repo_dir,'.env'))

class PDFRetrieval():
    def __init__(self):
        pass

    def get_config_info(self):
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config["api"]
        llm_info =  config["llm"]
        retreival_info = config["retrieval"]
        loader = config["loader"]
        
        return api_info, llm_info, retreival_info, loader
    
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

    def get_data_for_splitting(self, pdf_docs):
        """Extract text and metadata from all the pdf files

        Args:
            pdf_docs (list): list of pdf files

        Returns:
            list, list: list of extracted text and metadata per file
        """
        *_, loader = self.get_config_info()
        files_data = []
        files_metadatas = []
        for i in range(len(pdf_docs)):
            if loader == "unstructured":
                text, meta = self.get_pdf_text_and_metadata_unstructured(pdf_docs[i])
            elif loader == "pypdf2":
                text, meta = self.get_pdf_text_and_metadata_pypdf2(pdf_docs[i])
            elif loader == "fitz":
                text, meta = self.get_pdf_text_and_metadata_fitz(pdf_docs[i])
            files_data.extend(text)
            files_metadatas.extend(meta)
        return files_data, files_metadatas      
            
    def get_text_chunks_with_metadata(self, docs, chunk_size, chunk_overlap, meta_data):
        """Gets text chunks. .

        Args:
        doc (list): list of strings with text to split 
        chunk_size (int): chunk size in number of tokens
        chunk_overlap (int): chunk overlap in numb8er of tokens
        metadata (list, optional): list of metadata in dictionary format.

        Returns:
            list: list of documents 
        """
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

    def get_qa_retrieval_chain(self, vectorstore):
        """
        Generate a qa_retrieval chain using a language model.

        This function uses a language model, specifically a SambaNovaEndpoint, to generate a qa_retrieval chain
        based on the input vector store of text chunks.

        Parameters:
        vectorstore (Chroma): A Vector Store containing embeddings of text chunks used as context
                            for generating the conversation chain.

        Returns:
        RetrievalQA: A chain ready for QA without memory
        """
        api_info, llm_info, retrieval_info, _ = self.get_config_info()
        
        if api_info == "sambaverse":
            llm = SambaverseEndpoint(
                    sambaverse_model_name=llm_info["sambaverse_model_name"],
                    sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
                    model_kwargs={
                        "do_sample": False, 
                        "max_tokens_to_generate": llm_info["max_tokens_to_generate"],
                        "temperature": llm_info["temperature"],
                        "process_prompt": True,
                        "select_expert": llm_info["sambaverse_select_expert"]
                        #"stop_sequences": { "type":"str", "value":""},
                        # "repetition_penalty": {"type": "float", "value": "1"},
                        # "top_k": {"type": "int", "value": "50"},
                        # "top_p": {"type": "float", "value": "1"}
                    }
                )
            
        elif api_info == "sambastudio":
            llm = SambaNovaEndpoint(
                model_kwargs={
                    "do_sample": False,
                    "temperature": llm_info["temperature"],
                    "max_tokens_to_generate": llm_info["max_tokens_to_generate"],
                    #"stop_sequences": { "type":"str", "value":""},
                    # "repetition_penalty": {"type": "float", "value": "1"},
                    # "top_k": {"type": "int", "value": "50"},
                    # "top_p": {"type": "float", "value": "1"}
                }
            )
            
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": retrieval_info["score_treshold"], "k": retrieval_info["k_retrieved_documents"]},
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

        This function uses a language model, specifically a SambaNovaEndpoint, to generate a conversational_qa_retrieval chain
        based on the chat history and the relevant retrieved content from the input vector store of text chunks.

        Parameters:
        vectorstore (Chroma): A Vector Store containing embeddings of text chunks used as context
                                        for generating the conversation chain.

        Returns:
        RetrievalQA: A chain ready for QA with memory
        """


