import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(repo_dir,'.env'))

from vectordb.vector_db import VectorDb
from utils.sambanova_endpoint import SambaNovaEndpoint, SambaverseEndpoint

import yaml
import nest_asyncio
from typing import Tuple      
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate, load_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS 
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
from utils.sambanova_endpoint import SambaNovaEndpoint

nest_asyncio.apply()

DATA_DIRECTORY = os.path.join(kit_dir,"data")
CONFIG_PATH = os.path.join(kit_dir,'config.yaml')

class WebCrawlingRetrieval:
    
    def __init__(self, documents=None, config=None):
        if config is None:
            config = {}
        self.documents = documents
        self.config = config
        self.vectordb = VectorDb()
     
    @staticmethod   
    def _get_config_info():
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config["api"]
        llm_info =  config["llm"]
        retreival_info = config["retrieval"]
        web_crawling_params = config["web_crawling"]
        extra_loaders = config["extra_loaders"]
        
        
        return api_info, llm_info, retreival_info, web_crawling_params, extra_loaders
            
    @staticmethod
    def load_remote_pdf(url):
        """
        Load PDF files from the given URL.
        Args:
            url (str): URL to load pdf document from.
        Returns:
            list: A list of loaded pdf documents.
        """
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()
        return docs
    
    @staticmethod
    def load_htmls(urls, extra_loaders=None):
        """
        Load HTML documents from the given URLs.
        Args:
            urls (list): A list of URLs to load HTML documents from.
        Returns:
            list: A list of loaded HTML documents.
        """
        if extra_loaders is None:
            extra_loaders = []
        docs=[]
        for url in urls:
            if url.endswith(".pdf"):
                if "pdf" in extra_loaders:
                    docs.extend(WebCrawlingRetrieval.load_remote_pdf(url))
                else:
                    continue
            else:
                loader = AsyncHtmlLoader(url, verify_ssl=False)
                docs.extend(loader.load())
        return docs
    
    @staticmethod
    def link_filter(all_links, excluded_links):
        """
        Filters a list of links based on a list of excluded links.
        Args:
            all_links (List[str]): A list of links to filter.
            excluded_links (List[str]): A list of excluded links.
        Returns:
            Set[str]: A list of filtered links.
        """
        clean_excluded_links=set()
        for excluded_link in excluded_links:
            parsed_link=urlparse(excluded_link)
            clean_excluded_links.add(parsed_link.netloc + parsed_link.path)
        filtered_links = set()
        for link in all_links:
            # Check if the link contains any of the excluded links
            if not any(excluded_link in link for excluded_link in clean_excluded_links):
                filtered_links.add(link)
        return filtered_links
    @staticmethod
    def find_links(docs, excluded_links=None):
        """
        Find links in the given HTML documents, excluding specified links and not text content links.
        Args:
            docs (list): A list of documents with html content to search for links.
            excluded_links (list, optional): A list of links to exclude from the search. Defaults to None.
        Returns:
            set: A set of unique links found in the HTML documents.
        """
        if excluded_links is None:
            excluded_links = []
        all_links = set()  
        excluded_link_suffixes = {".ico", ".svg", ".jpg", ".png", ".jpeg", ".", ".docx", ".xls", ".xlsx"}
        for doc in docs:
            page_content = doc.page_content
            base_url = doc.metadata["source"]
            #excluded_links.append(base_url) #enable this to prevent to crail insithe the same root website
            soup = BeautifulSoup(page_content, 'html.parser')
            # Identify the main content section (customize based on HTML structure)
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main_content:
                links = main_content.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    # Check if the link is not an anchor link and not in the excluded links or suffixes
                    if (
                        not href.startswith(('#', 'data:', 'javascript:')) and
                        not any(href.endswith(suffix) for suffix in excluded_link_suffixes)
                    ):
                        full_url, _ = urldefrag(urljoin(base_url, href))
                        all_links.add(full_url)  
        all_links=WebCrawlingRetrieval.link_filter(all_links, set(excluded_links))
        return all_links
    
    @staticmethod
    def clean_docs(docs):
        """
        Clean the given HTML documents by transforming them into plain text.
        Args:
            docs (list): A list of langchain documents with html content to clean.
        Returns:
            list: A list of cleaned plain text documents.
        """
        html2text_transformer = Html2TextTransformer()
        docs=html2text_transformer.transform_documents(documents=docs)
        return docs
    
    @staticmethod
    def web_crawl(urls, excluded_links=None, depth = 1):
        """
        Perform web crawling, retrieve and clean HTML documents from the given URLs, with specified depth of exploration.
        Args:
            urls (list): A list of URLs to crawl.
            excluded_links (list, optional): A list of links to exclude from crawling. Defaults to None.
            depth (int, optional): The depth of crawling, determining how many layers of internal links to explore. Defaults to 1
        Returns:
            tuple: A tuple containing the langchain documents (list) and the scrapped URLs (list).
        """
        *_, web_crawling_params, extra_loaders = WebCrawlingRetrieval._get_config_info()
        if excluded_links is None:
            excluded_links = []
        excluded_links.extend(["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "telagram.me", "reddit.com", "whatsapp.com", "wa.me"])
        if depth > web_crawling_params["max_depth"]: #Max depth change with precausion number of sites grow exponentially
            depth = web_crawling_params["max_depth"]
        scrapped_urls=[]
        raw_docs=[]
        for _ in range(depth):
            
            if len(scrapped_urls)+len(urls) >= web_crawling_params["max_scraped_websites"]:
                urls = list(urls)[:web_crawling_params["max_scraped_websites"]-len(scrapped_urls)]
                urls = set(urls)
                
            scraped_docs = WebCrawlingRetrieval.load_htmls(urls, extra_loaders)
            scrapped_urls.extend(urls)
            urls=WebCrawlingRetrieval.find_links(scraped_docs, excluded_links)
            
            excluded_links.extend(scrapped_urls)
            raw_docs.extend(scraped_docs)
            
            if len(scrapped_urls) == web_crawling_params["max_scraped_websites"]:
                break
            
        docs=WebCrawlingRetrieval.clean_docs(raw_docs)
        return docs, scrapped_urls


    def init_llm_model(self) -> None:
        """Initializes the LLM endpoint
        """
        api_info, llm_info, *_ = WebCrawlingRetrieval._get_config_info()
        if api_info=="sambaverse":
            self.llm = SambaverseEndpoint(
                sambaverse_model_name=llm_info["sambaverse_model_name"],
                sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
                model_kwargs={
                    "do_sample": True, 
                    "max_tokens_to_generate": llm_info["max_tokens_to_generate"],
                    "temperature": llm_info["temperature"],
                    "process_prompt": True,
                    "select_expert": llm_info["sambaverse_select_expert"],
                    #"stop_sequences": { "type":"str", "value":""},
                    # "repetition_penalty": {"type": "float", "value": "1"},
                    # "top_k": {"type": "int", "value": "50"},
                    # "top_p": {"type": "float", "value": "1"}
                }
            )
        elif api_info=="sambastudio":
            self.llm = SambaNovaEndpoint(
                model_kwargs={
                    "do_sample": True, 
                    "temperature": llm_info["temperature"],
                    "max_tokens_to_generate": llm_info["max_tokens_to_generate"],
                    #"stop_sequences": { "type":"str", "value":""},
                    # "repetition_penalty": {"type": "float", "value": "1"},
                    # "top_k": {"type": "int", "value": "50"},
                    # "top_p": {"type": "float", "value": "1"}
                }
            ) 
    
    def create_load_vector_store(self, force_reload: bool = False, update: bool = False):
        
        *_, retrieval_info, _, _ = WebCrawlingRetrieval._get_config_info()
        
        
        persist_directory = self.config.get("persist_directory", "NoneDirectory")
        
        self.embeddings = self.vectordb.load_embedding_model()
        
        if os.path.exists(persist_directory) and not force_reload and not update:
            self.vector_store = self.vectordb.load_vdb(persist_directory, self.embeddings, db_type = retrieval_info["db_type"])
        
        elif os.path.exists(persist_directory) and update:
            self.chunks = self.vectordb.get_text_chunks(self.documents , retrieval_info["chunk_size"], retrieval_info["chunk_overlap"])
            self.vector_store = self.vectordb.load_vdb(persist_directory, self.embeddings, db_type = retrieval_info["db_type"])
            self.vector_store = self.vectordb.update_vdb(self.chunks, self.embeddings, retrieval_info["db_type"], persist_directory)
            
        else:
            self.chunks = self.vectordb.get_text_chunks(self.documents , retrieval_info["chunk_size"], retrieval_info["chunk_overlap"])
            self.vector_store = self.vectordb.create_vector_store(self.chunks, self.embeddings, retrieval_info["db_type"], None)
    
    def create_and_save_local(self, input_directory, persist_directory, update=False):
        
        *_, retrieval_info, _, _ = WebCrawlingRetrieval._get_config_info()
        
        self.chunks = self.vectordb.get_text_chunks(self.documents , retrieval_info["chunk_size"], retrieval_info["chunk_overlap"])
        self.embeddings = self.vectordb.load_embedding_model()
        if update:
            self.config["update"]=True
            self.vector_store = self.vectordb.update_vdb(self.chunks, self.embeddings, retrieval_info["db_type"], input_directory, persist_directory)

        else:
            self.vector_store = self.vectordb.create_vector_store(self.chunks, self.embeddings, retrieval_info["db_type"], persist_directory)

       
    def retrieval_qa_chain(self):
        *_, retrieval_info, _, _ = WebCrawlingRetrieval._get_config_info()
        prompt = load_prompt(os.path.join(kit_dir,"prompts/llama7b-web_crwling_data_retriever.yaml"))
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": retrieval_info["score_treshold"], "k": retrieval_info["k_retrieved_documents"]},
        )
        self.qa_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True,
            input_key="question",
            output_key="answer",
            prompt=prompt
        )
        return self.qa_chain

