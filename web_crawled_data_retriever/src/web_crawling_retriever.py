import os
import sys
sys.path.append("../")

from dotenv import load_dotenv
load_dotenv('../export.env')

from vectordb.vector_db import VectorDb
from utils.sambanova_endpoint import SambaNovaEndpoint

import nest_asyncio
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

DATA_DIRECTORY = "../data"

LAST_N_DOCUMENTS = 1
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS_TO_GENERATE = 500
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 240
DB_TYPE = "faiss"

K_RETRIEVED_DOCUMENTS = 4
SCORE_TRESHOLD = 0.5

class WebCrawlingRetrieval:
    
    def __init__(self, documents=None, config=None):
        if config is None:
            config = {}
        self.documents = documents
        self.config = config
        self.vectordb = VectorDb()
        
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
    def load_htmls(urls):
        """
        Load HTML documents from the given URLs.
        Args:
            urls (list): A list of URLs to load HTML documents from.
        Returns:
            list: A list of loaded HTML documents.
        """
        docs=[]
        for url in urls:
            if url.endswith(".pdf"):
                docs.extend(WebCrawlingRetrieval.load_remote_pdf(url))
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
        excluded_link_suffixes = {".ico", ".svg", ".jpg", ".png", ".jpeg", ".", ".docx", ".xls", ".xlsx", ".pdf"}
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
        if excluded_links is None:
            excluded_links = []
        excluded_links.extend(["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "telagram.me", "reddit.com", "whatsapp.com", "wa.me"])
        if depth > 3: #Max depth change with precausin number of sites grow exponentially
            depth = 3
        scrapped_urls=[]
        raw_docs=[]
        for _ in range(depth):
            scraped_docs = WebCrawlingRetrieval.load_htmls(urls)
            scrapped_urls.extend(urls)
            urls=WebCrawlingRetrieval.find_links(scraped_docs, excluded_links)
            
            excluded_links.extend(scrapped_urls)
            raw_docs.extend(scraped_docs)
        docs=WebCrawlingRetrieval.clean_docs(scraped_docs)
        return docs, scrapped_urls


    def init_llm_model(self) -> None:
        """Initializes the LLM endpoint
        """
        self.llm = SambaNovaEndpoint(
            model_kwargs={
                "do_sample": True, 
                "temperature": LLM_TEMPERATURE,
                "max_tokens_to_generate": LLM_MAX_TOKENS_TO_GENERATE,
            }
        ) 
    
    def create_load_vector_store(self, force_reload: bool = False, update: bool = False):
        
        persist_directory = self.config.get("persist_directory", "NoneDirectory")
        
        self.embeddings = self.vectordb.load_embedding_model()
        
        if os.path.exists(persist_directory) and not force_reload and not update:
            self.vector_store = self.vectordb.load_vdb(persist_directory, self.embeddings, db_type = DB_TYPE)
        
        elif os.path.exists(persist_directory) and update:
            self.chunks = self.vectordb.get_text_chunks(self.documents , CHUNK_SIZE, CHUNK_OVERLAP)
            self.vector_store = self.vectordb.load_vdb(persist_directory, self.embeddings, db_type = DB_TYPE)
            self.vector_store = self.vectordb.update_vdb(self.chunks, self.embeddings, DB_TYPE, persist_directory)
            
        else:
            self.chunks = self.vectordb.get_text_chunks(self.documents , CHUNK_SIZE, CHUNK_OVERLAP)
            self.vector_store = self.vectordb.create_vector_store(self.chunks, self.embeddings, DB_TYPE, None)
    
    def create_and_save_local(self, input_directory, persist_directory, update=False):
        self.chunks = self.vectordb.get_text_chunks(self.documents , CHUNK_SIZE, CHUNK_OVERLAP)
        self.embeddings = self.vectordb.load_embedding_model()
        if update:
            self.config["update"]=True
            self.vector_store = self.vectordb.update_vdb(self.chunks, self.embeddings, DB_TYPE, input_directory, persist_directory)

        else:
            self.vector_store = self.vectordb.create_vector_store(self.chunks, self.embeddings, DB_TYPE, persist_directory)

       
    def retrieval_qa_chain(self):
        prompt = load_prompt("./prompts/llama7b-web_crwling_data_retriever.yaml")
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": SCORE_TRESHOLD, "k": K_RETRIEVED_DOCUMENTS},
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

