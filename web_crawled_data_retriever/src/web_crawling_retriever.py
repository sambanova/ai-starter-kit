import os
import sys
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

import nest_asyncio
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.prompts import load_prompt
from langchain_community.document_loaders import UnstructuredURLLoader

from utils.model_wrappers.api_gateway import APIGateway

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from typing import Any, Dict, List

from utils.vectordb.vector_db import VectorDb

load_dotenv(os.path.join(repo_dir, '.env'))
nest_asyncio.apply()

DATA_DIRECTORY = os.path.join(kit_dir, 'data')
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')


class WebCrawlingRetrieval:
    """
    Class for web crawling retrieval.
    """

    def __init__(self, documents: Optional[List[Document]] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the WebCrawlingRetrieval class.

        Args:
        documents (list, optional): A list of langchain documents. Defaults to None.
        config (dict, optional): A dictionary of extra configuration parameters. Defaults to None.
        """
        if config is None:
            config = {}
        self.documents = documents
        self.config = config
        config_info = self._get_config_info(CONFIG_PATH)
        self.api_info = config_info[0]
        self.embedding_model_info = config_info[1]
        self.llm_info = config_info[2]
        self.retrieval_info = config_info[3]
        self.web_crawling_params = config_info[4]
        self.extra_loaders = config_info[5]
        self.vectordb = VectorDb()

    def _get_config_info(
        self, config_path: Optional[str] = CONFIG_PATH
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str]]:
        """
        Loads json config file
        Args:
            path (str, optional): The path to the config file. Defaults to CONFIG_PATH.
        Returns:
            api_info (string): string containing API to use sambastudio or sncloud.
            embedding_model_info (string): String containing embedding model type to use, SambaStudio or CPU.
            llm_info (dict): Dictionary containing LLM parameters.
            retrieval_info (dict): Dictionary containing retrieval parameters
            web_crawling_params (dict): Dictionary containing web crawling parameters
            extra_loaders (list): list containing extra loader to use when doing web crawling
            (only pdf available in base kit)
        """
        # Read config file
        assert config_path is not None
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config['api']
        embedding_model_info = config['embedding_model']
        llm_info = config['llm']
        retrieval_info = config['retrieval']
        web_crawling_params = config['web_crawling']
        extra_loaders = config['extra_loaders']

        return api_info, embedding_model_info, llm_info, retrieval_info, web_crawling_params, extra_loaders

    @staticmethod
    def load_remote_pdf(url: str) -> List[Document]:
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
    def load_htmls(urls: Set[str], extra_loaders: Optional[List[str]] = None) -> List[Document]:
        """
        Load HTML documents from the given URLs.
        Args:
            urls (list): A list of URLs to load HTML documents from.
        Returns:
            list: A list of loaded HTML documents.
        """
        if extra_loaders is None:
            extra_loaders = []
        docs = []
        for url in urls:
            if url.endswith('.pdf'):
                if 'pdf' in extra_loaders:
                    docs.extend(WebCrawlingRetrieval.load_remote_pdf(url))
                else:
                    continue
            else:
                loader = AsyncHtmlLoader(url, verify_ssl=False)
                docs.extend(loader.load())
        return docs

    @staticmethod
    def link_filter(all_links: Set[str], excluded_links: set[str]) -> Set[str]:
        """
        Filters a list of links based on a list of excluded links.
        Args:
            all_links (List[str]): A list of links to filter.
            excluded_links (List[str]): A list of excluded links.
        Returns:
            Set[str]: A list of filtered links.
        """
        clean_excluded_links = set()
        for excluded_link in excluded_links:
            parsed_link = urlparse(excluded_link)
            clean_excluded_links.add(parsed_link.netloc + parsed_link.path)
        filtered_links = set()
        for link in all_links:
            # Check if the link contains any of the excluded links
            if not any(excluded_link in link for excluded_link in clean_excluded_links):
                filtered_links.add(link)
        return filtered_links

    @staticmethod
    def find_links(docs: List[Document], excluded_links: Optional[List[str]] = None) -> Set[str]:
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
        excluded_link_suffixes = {'.ico', '.svg', '.jpg', '.png', '.jpeg', '.', '.docx', '.xls', '.xlsx'}
        for doc in docs:
            page_content = doc.page_content
            base_url = doc.metadata['source']
            # excluded_links.append(base_url) #enable this to prevent to crawl inside the same root website
            soup = BeautifulSoup(page_content, 'html.parser')
            # Identify the main content section (customize based on HTML structure)
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main_content:
                links = main_content.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    # Check if the link is not an anchor link and not in the excluded links or suffixes
                    if not href.startswith(('#', 'data:', 'javascript:')) and not any(
                        href.endswith(suffix) for suffix in excluded_link_suffixes
                    ):
                        full_url, _ = urldefrag(urljoin(base_url, href))
                        all_links.add(full_url)
        all_links = WebCrawlingRetrieval.link_filter(all_links, set(excluded_links))
        return all_links

    @staticmethod
    def clean_docs(docs: Any) -> Any:
        """
        Clean the given HTML documents by transforming them into plain text.
        Args:
            docs (list): A list of langchain documents with html content to clean.
        Returns:
            list: A list of cleaned plain text documents.
        """
        html2text_transformer = Html2TextTransformer()
        docs = html2text_transformer.transform_documents(documents=docs)
        return docs

    def web_crawl(
        self, urls: Set[str], excluded_links: Optional[List[str]] = None, depth: int = 1
    ) -> Tuple[List[Document], List[str]]:
        """
        Perform web crawling, retrieve and clean HTML documents from the given URLs, with specified depth
        of exploration.
        Args:
            urls (list): A list of URLs to crawl.
            excluded_links (list, optional): A list of links to exclude from crawling. Defaults to None.
            depth (int, optional): The depth of crawling, determining how many layers of internal links to explore.
            Defaults to 1
        Returns:
            tuple: A tuple containing the langchain documents (list) and the scrapped URLs (list).
        """
        if excluded_links is None:
            excluded_links = []
        excluded_links.extend(self.web_crawling_params['excluded_links'])
        if (
            depth > self.web_crawling_params['max_depth']
        ):  # Max depth change with precaution number of sites grow exponentially
            depth = self.web_crawling_params['max_depth']
        scraped_urls: List[Any] = []
        raw_docs: List[Any] = []
        for _ in range(depth):
            if len(scraped_urls) + len(urls) >= self.web_crawling_params['max_scraped_websites']:
                list_urls = list(urls)[: self.web_crawling_params['max_scraped_websites'] - len(scraped_urls)]
                urls = set(list_urls)

            scraped_docs = WebCrawlingRetrieval.load_htmls(urls, self.extra_loaders)
            scraped_urls.extend(urls)
            urls = WebCrawlingRetrieval.find_links(scraped_docs, excluded_links)

            excluded_links.extend(scraped_urls)
            raw_docs.extend(scraped_docs)

            if len(scraped_urls) == self.web_crawling_params['max_scraped_websites']:
                break

        docs = WebCrawlingRetrieval.clean_docs(raw_docs)
        return docs, scraped_urls

    def init_llm_model(self) -> None:
        """
        Initializes the LLM endpoint
        """
        self.llm = APIGateway.load_llm(
            type=self.api_info,
            streaming=True,
            bundle=self.llm_info['bundle'],
            do_sample=self.llm_info['do_sample'],
            max_tokens_to_generate=self.llm_info['max_tokens_to_generate'],
            temperature=self.llm_info['temperature'],
            select_expert=self.llm_info['model'],
            process_prompt=False,
        )

    def create_load_vector_store(self, force_reload: bool = False, update: bool = False) -> None:
        """
        Create a vector store based on the given documents.
        Args:
            force_reload (bool, optional): Whether to force reloading the vector store. Defaults to False.
            update (bool, optional): Whether to update the vector store. Defaults to False.
        """
        persist_directory = self.config.get('persist_directory', 'NoneDirectory')

        self.embeddings = APIGateway.load_embedding_model(
            type=self.embedding_model_info.get('type'),
            batch_size=self.embedding_model_info.get('batch_size'),
            bundle=self.embedding_model_info.get('bundle'),
            model=self.embedding_model_info.get('model'),
        )

        if os.path.exists(persist_directory) and not force_reload and not update:
            self.vector_store = self.vectordb.load_vdb(
                persist_directory, self.embeddings, db_type=self.retrieval_info['db_type']
            )

        elif os.path.exists(persist_directory) and update:
            assert self.documents is not None
            self.chunks = self.vectordb.get_text_chunks(
                self.documents, self.retrieval_info['chunk_size'], self.retrieval_info['chunk_overlap']
            )
            self.vector_store = self.vectordb.load_vdb(
                persist_directory, self.embeddings, db_type=self.retrieval_info['db_type']
            )
            self.vector_store = self.vectordb.update_vdb(
                self.chunks, self.embeddings, self.retrieval_info['db_type'], persist_directory
            )

        else:
            assert self.documents is not None
            self.chunks = self.vectordb.get_text_chunks(
                self.documents, self.retrieval_info['chunk_size'], self.retrieval_info['chunk_overlap']
            )
            self.vector_store = self.vectordb.create_vector_store(
                self.chunks, self.embeddings, self.retrieval_info['db_type'], None
            )

    def create_and_save_local(
        self, input_directory: str, persist_directory: Optional[str], update: bool = False
    ) -> None:
        """
        Create a vector store based on the given documents.
        Args:
            input_directory: The directory containing the previously created vectorstore.
            persist_directory: The directory to save the vectorstore.
            update (bool, optional): Whether to update the vector store. Defaults to False.
        """
        assert self.documents is not None
        self.chunks = self.vectordb.get_text_chunks(
            self.documents, self.retrieval_info['chunk_size'], self.retrieval_info['chunk_overlap']
        )
        self.embeddings = APIGateway.load_embedding_model(
            type=self.embedding_model_info.get('type'),
            batch_size=self.embedding_model_info.get('batch_size'),
            bundle=self.embedding_model_info.get('bundle'),
            model=self.embedding_model_info.get('model'),
        )
        if update:
            self.config['update'] = True
            self.vector_store = self.vectordb.update_vdb(
                self.chunks, self.embeddings, self.retrieval_info['db_type'], input_directory, persist_directory
            )

        else:
            self.vector_store = self.vectordb.create_vector_store(
                self.chunks, self.embeddings, self.retrieval_info['db_type'], persist_directory
            )

    def retrieval_qa_chain(self) -> RetrievalQA:
        """
        Creates a RetrievalQA chain from LangChain chains.

        Returns:
        RetrievalQA: A LangChain RetrievalQA chain object.
        """
        prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama7b-web_crwling_data_retriever.yaml'))
        retriever = self.vector_store.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={
                'score_threshold': self.retrieval_info['score_treshold'],
                'k': self.retrieval_info['k_retrieved_documents'],
            },
        )
        self.qa_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True,
            input_key='question',
            output_key='answer',
            prompt=prompt,
        )
        return self.qa_chain
