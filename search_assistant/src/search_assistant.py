import os
import re
import sys
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import requests
import json
from dotenv import load_dotenv
from serpapi import GoogleSearch

from langchain.prompts import PromptTemplate, load_prompt
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from urllib.parse import urljoin, urlparse, urldefrag
from vectordb.vector_db import VectorDb
from langchain.chains import RetrievalQA

from utils.sambanova_endpoint import SambaNovaEndpoint, SambaverseEndpoint

load_dotenv(os.path.join(repo_dir,".env"))

from langchain.globals import set_debug

set_debug(False)

CONFIG_PATH = os.path.join(kit_dir,"config.yaml")

class SearchAssistant():
    def __init__(self, config=None) -> None:
        if config is None:
            self.config = {}
        else:
            self.config = config
        config_info=self._get_config_info(CONFIG_PATH)
        self.api_info = config_info[0]
        self.llm_info = config_info[1]
        self.retrieval_info = config_info[2]
        self.web_crawling_params = config_info[3]
        self.extra_loaders = config_info[4]
        self.documents = None
        self.urls = None
        self.llm = self.init_llm_model()
        self.vectordb = VectorDb()
        self.qa_chain = None
        
    def _get_config_info(self, config_path):
        """
        Loads json config file
        """
        # Read config file
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config["api"]
        llm_info =  config["llm"]
        retrieval_info = config["retrieval"]
        web_crawling_params = config["web_crawling"]
        extra_loaders = config["extra_loaders"]
        
        return api_info, llm_info, retrieval_info, web_crawling_params, extra_loaders
        
    def init_llm_model(self) -> None:
        """Initializes the LLM endpoint
        """
        if self.api_info=="sambaverse":
            llm = SambaverseEndpoint(
                sambaverse_model_name=self.llm_info["sambaverse_model_name"],
                sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
                model_kwargs={
                    "do_sample": True, 
                    "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                    "temperature": self.llm_info["temperature"],
                    "process_prompt": True,
                    "select_expert": self.llm_info["sambaverse_select_expert"],
                    #"stop_sequences": { "type":"str", "value":""},
                    # "repetition_penalty": {"type": "float", "value": "1"},
                    # "top_k": {"type": "int", "value": "50"},
                    # "top_p": {"type": "float", "value": "1"}
                }
            )
        elif self.pi_info=="sambastudio":
            llm = SambaNovaEndpoint(
                model_kwargs={
                    "do_sample": True, 
                    "temperature": self.llm_info["temperature"],
                    "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                    #"stop_sequences": { "type":"str", "value":""},
                    # "repetition_penalty": {"type": "float", "value": "1"},
                    # "top_k": {"type": "int", "value": "50"},
                    # "top_p": {"type": "float", "value": "1"}
                }
            ) 
        return llm
    
    def querySerper(self, query: str, limit: int = 5, do_analysis: bool = True ,include_site_links: bool = False):
        """A search engine. Useful for when you need to answer questions about current events. Input should be a search query."""
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query,
            "num": limit
        })
        headers = {
            'X-API-KEY': os.environ.get("SERPER_API_KEY"),
            'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=payload).json()
        results=response["organic"]
        links = [r["link"] for r in results]
        if include_site_links:
            sitelinks = []
            for r in [r.get("sitelinks",[]) for r in results]:
                sitelinks.extend([site.get("link", None) for site in r])
            links.extend(sitelinks)
        links=list(filter(lambda x: x is not None, links))
        
        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, "prompts/llama70b-SerperSearchAnalysis.yaml"))
            formatted_prompt = prompt.format(question=query, context=json.dumps(results))
            return self.llm.invoke(formatted_prompt), links
        else:
            return response, links
        
    def queryOpenSerp(self, query: str, limit: int = 5, do_analysis: bool = True, engine="google") -> str:
        """A search engine. Useful for when you need to answer questions about current events. Input should be a search query."""
        if engine not in ["google","yandex","baidu"]:
            raise ValueError("engine must be either google, yandex or baidu")
        url = f"http://127.0.0.1:7000/{engine}/search"
        params = {
            "lang": "EN",
            "limit": limit,
            "text": query
        }

        results = requests.get(url, params=params).json()
        
        links = [r["url"] for r in results]
        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, "prompts/llama70b-OpenSearchAnalysis.yaml"))
            formatted_prompt = prompt.format(question=query, context=json.dumps(results))
            return self.llm.invoke(formatted_prompt), links
        else:
            return results, links

    def remove_links(self, text):
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    def querySerpapi(self, query: str, limit: int = 1, do_analysis: bool = True, engine="google") -> str:
        if engine not in ["google", "bing"]:
            raise ValueError("engine must be either google or bing")
        params = {
            "q": query,
            "num": limit,
            "engine":engine,
            "api_key": os.environ.get("SERPAPI_API_KEY")
            }

        search = GoogleSearch(params)
        response= search.get_dict()
        
        knowledge_graph = response.get("knowledge_graph", None)
        results =  response.get("organic_results",None)

        links = []
        links = [r["link"] for r in results]
        
        
        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, "prompts/llama70b-SerpapiSearchAnalysis.yaml"))
            if knowledge_graph:
                knowledge_graph_str = json.dumps(knowledge_graph)
                knowledge_graph = self.remove_links(knowledge_graph_str)
                print(knowledge_graph)
                formatted_prompt = prompt.format(question=query, context=json.dumps(knowledge_graph))
            else:
                results_str = json.dumps(results)
                results_str = self.remove_links(results_str)
                formatted_prompt = prompt.format(question=query, context=json.dumps(results))
            return self.llm.invoke(formatted_prompt), links
        else:
            return response, links

    def load_remote_pdf(self, url):
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
    
    def load_htmls(self, urls, extra_loaders=None):
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
                    docs.extend(self.load_remote_pdf(url))
                else:
                    continue
            else:
                loader = AsyncHtmlLoader(url, verify_ssl=False)
                docs.extend(loader.load())
        return docs
    
    def link_filter(self, all_links, excluded_links):
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
    
    def clean_docs(self, docs):
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
    
    def web_crawl(self, urls, excluded_links=None):
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
        excluded_link_suffixes = {".ico", ".svg", ".jpg", ".png", ".jpeg", ".", ".docx", ".xls", ".xlsx"}
        scrapped_urls=[]
        
        urls = [url for url in urls if not url.endswith(tuple(excluded_link_suffixes))]
        urls = self.link_filter(urls, set(excluded_links)) 
        urls = list(urls)[:self.web_crawling_params["max_scraped_websites"]]   
            
        scraped_docs = self.load_htmls(urls, self.extra_loaders)
        scrapped_urls.append(urls)
            
        docs=self.clean_docs(scraped_docs)
        self.documents=docs
        self.urls=scrapped_urls
    
    def create_load_vector_store(self, force_reload: bool = False, update: bool = False):
            
            persist_directory = self.config.get("persist_directory", "NoneDirectory")
            
            embeddings = self.vectordb.load_embedding_model()
            
            if os.path.exists(persist_directory) and not force_reload and not update:
                self.vector_store = self.vectordb.load_vdb(persist_directory, embeddings, db_type = self.retrieval_info["db_type"])
            
            elif os.path.exists(persist_directory) and update:
                chunks = self.vectordb.get_text_chunks(self.documents , self.retrieval_info["chunk_size"], self.retrieval_info["chunk_overlap"])
                self.vector_store = self.vectordb.load_vdb(persist_directory, embeddings, db_type = self.retrieval_info["db_type"])
                self.vector_store = self.vectordb.update_vdb(chunks, embeddings, self.retrieval_info["db_type"], persist_directory)
                
            else:
                chunks = self.vectordb.get_text_chunks(self.documents , self.retrieval_info["chunk_size"], self.retrieval_info["chunk_overlap"])
                self.vector_store = self.vectordb.create_vector_store(chunks, embeddings, self.retrieval_info["db_type"], None)
                
        
    def create_and_save_local(self, input_directory, persist_directory, update=False):
        
        chunks = self.vectordb.get_text_chunks(self.documents , self.retrieval_info["chunk_size"], self.retrieval_info["chunk_overlap"])
        embeddings = self.vectordb.load_embedding_model()
        if update:
            self.config["update"]=True
            self.vector_store = self.vectordb.update_vdb(chunks, embeddings, self.retrieval_info["db_type"], input_directory, persist_directory)

        else:
            self.vector_store = self.vectordb.create_vector_store(chunks, embeddings, self.retrieval_info["db_type"], persist_directory)
          
    def basic_call(self, query, search_method="serpapi", max_results=5, search_engine="google"):
        if search_method == "serpapi":
            answer, links = self.querySerpapi(
                query=query, 
                limit=max_results,
                engine=search_engine,
                do_analysis=True
                )
        elif search_method == "serper":
            answer, links = self.querySerper( 
                query=query, 
                limit=max_results,
                do_analysis=True
                )
        elif search_method == "openserp":
            answer, links = self.queryOpenSerp(                
                query=query, 
                limit=max_results,
                engine=search_engine,
                do_analysis=True
                )
        return {"answer": answer, "metadata": {"source":links}}
            
    def set_retrieval_qa_chain(self):
        prompt = load_prompt(os.path.join(kit_dir,"prompts/llama7b-web_scraped_data_retriever.yaml"))
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": self.retrieval_info["score_treshold"], "k": self.retrieval_info["k_retrieved_documents"]},
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
    
    def search_and_scrape(self, query, search_method="serpapi", max_results=5, search_engine="google"):
        if search_method == "serpapi":
            _, links = self.querySerpapi(
                query=query, 
                limit=max_results,
                engine=search_engine,
                do_analysis=False
                )
        elif search_method == "serper":
            _, links = self.querySerper(
                query=query, 
                limit=max_results,
                do_analysis=False
                )
        elif search_method == "openserp":
            _, links = self.queryOpenSerp(
                query=query, 
                limit=max_results,
                engine=search_engine,
                do_analysis=False
                )
        self.web_crawl(urls=links)
        self.create_load_vector_store()
        self.set_retrieval_qa_chain()
        
    def retrieval_call(self, query):
        result = self.qa_chain.invoke(query)
        return result