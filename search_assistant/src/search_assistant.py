import json
import logging
import os
import re
import sys
from urllib.parse import urlparse

import requests
import yaml
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.docstore.document import Document
from langchain.memory import ConversationSummaryMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import load_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader, UnstructuredURLLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.language_models.llms import LLM

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from serpapi import GoogleSearch

from utils.model_wrappers.api_gateway import APIGateway
from utils.vectordb.vector_db import VectorDb

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir, 'data/my-vector-db')

load_dotenv(os.path.join(repo_dir, '.env'), override=True)


class SearchAssistant:
    """
    Class used to do generation over search query results and scraped sites
    """

    def __init__(self, sambanova_api_key: str, serpapi_api_key: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the search assistant with the given configuration parameters.

        Args:
        config (dict, optional):  Extra configuration parameters for the search Assistant.
        If not provided, default values will be used.
        """

        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if config is None:
            self.config = {}
        else:
            self.config = config
        config_info = self._get_config_info(CONFIG_PATH)
        self.sambanova_api_key = sambanova_api_key
        self.serpapi_api_key = serpapi_api_key
        self.embedding_model_info = config_info[0]
        self.llm_info = config_info[1]
        self.retrieval_info = config_info[2]
        self.web_crawling_params = config_info[3]
        self.extra_loaders: List[str] = config_info[4]
        self.prod_mode = config_info[5]
        self.documents: Sequence[Document]
        self.urls: List[Any] = []
        self.llm = self.init_llm_model()
        self.vectordb = VectorDb()
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
        self.memory: Optional[ConversationSummaryMemory] = None

    def _get_config_info(
        self, config_path: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str], bool]:
        """
        Loads json config file

        Args:
        config_path (str): Path to the YAML configuration file.

        Returns:
        embedding_model_info (string): String containing embedding model type to use, SambaStudio or CPU.
        llm_info (dict): Dictionary containing LLM parameters.
        retrieval_info (dict): Dictionary containing retrieval parameters
        web_crawling_params (dict): Dictionary containing web crawling parameters
        extra_loaders (list): list containing extra loader to use when doing web crawling (only pdf available
        in base kit)
        prod_mode (bool): Boolean indicating whether the app is in production mode
        """
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        embedding_model_info = config['embedding_model']
        llm_info = config['llm']
        retrieval_info = config['retrieval']
        web_crawling_params = config['web_crawling']
        extra_loaders = config['extra_loaders']
        prod_mode = config['prod_mode']

        return embedding_model_info, llm_info, retrieval_info, web_crawling_params, extra_loaders, prod_mode

    def init_memory(self) -> None:
        """
        Initialize conversation summary memory for the conversation
        """
        summary_prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-summary.yaml'))

        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            buffer='The human and AI greet each other to start a conversation.',
            memory_key='chat_history',
            return_messages=True,
            output_key='answer',
            prompt=summary_prompt,
        )

    def init_llm_model(self) -> LLM:
        """
        Initializes the LLM endpoint

        Returns:
        llm (SambaStudio or SambaNovaCloud): Langchain LLM to use
        """

        llm = APIGateway.load_llm(
            type=self.llm_info['type'],
            streaming=True,
            bundle=self.llm_info['bundle'],
            do_sample=self.llm_info['do_sample'],
            max_tokens_to_generate=self.llm_info['max_tokens_to_generate'],
            temperature=self.llm_info['temperature'],
            select_expert=self.llm_info['select_expert'],
            process_prompt=False,
            sambanova_api_key=self.sambanova_api_key,
        )
        return llm

    def reformulate_query_with_history(self, query: str) -> str:
        """
        Reformulates the query based on the conversation history.

        Args:
        query (str): The current query to reformulate.

        Returns:
        str: The reformulated query.
        """
        if self.memory is None:
            self.init_memory()
        custom_condensed_question_prompt = load_prompt(
            os.path.join(kit_dir, 'prompts', 'llama3-multiturn-custom_condensed_question.yaml')
        )
        assert self.memory is not None
        history = self.memory.load_memory_variables({})
        reformulated_query = self.llm.invoke(
            custom_condensed_question_prompt.format(chat_history=history, question=query)
        )
        return reformulated_query

    def remove_links(self, text: str) -> Any:
        """
        Removes all URLs from the given text.

        Args:
        text (str): The text from which to remove URLs.

        Returns:
        str: The text with all URLs removed.
        """
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    def parse_serp_analysis_output(self, answer: str, links: List[str]) -> str:
        """
        Parse the output of the SERP analysis prompt to replace the reference numbers with HTML links.

        Parameters:
        answer (str): The LLM answer of the query using the SERP tool output.
        links (list): A list of links corresponding to the reference numbers in the prompt.

        Returns:
        str: The parsed output with HTML links instead of reference numbers.
        """
        for i, link in enumerate(links):
            answer = answer.replace(f'[reference:{i+1}]', f'[<sup>{i+1}</sup>]({link})')
            answer = answer.replace(f'[reference: {i+1}]', f'[<sup>{i+1}</sup>]({link})')
            answer = answer.replace(f'[Reference:{i+1}]', f'[<sup>{i+1}</sup>]({link})')
            answer = answer.replace(f'[Reference: {i+1}]', f'[<sup>{i+1}</sup>]({link})')
        return answer

    def querySerper(
        self,
        query: str,
        limit: int = 5,
        do_analysis: bool = True,
        include_site_links: bool = False,
        conversational: bool = False,
    ) -> Tuple[str, List[str | None]]:
        """
        A search engine using Serper API. Useful for when you need to answer questions about current events. Input
        should be a search query.

        Parameters:
        query (str): The query to search.
        limit (int, optional): The maximum number of search results to retrieve. Defaults to 5.
        do_analysis (bool, optional): Whether to perform the LLM analysis directly on the search results. Defaults to
        True.
        include_site_links (bool, optional): Whether to include site links in the search results. Defaults to False.

        Returns:
        tuple: A tuple containing the search results or parsed llm generation and the corresponding links.
        """
        url = 'https://google.serper.dev/search'
        payload = json.dumps({'q': query, 'num': limit})
        headers = {'X-API-KEY': os.environ.get('SERPER_API_KEY'), 'Content-Type': 'application/json'}

        try:
            response = requests.post(url, headers=headers, data=payload)
            if response.status_code == 200:
                results = response.json().get('organic', [])
                if len(results) > 0:
                    links = [r['link'] for r in results]
                    context_list = []
                    for i, result in enumerate(results):
                        context_list.append(f'[reference:{i+1}] {result.get("title", "")}: {result.get("snippet", "")}')
                    context = '\n\n'.join(context_list)
                    self.logger.info(f'Context found: {context}')
                    if include_site_links:
                        sitelinks = []
                        for r in [r.get('sitelinks', []) for r in results]:
                            sitelinks.extend([site.get('link', None) for site in r])
                        links.extend(sitelinks)
                    links = list(filter(lambda x: x is not None, links))
                else:
                    context = 'Answer not found'
                    links = []
                    self.logger.info(f'No answer found for query: {query}')
            else:
                context = 'Answer not found'
                links = []
                self.logger.error(f'Request failed with status code: {response.status_code}')
                self.logger.error(f'Error message: {response.text}')
        except Exception as e:
            context = 'Answer not found'
            links = []
            self.logger.error(f'Error message: {e}')

        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-serp_analysis.yaml'))
            formatted_prompt = prompt.format(question=query, context=context)
            answer = self.llm.invoke(formatted_prompt)
            return self.parse_serp_analysis_output(answer, links), links
        else:
            return context, links

    def queryOpenSerp(
        self,
        query: str,
        limit: int = 5,
        do_analysis: bool = True,
        engine: Optional[str] = 'google',
        conversational: bool = False,
    ) -> Tuple[str, List[Any]]:
        """
        A search engine using OpenSerp local API. Useful for when you need to answer questions about current events.
        Input should be a search query.

        Parameters:
        query (str): The query to search.
        limit (int, optional): The maximum number of search results to retrieve. Defaults to 5.
        do_analysis (bool, optional): Whether to perform the LLM analysis directly on the search results. Defaults
        to True.
        include_site_links (bool, optional): Whether to include site links in the search results. Defaults to False.
        engine (str, optional): The search engine to use

        Returns:
        tuple: A tuple containing the search results or parsed llm generation and the corresponding links.
        """
        if engine not in ['google', 'yandex', 'baidu']:
            raise ValueError('engine must be either google, yandex or baidu')
        url = f'http://127.0.0.1:7000/{engine}/search'
        params = {'lang': 'EN', 'limit': limit, 'text': query}

        try:
            response = requests.get(url, params=params)  # type: ignore
            if response.status_code == 200:
                results = response.json()
                if len(results) > 0:
                    links = [r['url'] for r in results]
                    context_list = []
                    for i, result in enumerate(results):
                        context_list.append(
                            f'[reference:{i+1}] {result.get("title", "")}: {result.get("description", "")}'
                        )
                    context = '\n\n'.join(context_list)
                    self.logger.info(f'Context found: {context}')
                else:
                    context = 'Answer not found'
                    links = []
                    self.logger.info(f'No answer found for query: {query}')
            else:
                context = 'Answer not found'
                links = []
                self.logger.error(f'Request failed with status code: {response.status_code}')
                self.logger.error(f'Error message: {response.text}')
        except Exception as e:
            context = 'Answer not found'
            links = []
            self.logger.error(f'Error message: {e}')

        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-serp_analysis.yaml'))
            formatted_prompt = prompt.format(question=query, context=context)
            answer = self.llm.invoke(formatted_prompt)
            return self.parse_serp_analysis_output(answer, links), links
        else:
            return context, links

    def querySerpapi(
        self,
        query: str,
        limit: int = 1,
        do_analysis: bool = True,
        engine: Optional[str] = 'google',
    ) -> Tuple[str, List[Any]]:
        """
        A search engine using Serpapi API. Useful for when you need to answer questions about current events. Input
        should be a search query.

        Parameters:
        query (str): The query to search.
        limit (int, optional): The maximum number of search results to retrieve. Defaults to 5.
        do_analysis (bool, optional): Whether to perform the LLM analysis directly on the search results. Defaults
        to True.
        engine (str, optional): The search engine to use

        Returns:
        tuple: A tuple containing the search results or parsed llm generation and the corresponding links.
        """
        if engine not in ['google', 'bing']:
            raise ValueError('engine must be either google or bing')
        params = {'q': query, 'num': limit, 'engine': engine, 'api_key': self.serpapi_api_key}

        try:
            search = GoogleSearch(params)
            response = search.get_dict()

            knowledge_graph = response.get('knowledge_graph', None)
            results = response.get('organic_results', [])

            links = []
            if len(results) > 0:
                links = [r['link'] for r in results]
                context_list = []
                for i, result in enumerate(results):
                    context_list.append(f'[reference:{i+1}] {result.get("title", "")}: {result.get("snippet", "")}')
                context = '\n\n'.join(context_list)
                self.logger.info(f'Context found: {context}')
            else:
                context = 'Answer not found'
                links = []
                self.logger.info(f'No answer found for query: {query}. Raw response: {response}')
        except Exception as e:
            context = 'Answer not found'
            links = []
            self.logger.error(f'Error message: {e}')

        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-serp_analysis.yaml'))
            formatted_prompt = prompt.format(question=query, context=context)
            answer = self.llm.invoke(formatted_prompt)
            return self.parse_serp_analysis_output(answer, links), links
        else:
            return context, links

    def load_remote_pdf(self, url: str) -> List[Document]:
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

    def load_htmls(self, urls: List[str], extra_loaders: Optional[List[str]] = None) -> List[Document]:
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
                    docs.extend(self.load_remote_pdf(url))
                else:
                    continue
            else:
                loader = AsyncHtmlLoader(url, verify_ssl=False)
                docs.extend(loader.load())
        return docs

    def link_filter(self, all_links: List[str], excluded_links: Set[str]) -> Set[str]:
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

    def clean_docs(self, docs: Sequence[Document]) -> Sequence[Document]:
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

    def web_crawl(self, urls: List[str], excluded_links: Optional[List[str]] = None) -> None:
        """
        Perform web crawling, retrieve and clean HTML documents from the given URLs, with specified depth of
        exploration.
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
        excluded_link_suffixes = {'.ico', '.svg', '.jpg', '.png', '.jpeg', '.', '.docx', '.xls', '.xlsx'}
        scrapped_urls = []

        urls = [url for url in urls if not url.endswith(tuple(excluded_link_suffixes))]
        unique_urls = self.link_filter(urls, set(excluded_links))
        print(f'{unique_urls=}')
        if len(unique_urls) == 0:
            raise ValueError(
                """not sites to scrape after filtering links, check the excluded_links config or increase Max number of
                results to retrieve"""
            )
        urls = list(unique_urls)[: self.web_crawling_params['max_scraped_websites']]

        scraped_docs = self.load_htmls(urls, self.extra_loaders)
        scrapped_urls.extend(urls)

        docs = self.clean_docs(scraped_docs)
        self.documents = docs
        self.urls = scrapped_urls

    def get_text_chunks_with_references(
        self, docs: Sequence[Document], chunk_size: int, chunk_overlap: int
    ) -> List[Document]:
        """Gets text chunks. If metadata is not None, it will create chunks with metadata elements.

        Args:
            docs (list): list of documents or texts. If no metadata is passed, this parameter is a list of documents.
            If metadata is passed, this parameter is a list of texts.
            chunk_size (int): chunk size in number of characters
            chunk_overlap (int): chunk overlap in number of characters

        Returns:
            list: list of documents
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        sources = {site: i + 1 for i, site in enumerate(self.urls)}
        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            reference = chunk.metadata['source']  # get the number in the dict
            chunk.page_content = f'[reference:{sources[reference]}] {chunk.page_content}\n\n'

        return chunks

    def create_and_save_local(
        self,
        input_directory: Optional[str] = None,
        persist_directory: Optional[str] = None,
        update: Optional[bool] = False,
    ) -> None:
        """
        Create a vector store based on the given documents.
        Args:
            input_directory: The directory containing the previously created vectorstore.
            persist_directory: The directory to save the vectorstore.
            update (bool, optional): Whether to update the vector store. Defaults to False.
        """
        persist_directory = persist_directory or self.config.get('persist_directory', 'NoneDirectory')

        chunks = self.get_text_chunks_with_references(
            self.documents, self.retrieval_info['chunk_size'], self.retrieval_info['chunk_overlap']
        )
        embeddings = APIGateway.load_embedding_model(
            type=self.embedding_model_info.get('type'),
            batch_size=self.embedding_model_info.get('batch_size'),
            bundle=self.embedding_model_info.get('bundle'),
            select_expert=self.embedding_model_info.get('model'),
        )
        if update and os.path.exists(persist_directory):
            self.config['update'] = True
            self.vector_store = self.vectordb.update_vdb(
                chunks, embeddings, self.retrieval_info['db_type'], input_directory, persist_directory
            )

        else:
            self.vector_store = self.vectordb.create_vector_store(
                chunks, embeddings, self.retrieval_info['db_type'], persist_directory, 'default_collection'
            )

    def basic_call(
        self,
        query: str,
        reformulated_query: Optional[str] = None,
        search_method: Optional[str] = 'serpapi',
        max_results: int = 5,
        search_engine: Optional[str] = 'google',
        conversational: bool = False,
    ) -> Dict[str, Any]:
        """
        Do a basic call to the llm using the query result snippets as context
        Args:
            query (str): The query to search.
            reformulated_query (str, optional): The reformulated query to search. Defaults to None.
            search_method (str, optional): The search method to use. Defaults to "serpapi".
            max_results (int, optional): The maximum number of search results to retrieve. Defaults to 5.
            search_engine (str, optional): The search engine to use. Defaults to "google".
            conversational (bool, optional): Whether to save conversation to memory. Defaults to False.
        """
        if reformulated_query is None:
            reformulated_query = query

        if search_method == 'serpapi':
            answer, links = self.querySerpapi(
                query=reformulated_query, limit=max_results, engine=search_engine, do_analysis=True
            )
        elif search_method == 'serper':
            answer, links = self.querySerper(query=reformulated_query, limit=max_results, do_analysis=True)
        elif search_method == 'openserp':
            answer, links = self.queryOpenSerp(
                query=reformulated_query, limit=max_results, engine=search_engine, do_analysis=True
            )

        if conversational and self.memory is not None:
            self.memory.save_context(inputs={'input': query}, outputs={'answer': answer})

        return {'answer': answer, 'sources': links}

    def set_retrieval_qa_chain(self, conversational: bool = False) -> None:
        """
        Set a retrieval chain for queries that use as retriever a previously created vectorstore
        """
        retrieval_qa_prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-web_scraped_data_retriever.yaml'))
        retriever = self.vector_store.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={
                'score_threshold': self.retrieval_info['score_treshold'],
                'k': self.retrieval_info['k_retrieved_documents'],
            },
        )
        if conversational:
            self.init_memory()

            custom_condensed_question_prompt = load_prompt(
                os.path.join(kit_dir, 'prompts', 'llama3-multiturn-custom_condensed_question.yaml')
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                chain_type='stuff',
                return_source_documents=True,
                verbose=False,
                condense_question_prompt=custom_condensed_question_prompt,
                combine_docs_chain_kwargs={'prompt': retrieval_qa_prompt},
            )

        else:
            self.qa_chain = RetrievalQA.from_llm(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                verbose=False,
                input_key='question',
                output_key='answer',
                prompt=retrieval_qa_prompt,
            )

    def search_and_scrape(
        self,
        query: str,
        search_method: Optional[str] = 'serpapi',
        max_results: int = 5,
        search_engine: Optional[str] = 'google',
        persist_directory: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Do a call to the serp tool, scrape the url results, and save the scraped data in a a vectorstore
        Args:
            query (str): The query to search.
            max_results (int): The maximum number of search results. Default is 5
            search_method (str, optional): The search method to use. Defaults to "serpapi".
            search_engine (str, optional): The search engine to use. Defaults to "google".
        """
        if search_method == 'serpapi':
            _, links = self.querySerpapi(query=query, limit=max_results, engine=search_engine, do_analysis=False)
        elif search_method == 'serper':
            _, links = self.querySerper(query=query, limit=max_results, do_analysis=False)
        elif search_method == 'openserp':
            _, links = self.queryOpenSerp(query=query, limit=max_results, engine=search_engine, do_analysis=False)
        if len(links) > 0:
            self.web_crawl(urls=links)
            self.create_and_save_local(persist_directory=persist_directory)
            self.set_retrieval_qa_chain(conversational=True)
            return None
        else:
            return {
                'message': f"No links found for '{query}'. Try again, "
                'increase the number of results or check your api keys'
            }

    def get_relevant_queries(self, query: str) -> Any:
        """
        Generates a list of related queries based on the input query.

        Args:
        query (str): The input query for which related queries are to be generated.

        Returns:
        list: A list of related queries based on the input query.
        """
        prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-related_questions.yaml'))
        response_schemas = [ResponseSchema(name='related_queries', description=f'related search queries', type='list')]
        list_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        list_format_instructions = list_output_parser.get_format_instructions()
        relevant_queries_chain = prompt | self.llm | list_output_parser
        input_variables = {'question': query, 'format_instructions': list_format_instructions}
        return relevant_queries_chain.invoke(input_variables).get('related_queries', [])

    def parse_retrieval_output(self, result: Dict[str, Any]) -> str:
        """
        Parses the output of the retrieval chain to map the original source numbers with the numbers in generation.

        Args:
        result (dict): The result from the retrieval chain, containing the answer and source documents.

        Returns:
        str: The parsed answer with the mapped source numbers.
        """
        parsed_answer = self.parse_serp_analysis_output(result['answer'], self.urls)
        # mapping original sources order with question used sources order
        question_sources = set(f'{doc.metadata["source"]}' for doc in result['source_documents'])
        question_sources_map = {source: i + 1 for i, source in enumerate(question_sources)}
        for i, link in enumerate(self.urls):
            if link in parsed_answer:
                parsed_answer = parsed_answer.replace(
                    f'[<sup>{i+1}</sup>]({link})', f'[<sup>{question_sources_map[link]}</sup>]({link})'
                )
        return parsed_answer

    def retrieval_call(self, query: str) -> Any:
        """
        Do a call to the retriever chain

        Args:
        query (str): The query to search.

        Returns:
        result (str): The final Result to the user query
        """
        assert self.qa_chain is not None
        result = self.qa_chain.invoke(query)
        result['answer'] = self.parse_retrieval_output(result)
        return result
