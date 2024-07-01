import json
import os
import re
import sys
from urllib.parse import urldefrag, urljoin, urlparse

import requests
import yaml
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.globals import set_debug
from langchain.memory import ConversationSummaryMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import load_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader, UnstructuredURLLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.llms.sambanova import SambaStudio, Sambaverse
from serpapi import GoogleSearch

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from vectordb.vector_db import VectorDb

set_debug(False)
load_dotenv(os.path.join(repo_dir, '.env'))

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')


class SearchAssistant:
    """
    class used to do generation over search query results and scraped sites
    """

    def __init__(self, config=None) -> None:
        """
        Initializes the search assistant with the given configuration parameters.

        Args:
        config (dict, optional):  Extra configuration parameters for the search Assistant.
        If not provided, default values will be used.
        """
        if config is None:
            self.config = {}
        else:
            self.config = config
        config_info = self._get_config_info(CONFIG_PATH)
        self.api_info = config_info[0]
        self.embedding_model_info = config_info[1]
        self.llm_info = config_info[2]
        self.retrieval_info = config_info[3]
        self.web_crawling_params = config_info[4]
        self.extra_loaders = config_info[5]
        self.documents = None
        self.urls = None
        self.llm = self.init_llm_model()
        self.vectordb = VectorDb()
        self.qa_chain = None
        self.memory = None

    def _get_config_info(self, config_path):
        """
        Loads json config file

        Args:
        config_path (str): Path to the YAML configuration file.

        Returns:
        api_info (string): string containing API to use SambaStudio or Sambaverse.
        embedding_model_info (string): String containing embedding model type to use, SambaStudio or CPU.
        llm_info (dict): Dictionary containing LLM parameters.
        retrieval_info (dict): Dictionary containing retrieval parameters
        web_crawling_params (dict): Dictionary containing web crawling parameters
        extra_loaders (list): list containing extra loader to use when doing web crawling (only pdf available in base kit)
        """
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config['api']
        embedding_model_info = config['embedding_model']
        llm_info = config['llm']
        retrieval_info = config['retrieval']
        web_crawling_params = config['web_crawling']
        extra_loaders = config['extra_loaders']

        return api_info, embedding_model_info, llm_info, retrieval_info, web_crawling_params, extra_loaders

    def init_memory(self):
        """
        Initialize conversation summary memory for the conversation
        """
        summary_prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-summary.yaml'))

        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            max_token_limit=100,
            buffer='The human and AI greet each other to start a conversation.',
            memory_key='chat_history',
            return_messages=True,
            output_key='answer',
            prompt=summary_prompt,
        )

    def init_llm_model(self) -> None:
        """
        Initializes the LLM endpoint

        Returns:
        llm (SambaStudio or Sambaverse): Langchain LLM to use
        """
        if self.api_info == 'sambaverse':
            llm = Sambaverse(
                sambaverse_model_name=self.llm_info['sambaverse_model_name'],
                model_kwargs={
                    'do_sample': self.llm_info['do_sample'],
                    'max_tokens_to_generate': self.llm_info['max_tokens_to_generate'],
                    'temperature': self.llm_info['temperature'],
                    'top_p': self.llm_info['top_p'],
                    'process_prompt': True,
                    'select_expert': self.llm_info['select_expert'],
                },
            )
        elif self.api_info == 'sambastudio':
            if self.llm_info['coe'] == True:
                llm = SambaStudio(
                    streaming=True,
                    model_kwargs = {
                    'do_sample': True,
                    'do_sample': self.llm_info['do_sample'],
                    'top_p': self.llm_info['top_p'],
                    'temperature': self.llm_info['temperature'],
                    'max_tokens_to_generate': self.llm_info['max_tokens_to_generate'],
                    'select_expert': self.llm_info['select_expert'],
                }
                    
                )
                
            else:
                llm = SambaStudio(
                    streaming=True,
                    model_kwargs={
                        'do_sample': True,
                        'do_sample': self.llm_info['do_sample'],
                        'top_p': self.llm_info['top_p'],
                        'temperature': self.llm_info['temperature'],
                        'max_tokens_to_generate': self.llm_info['max_tokens_to_generate'],
                    }
                )
        return llm

    def reformulate_query_with_history(self, query):
        if self.memory is None:
            self.init_memory()
        custom_condensed_question_prompt = load_prompt(
            os.path.join(kit_dir, 'prompts', 'llama3-multiturn-custom_condensed_question.yaml')
        )
        history = self.memory.load_memory_variables({})
        reformulated_query = self.llm.invoke(
            custom_condensed_question_prompt.format(chat_history=history, question=query)
        )
        return reformulated_query

    def remove_links(self, text):
        """
        Removes all URLs from the given text.

        Args:
        text (str): The text from which to remove URLs.

        Returns:
        (str) The text with all URLs removed.
        """
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    def parse_serp_analysis_output(self, answer, links):
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
    ):
        """
        A search engine using Serper API. Useful for when you need to answer questions about current events. Input should be a search query.

        Parameters:
        query (str): The query to search.
        limit (int, optional): The maximum number of search results to retrieve. Defaults to 5.
        do_analysis (bool, optional): Whether to perform the LLM analysis directly on the search results. Defaults to True.
        include_site_links (bool, optional): Whether to include site links in the search results. Defaults to False.

        Returns:
        tuple: A tuple containing the search results or parsed llm generation and the corresponding links.
        """
        url = 'https://google.serper.dev/search'
        payload = json.dumps({'q': query, 'num': limit})
        headers = {'X-API-KEY': os.environ.get('SERPER_API_KEY'), 'Content-Type': 'application/json'}

        response = requests.post(url, headers=headers, data=payload).json()
        results = response['organic']
        links = [r['link'] for r in results]

        context = []
        for i, result in enumerate(results):
            context.append(f'[reference:{i+1}] {result.get("title", "")}: {result.get("snippet", "")}')
        context = '\n\n'.join(context)

        if include_site_links:
            sitelinks = []
            for r in [r.get('sitelinks', []) for r in results]:
                sitelinks.extend([site.get('link', None) for site in r])
            links.extend(sitelinks)
        links = list(filter(lambda x: x is not None, links))
        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-serp_analysis.yaml'))
            formatted_prompt = prompt.format(question=query, context=json.dumps(results))
            answer = self.llm.invoke(formatted_prompt)
            return self.parse_serp_analysis_output(answer, links), links
        else:
            return response, links

    def queryOpenSerp(
        self,
        query: str,
        limit: int = 5,
        do_analysis: bool = True,
        engine='google',
        conversational: bool = False,
    ) -> str:
        """
        A search engine using OpenSerp local API. Useful for when you need to answer questions about current events. Input should be a search query.

        Parameters:
        query (str): The query to search.
        limit (int, optional): The maximum number of search results to retrieve. Defaults to 5.
        do_analysis (bool, optional): Whether to perform the LLM analysis directly on the search results. Defaults to True.
        include_site_links (bool, optional): Whether to include site links in the search results. Defaults to False.
        engine (str, optional): The search engine to use

        Returns:
        tuple: A tuple containing the search results or parsed llm generation and the corresponding links.
        """
        if engine not in ['google', 'yandex', 'baidu']:
            raise ValueError('engine must be either google, yandex or baidu')
        url = f'http://127.0.0.1:7000/{engine}/search'
        params = {'lang': 'EN', 'limit': limit, 'text': query}

        results = requests.get(url, params=params).json()
        links = [r['url'] for r in results]

        context = []
        for i, result in enumerate(results):
            context.append(f'[reference:{i+1}] {result.get("title", "")}: {result.get("description", "")}')
        context = '\n\n'.join(context)

        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-serp_analysis.yaml'))
            formatted_prompt = prompt.format(question=query, context=context)
            answer = self.llm.invoke(formatted_prompt)
            return self.parse_serp_analysis_output(answer, links), links
        else:
            return results, links

    def querySerpapi(
        self,
        query: str,
        limit: int = 1,
        do_analysis: bool = True,
        engine='google',
    ) -> str:
        """
        A search engine using Serpapi API. Useful for when you need to answer questions about current events. Input should be a search query.

        Parameters:
        query (str): The query to search.
        limit (int, optional): The maximum number of search results to retrieve. Defaults to 5.
        do_analysis (bool, optional): Whether to perform the LLM analysis directly on the search results. Defaults to True.
        engine (str, optional): The search engine to use

        Returns:
        tuple: A tuple containing the search results or parsed llm generation and the corresponding links.
        """
        if engine not in ['google', 'bing']:
            raise ValueError('engine must be either google or bing')
        params = {'q': query, 'num': limit, 'engine': engine, 'api_key': os.environ.get('SERPAPI_API_KEY')}

        search = GoogleSearch(params)
        response = search.get_dict()

        knowledge_graph = response.get('knowledge_graph', None)
        results = response.get('organic_results', None)

        links = []
        links = [r['link'] for r in results]

        context = []
        for i, result in enumerate(results):
            context.append(f'[reference:{i+1}] {result.get("title", "")}: {result.get("snippet", "")}')
        context = '\n\n'.join(context)

        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-serp_analysis.yaml'))
            # results_str = json.dumps(results) #TODO remove if works with serpapi
            # results_str = self.remove_links(results_str)
            formatted_prompt = prompt.format(question=query, context=context)
            answer = self.llm.invoke(formatted_prompt)
            return self.parse_serp_analysis_output(answer, links), links
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

    def link_filter(self, all_links, excluded_links):
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

    def clean_docs(self, docs):
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
        excluded_links.extend(self.web_crawling_params['excluded_links'])
        excluded_link_suffixes = {'.ico', '.svg', '.jpg', '.png', '.jpeg', '.', '.docx', '.xls', '.xlsx'}
        scrapped_urls = []

        urls = [url for url in urls if not url.endswith(tuple(excluded_link_suffixes))]
        urls = self.link_filter(urls, set(excluded_links))
        print(f'{urls=}')
        if len(urls) == 0:
            raise ValueError(
                'not sites to scrape after filtering links, check the excluded_links config or increase Max number of results to retrieve'
            )
        urls = list(urls)[: self.web_crawling_params['max_scraped_websites']]

        scraped_docs = self.load_htmls(urls, self.extra_loaders)
        scrapped_urls.extend(urls)

        docs = self.clean_docs(scraped_docs)
        self.documents = docs
        self.urls = scrapped_urls

    def get_text_chunks_with_references(self, docs: list, chunk_size: int, chunk_overlap: int) -> list:
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

    def create_load_vector_store(self, force_reload: bool = False, update: bool = False):
        """
        Create a vector store based on the given documents.
        Args:
            force_reload (bool, optional): Whether to force reloading the vector store. Defaults to False.
            update (bool, optional): Whether to update the vector store. Defaults to False.
        """

        persist_directory = self.config.get('persist_directory', 'NoneDirectory')

        embeddings = self.vectordb.load_embedding_model(type=self.embedding_model_info)

        if os.path.exists(persist_directory) and not force_reload and not update:
            self.vector_store = self.vectordb.load_vdb(
                persist_directory, embeddings, db_type=self.retrieval_info['db_type']
            )

        elif os.path.exists(persist_directory) and update:
            chunks = self.get_text_chunks_with_references(
                self.documents, self.retrieval_info['chunk_size'], self.retrieval_info['chunk_overlap']
            )
            self.vector_store = self.vectordb.load_vdb(
                persist_directory, embeddings, db_type=self.retrieval_info['db_type']
            )
            self.vector_store = self.vectordb.update_vdb(
                chunks, embeddings, self.retrieval_info['db_type'], persist_directory
            )

        else:
            chunks = self.get_text_chunks_with_references(
                self.documents, self.retrieval_info['chunk_size'], self.retrieval_info['chunk_overlap']
            )
            self.vector_store = self.vectordb.create_vector_store(
                chunks, embeddings, self.retrieval_info['db_type'], None
            )

    def create_and_save_local(self, input_directory=None, persist_directory=None, update=False):
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
        embeddings = self.vectordb.load_embedding_model(type=self.embedding_model_info)
        if update and os.path.exists(persist_directory):
            self.config['update'] = True
            self.vector_store = self.vectordb.update_vdb(
                chunks, embeddings, self.retrieval_info['db_type'], input_directory, persist_directory
            )

        else:
            if os.path.exists(persist_directory):
                self.vector_store = self.vectordb.create_vector_store(
                    chunks, embeddings, self.retrieval_info['db_type'], persist_directory
                )
            else:
                self.vector_store = self.vectordb.create_vector_store(
                    chunks, embeddings, self.retrieval_info['db_type'], None
                )

    def basic_call(
        self,
        query,
        reformulated_query=None,
        search_method='serpapi',
        max_results=5,
        search_engine='google',
        conversational=False,
    ):
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

        if conversational:
            self.memory.save_context(inputs={'input': query}, outputs={'answer': answer})

        return {'answer': answer, 'sources': links}

    def set_retrieval_qa_chain(self, conversational=False):
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

    def search_and_scrape(self, query, search_method='serpapi', max_results=5, search_engine='google'):
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
        self.web_crawl(urls=links)
        # self.create_load_vector_store()
        self.create_and_save_local()
        self.set_retrieval_qa_chain(conversational=True)

    def get_relevant_queries(self, query):
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

    def parse_retrieval_output(self, result):
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

    def retrieval_call(self, query):
        """
        Do a call to the retriever chain

        Args:
        query (str): The query to search.

        Returns:

        result (str): The final Result to que user query

        """
        result = self.qa_chain.invoke(query)
        result['answer'] = self.parse_retrieval_output(result)
        return result
