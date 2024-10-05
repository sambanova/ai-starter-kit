import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import load_prompt
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import BaseOutputParser
from sec_edgar_downloader import Downloader
from xbrl import XBRLParser

from utils.model_wrappers.api_gateway import APIGateway
from utils.vectordb.vector_db import VectorDb

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from dotenv import load_dotenv

load_dotenv(os.path.join(repo_dir, '.env'))

DATA_DIRECTORY = os.path.join(kit_dir, 'data')
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')


class SecFiling:
    """
    Class that handles SEC Filing data set creation as vector database and retrieving information in different
    ways.
    """

    def __init__(self, config: Dict[str, Any] = {}) -> None:
        """Initializes SecFiling class

        Args:
            config (dict, optional): config object with entries like filing ticker and vector db persistent directory.
            Defaults to {}.
        """

        self.config = config
        self.vector_store = None
        self.llm: Optional[LLM] = None
        self.qa_chain = None
        self.conversational_chain = None
        self.comparative_process: Optional[Callable[[str], Dict[Any, Any]]] = None
        self.retriever = None

        params = self._get_config_info()
        self.api_info = params[0]
        self.embedding_model_info = params[1]
        self.llm_info = params[2]
        self.retrieval_info = params[3]
        self.query_decomposition_info = params[4]
        self.sec_info = params[5]

    def _get_config_info(
        self,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config['api']
        llm_info = config['llm']
        embedding_model_info = config['embedding_model']
        retrieval_info = config['retrieval']
        query_decomposition_info = config['query_decomposition']
        sec_info = config['sec']

        return api_info, embedding_model_info, llm_info, retrieval_info, query_decomposition_info, sec_info

    def init_llm_model(self) -> None:
        """Initializes the LLM endpoint"""
        self.llm = APIGateway.load_llm(
            type=self.api_info,
            streaming=True,
            coe=self.llm_info['coe'],
            do_sample=self.llm_info['do_sample'],
            max_tokens_to_generate=self.llm_info['max_tokens_to_generate'],
            temperature=self.llm_info['temperature'],
            select_expert=self.llm_info['select_expert'],
            process_prompt=False,
        )

    def download_sec_data(self, ticker: str) -> List[Any]:
        """Downloads SEC data based on ticker name

        Args:
            ticker (str): name of the ticker

        Raises:
            Exception: related to fetching data from Edgar's db

        Returns:
            list: list of loaded documents
        """

        try:
            dl = Downloader(self.sec_info['company'], self.sec_info['email'], DATA_DIRECTORY)
            dl.get(self.sec_info['report_type'], ticker, limit=self.sec_info['last_n_documents'])
        except Exception as ex:
            raise Exception(f'Failed to fetch data for {ticker} from Edgar database') from ex

        sec_dir = f"{DATA_DIRECTORY}/sec-edgar-filings/{ticker}/{self.sec_info['report_type']}"
        dir_loader = DirectoryLoader(sec_dir, glob='**/*.txt', loader_cls=TextLoader)
        documents = dir_loader.load()

        return documents

    def parse_xbrl_data(self, raw_documents: List[Any], company_ticker: str) -> List[Any]:
        """Parses XBRL data from a list of documents

        Args:
            raw_documents (list): list of documents

        Returns:
            parsed_documents: list of parsed documents from raw XBRL documents. Includes metadata about the source,
            ticker and report type.
        """

        xbrl_parser = XBRLParser()
        parsed_documents = []

        for raw_document in raw_documents:
            # get metadata information
            source = raw_document.metadata['source']
            report_type = self.sec_info['report_type']

            # parse raw document
            xbrl_doc = xbrl_parser.parse(raw_document.metadata['source'])
            p_span_tags = xbrl_doc.find_all(lambda x: x.name == 'div' and x.find('span'))
            xbrl_text = ' '.join(tag.get_text() for tag in p_span_tags)

            # create document
            parsed_document = Document(
                page_content=xbrl_text,
                metadata={'source': source, 'company_ticker': company_ticker, 'report_type': report_type},
            )
            parsed_documents.append(parsed_document)

        return parsed_documents

    def create_load_vector_store(self) -> None:
        """Creates or loads a vector db if it exists"""

        persist_directory = self.config.get('persist_directory', None)
        tickers = self.config.get('tickers', None)
        force_reload = self.config.get('force_reload', None)

        vectordb = VectorDb()
        embeddings = APIGateway.load_embedding_model(
            type=self.embedding_model_info['type'],
            batch_size=self.embedding_model_info['batch_size'],
            coe=self.embedding_model_info['coe'],
            select_expert=self.embedding_model_info['select_expert'],
        )

        if persist_directory and os.path.exists(persist_directory) and not force_reload:
            self.vector_store = vectordb.load_vdb(persist_directory, embeddings)
        else:
            all_chunks = []
            for ticker in tickers:
                documents = self.download_sec_data(ticker)
                parsed_documents = self.parse_xbrl_data(documents, ticker)
                chunks = vectordb.get_text_chunks(
                    parsed_documents, self.retrieval_info['chunk_size'], self.retrieval_info['chunk_overlap']
                )
                all_chunks.extend(chunks)

            self.vector_store = vectordb.create_vector_store(
                all_chunks, embeddings, self.retrieval_info['db_type'], persist_directory
            )

    def retrieval_qa_chain(self) -> None:
        """Defines the retrieval chain"""
        assert self.vector_store is not None
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={'k': self.retrieval_info['n_retrieved_documents']}
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            input_key='question',
            output_key='answer',
            return_source_documents=True,
            verbose=True,
        )

        custom_prompt = load_prompt(os.path.join(kit_dir, 'prompts/edgar_qna.yaml'))

        self.qa_chain.combine_documents_chain.llm_chain.prompt = custom_prompt

    def retrieval_conversational_chain(self) -> None:
        """Defines the conversational retrieval chain"""

        custom_condensed_question_prompt = load_prompt(
            os.path.join(kit_dir, 'prompts/edgar_multiturn-custom_condensed_question.yaml')
        )
        custom_qa_prompt = load_prompt(os.path.join(kit_dir, 'prompts/edgar_multiturn-custom_qa_prompt.yaml'))

        memory = ConversationSummaryMemory(
            llm=self.llm,
            buffer='The human and AI greet each other to start a conversation.',
            output_key='answer',
            memory_key='chat_history',
            return_messages=True,
        )
        assert self.vector_store is not None
        retriever = self.vector_store.as_retriever(search_kwargs={'k': 3})

        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=retriever,
            memory=memory,
            chain_type='stuff',
            return_source_documents=True,
            verbose=True,
            condense_question_prompt=custom_condensed_question_prompt,
            combine_docs_chain_kwargs={'prompt': custom_qa_prompt},
        )

    def retrieve_decomposed_subquestions(self, question: str) -> List[Any]:
        """Retrieves documents from generated subquestions based on a complex input question.
           Filters are applied according to the tickers related.

        Args:
            question (str): complex question

        Returns:
            list: list of retrieved documents
        """

        # customize output parser. Splits the LLM result into a list of questions
        n_generated_subquestions = self.query_decomposition_info['n_generated_subquestions']

        class LineListOutputParser(BaseOutputParser[List[str]]):
            """Output parser for a list of lines."""

            def parse(self, text: str) -> List[str]:
                lines = text.strip().split('\n')
                questions = [question.strip() for question in lines if '?' in question]
                return list(filter(None, questions))[:n_generated_subquestions]

        output_parser = LineListOutputParser()

        # load prompt for query decomposition
        query_decomposition_prompt = load_prompt(
            os.path.join(kit_dir, 'prompts/edgar_comparative_qna-query_decomposition_prompt.yaml')
        )

        # define chain and retriever
        llm_chain = LLMChain(llm=self.llm, prompt=query_decomposition_prompt, output_parser=output_parser)

        tickers = self.config.get('tickers', None)
        filter_rule = [{'company_ticker': {'$eq': ticker}} for ticker in tickers]
        assert self.vector_store is not None
        multiquery_retriever = MultiQueryRetriever(
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    'k': self.retrieval_info['n_retrieved_documents'],
                    'filter': {'$or': filter_rule},
                }
            ),
            llm_chain=llm_chain,
            parser_key='decomposed_questions',
            verbose=True,
        )

        # multiquestion retrievals
        multiquery_retrieved_docs = multiquery_retriever.get_relevant_documents(query=question)

        # format retrieved docs for summary context
        formatted_retrieved_docs = []
        for doc in multiquery_retrieved_docs:
            metadata_str = ', '.join(
                [f'{key}: {value}' for key, value in doc.metadata.items() if key in ('company_ticker', 'report_type')]
            )
            extended_page_content = f'Metadata: "{metadata_str}", Information: "{doc.page_content}"'
            extended_doc = Document(page_content=extended_page_content)
            formatted_retrieved_docs.append(extended_doc)

        return formatted_retrieved_docs

    def summarize_answer(self, multiquery_retrieved_docs: List[Any], question: str) -> Dict[str, Any]:
        """Summarizes an answer based on a list of retrieved documents and a question

        Args:
            multiquery_retrieved_docs (list): list of documents
            question (str): question to be answered

        Returns:
            dict: dictionary with a response structure. Contains the input documents with metadata and the output text
            requested.
        """

        # load prompt for answer summarization
        summarization_prompt = load_prompt(
            os.path.join(kit_dir, 'prompts/edgar_comparative_qna-answering_and_summarization_prompt.yaml')
        )

        # define chain
        llm_chain = LLMChain(llm=self.llm, prompt=summarization_prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='context')

        # get response
        response = stuff_chain.invoke({'input_documents': multiquery_retrieved_docs, 'original_question': question})

        return response

    def answer_comparative_multisource_question(self, question: str) -> Dict[str, Any]:
        """Answers a complex comparative query. Retrieves documents based on query decomposition and then summarizes
        the answer.

        Args:
            question (str): complex comparative question

        Returns:
            dict: dictionary with a response structure. Contains the input documents with metadata and the output text
            requested.
        """

        # retrieve documents with query decomposition
        multiquery_retrieved_docs = self.retrieve_decomposed_subquestions(question)

        # summarize retrieved docs answering the original question
        response = self.summarize_answer(multiquery_retrieved_docs, question)

        return response

    def retrieval_comparative_process(self) -> None:
        """Sets function to be used for documents comparison as object attribute"""

        self.comparative_process = self.answer_comparative_multisource_question

    def answer_sec(self, question: str) -> Any:
        """Answers a question

        Args:
            question (str): question

        Returns:
            str: answer to question
        """

        if self.qa_chain is not None:
            response = self.qa_chain.invoke(question)
        elif self.conversational_chain is not None:
            response = self.conversational_chain.invoke(question)
        elif self.comparative_process is not None:
            response = self.comparative_process(question)

        return response


if __name__ == '__main__':
    persist_vdb = os.path.join(kit_dir, 'data/vectordbs/tsla')
    config = {'persist_directory': persist_vdb, 'ticker': 'tsla'}
    sec_qa = SecFiling(config)
    sec_qa.init_llm_model()
    sec_qa.create_load_vector_store()
