import os
import sys
sys.path.append("../../")

from langchain.memory import ConversationSummaryMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, load_prompt
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from sec_edgar_downloader import Downloader
from xbrl import XBRLParser

from dotenv import load_dotenv
load_dotenv('../../export.env')

from vectordb.vector_db import VectorDb
from utils.sambanova_endpoint import SambaNovaEndpoint

EMAIL = "mlengineer@snova_dummy.ai"
COMPANY = "snova_dummy"
DATA_DIRECTORY = "../data"

LAST_N_DOCUMENTS = 1
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS_TO_GENERATE = 500
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 240
DB_TYPE = "chroma"

K_RETRIEVED_DOCUMENTS = 3


class SecFiling:
    """Class that handles SEC Filing data set creation as vector database and retrieving information in different ways.
    """
    def __init__(self, config: dict = {}):
        """Initializes SecFiling class
        
        Args:
            config (dict, optional): config object with entries like filing ticker and vector db persistent directory. Defaults to {}.
        """
        
        self.config = config
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.conversational_chain = None
        self.retriever = None

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

        
    def download_sec_data(self, ticker: str) -> list:
        """Downloads SEC data based on ticker name

        Args:
            ticker (str): name of the ticker

        Raises:
            Exception: related to fetching data from Edgar's db

        Returns:
            list: list of loaded documents
        """
        
        try:
            dl = Downloader(COMPANY, EMAIL, DATA_DIRECTORY)
            dl.get("10-K", ticker, limit=LAST_N_DOCUMENTS)
        except Exception as ex:
            raise Exception(
                f"Failed to fetch data for {ticker} from Edgar database"
            ) from ex
            
        sec_dir = f"{DATA_DIRECTORY}/sec-edgar-filings/{ticker}"
        dir_loader = DirectoryLoader(
            sec_dir, glob="**/*.txt", loader_cls=TextLoader
        )
        documents = dir_loader.load()
        
        return documents
    
    def parse_xbrl_data(self, documents: list) -> Document:
        """Parses XBRL data from a list of documents

        Args:
            documents (list): list of documents

        Returns:
            Document: document that consolidates all parsed documents
        """
        
        xbrl_parser = XBRLParser()
        xbrl_texts = []
        for document in documents:
            xbrl_doc = xbrl_parser.parse(document.metadata['source'])
            p_span_tags = xbrl_doc.find_all(lambda x: x.name == 'div' and x.find('span'))
            xbrl_text = ' '.join(tag.get_text() for tag in p_span_tags)  
            xbrl_texts.append(xbrl_text)
        
        all_document_texts = '\n\n'.join(xbrl_texts)
        all_parsed_documents =  Document(page_content=all_document_texts, metadata={})
        
        return all_parsed_documents

    def create_load_vector_store(self, force_reload: bool = False) -> None:
        """Creates or loads a vector db if it exists

        Args:
            force_reload (bool, optional): force to reload/recreate vector db. Defaults to False.
        """
        
        persist_directory = self.config.get("persist_directory", None)
        ticker = self.config.get("ticker", None)
        
        vectordb = VectorDb()
        
        embeddings = vectordb.load_embedding_model()

        if persist_directory and os.path.exists(persist_directory) and not force_reload:
            self.vector_store = vectordb.load_vdb(persist_directory, embeddings)
        
        else:
            documents = self.download_sec_data(ticker)
            complete_document = self.parse_xbrl_data(documents)
            chunks = vectordb.get_text_chunks([complete_document], CHUNK_SIZE, CHUNK_OVERLAP)
            self.vector_store = vectordb.create_vector_store(chunks, embeddings, DB_TYPE, persist_directory)

    def retrieval_qa_chain(self) -> None:
        """Defines the retrieval chain
        """
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": K_RETRIEVED_DOCUMENTS})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            input_key="question",
            output_key="answer",
            return_source_documents=True,
        )

        custom_prompt = load_prompt("../prompts/llama7b-edgar_qna.yaml")

        self.qa_chain.combine_documents_chain.llm_chain.prompt = custom_prompt
        
    def retrieval_conversational_chain(self):
        """Defines the conversational retrieval chain
        """
        
        custom_condensed_question_prompt = load_prompt("../prompts/llama7b-edgar_multiturn-custom_condensed_question.yaml")
        custom_qa_prompt = load_prompt("../prompts/llama7b-edgar_multiturn-custom_qa_prompt.yaml")

        memory = ConversationSummaryMemory(
            llm=self.llm, 
            max_token_limit=50,
            buffer="The human and AI greet each other to start a conversation.",
            output_key='answer',
            memory_key='chat_history',
            return_messages=True,
        )

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            self.llm, 
            retriever=retriever, 
            memory=memory, 
            chain_type="stuff",
            return_source_documents=True, 
            verbose=True,
            condense_question_prompt = custom_condensed_question_prompt,
            combine_docs_chain_kwargs={'prompt': custom_qa_prompt}
        )

    def answer_sec(self, question: str) -> str:
        """Answers a question

        Args:
            question (str): question

        Returns:
            str: answer to question
        """
        
        if self.qa_chain is not None:
            response = self.qa_chain(question)
        elif self.conversational_chain is not None:
            response = self.conversational_chain(question)
        
        return response

if __name__ == '__main__':
    persist_vdb = "../data/vectordbs/tsla"
    config = {'persist_directory': persist_vdb, 'ticker': 'tsla'}
    sec_qa = SecFiling(config)
    sec_qa.init_llm_model()
    sec_qa.create_load_vector_store()