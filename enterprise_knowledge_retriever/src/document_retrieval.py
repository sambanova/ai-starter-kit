import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import nltk
import torch
import yaml
from dotenv import load_dotenv
from langchain_classic.chains.base import Chain
from langchain_classic.docstore.document import Document
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_sambanova import ChatSambaNova, SambaNovaEmbeddings
from pydantic import SecretStr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.vectordb.vector_db import VectorDb

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir, 'data/my-vector-db')

load_dotenv(os.path.join(repo_dir, '.env'), override=True)

from utils.parsing.sambaparse import parse_doc_universal

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG)
    format='%(asctime)s [%(levelname)s] - %(message)s',  # Define the log message format
)
# Create a logger object
logger = logging.getLogger(__name__)

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def load_chat_prompt(path: str) -> ChatPromptTemplate:
    """Load chat prompt from yaml file"""

    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    config.pop('_type')

    template = config.pop('template')

    if not template:
        msg = "Can't load chat prompt without template"
        raise ValueError(msg)

    messages = []
    if isinstance(template, str):
        messages.append(('human', template))

    elif isinstance(template, list):
        for item in template:
            messages.append((item['role'], item['content']))

    return ChatPromptTemplate(messages=messages, **config)


class RetrievalQAChain(Chain):
    """class for question-answering.

    Do retrieval over relevant documents in the vectorstore set in the retriever

    If conversational enabled, before doing the retrieval QA call
    the llm is called to rephrase original user query to  include relevant details in the history
    then the rephrased query is used as input to the retriever and the QA call is done using
    relevant documents, then the user query and final answer is added to the history and history
    is summarized using the llm

    When reranking enabled, reranker model is used to filter final_k_retrieved_documents
    """

    retriever: BaseRetriever
    rerank: bool = True
    llm: BaseChatModel
    qa_prompt: ChatPromptTemplate
    final_k_retrieved_documents: int = 3
    conversational: bool = False
    # wether or not to use memory and answer over reformulated query with history summary
    # instead of answer over raw user query with our using history
    summary_prompt: Optional[ChatPromptTemplate] = None
    condensed_query_prompt: Optional[ChatPromptTemplate] = None

    @property
    def input_keys(self) -> List[str]:
        """Input keys.
        :meta private:
        """
        return ['question']

    @property
    def output_keys(self) -> List[str]:
        """Output keys.
        :meta private:
        """
        return ['answer', 'source_documents']

    def __init__(self, **kwargs: Any) -> None:
        """init and validate environment variables"""
        super().__init__(**kwargs)
        if self.conversational:
            self.init_memory()

    def _format_docs(self, docs: List[Document]) -> str:
        return '\n\n'.join(doc.page_content for doc in docs)

    def rerank_docs(self, query: str, docs: List[Document], final_k: int) -> List[Document]:
        # Lazy hardcoding for now
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')  # type: ignore[no-untyped-call]
        reranker = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
        pairs = []
        for d in docs:
            pairs.append([query, d.page_content])

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            )
            scores = (
                reranker(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )

        scores_list = scores.tolist()
        scores_sorted_idx = sorted(range(len(scores_list)), key=lambda k: scores_list[k], reverse=True)

        docs_sorted = [docs[k] for k in scores_sorted_idx]
        # docs_sorted = [docs[k] for k in scores_sorted_idx if scores_list[k]>0]
        docs_sorted = docs_sorted[:final_k]

        return docs_sorted

    def init_memory(self) -> None:
        """
        Initialize conversation summary memory for the conversation
        """
        assert self.summary_prompt is not None
        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            buffer='The conversation just started',
            memory_key='chat_history',
            return_messages=True,
            output_key='answer',
            prompt=self.summary_prompt,
        )

    def update_memory(self, query: str, response: str) -> None:
        assert self.memory is not None
        self.memory.save_context(inputs={'input': query}, outputs={'answer': response})

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
        assert self.condensed_query_prompt is not None
        custom_condensed_question_prompt = self.condensed_query_prompt
        assert self.memory is not None
        history = self.memory.load_memory_variables({})
        logger.info(f'HISTORY: {history}')
        reformulation_chain = custom_condensed_question_prompt | self.llm | StrOutputParser()
        reformulated_query = reformulation_chain.invoke(input={'chat_history': history, 'question': query})
        return reformulated_query

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """ "call the Retrieval QA Chain"""

        qa_chain = self.qa_prompt | self.llm | StrOutputParser()

        logger.info(f'USER QUERY: {inputs["question"]}')

        # when conversational enabled, before retrieval
        # the user query is reformulated using stored history summary
        if self.conversational:
            query = self.reformulate_query_with_history(inputs['question'])
            logger.info(f'REFORMULATED QUERY: {query}')
        else:
            query = inputs['question']

        documents = self.retriever.invoke(query)
        if self.rerank:
            documents = self.rerank_docs(query, documents, self.final_k_retrieved_documents)
        docs = self._format_docs(documents)
        response: Dict[str, Any] = {}
        response['answer'] = qa_chain.invoke({'question': query, 'context': docs})
        response['source_documents'] = documents

        # Update memory when conversational mode is enabled
        if self.conversational:
            threading.Thread(target=self.update_memory, args=(inputs['question'], response['answer'])).start()

        return response


class DocumentRetrieval:
    def __init__(self, sambanova_api_key: str) -> None:
        self.vectordb = VectorDb()
        config_info = self.get_config_info()
        self.llm_info: Dict[str, Any] = config_info[0]
        self.embedding_model_info: Dict[str, Any] = config_info[1]
        self.retrieval_info: Dict[str, Any] = config_info[2]
        self.prompts: Dict[str, Any] = config_info[3]
        self.prod_mode: bool = config_info[4]
        self.pdf_only_mode: bool = config_info[5]
        self.retriever = None
        self.sambanova_api_key = SecretStr(sambanova_api_key)
        self.set_llm()

    def get_config_info(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, str], bool, bool]:
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        llm_info = config['llm']
        embedding_model_info = config['embedding_model']
        retrieval_info = config['retrieval']
        prompts = config['prompts']
        prod_mode = config['prod_mode']
        pdf_only_mode = config['pdf_only_mode']

        return llm_info, embedding_model_info, retrieval_info, prompts, prod_mode, pdf_only_mode

    def set_llm(self, model: Optional[str] = None) -> None:
        """
        Sets the sambanova LLM

        Parameters:
        Model (str): The name of the model to use for the LLM (overwrites the param set in config).
        """
        if model is None:
            model = self.llm_info['model']
        llm_info = {k: v for k, v in self.llm_info.items() if k != 'model'}
        llm = ChatSambaNova(
            api_key=self.sambanova_api_key,
            **llm_info,
            model=model,
        )
        self.llm = llm

    def parse_doc(self, doc_folder: str, additional_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Parse specified documents and return a list of LangChain documents.

        Args:
            doc_folder (str): Path to the documents.
            additional_metadata (Optional[Dict], optional): Additional metadata to include in the processed documents.
                Defaults to an empty dictionary.

        Returns:
            List[Document]: A list of LangChain documents.
        """
        if additional_metadata is None:
            additional_metadata = {}

        _, _, langchain_docs = parse_doc_universal(
            doc=doc_folder, additional_metadata=additional_metadata, lite_mode=self.pdf_only_mode
        )

        return langchain_docs

    def load_embedding_model(self) -> Embeddings:
        embeddings = SambaNovaEmbeddings(api_key=self.sambanova_api_key, **self.embedding_model_info)
        return embeddings

    def create_vector_store(
        self,
        text_chunks: List[Document],
        embeddings: Any,
        output_db: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Any:
        logger.info(f'Created collection, name is {collection_name}')
        vectorstore = self.vectordb.create_vector_store(
            text_chunks,
            embeddings,
            output_db=output_db,
            collection_name=collection_name,
            db_type=self.retrieval_info['db_type'],
        )
        return vectorstore

    def load_vdb(self, db_path: str, embeddings: Any, collection_name: Optional[str] = None) -> Any:
        logger.info(f'Loading collection, name is {collection_name}')
        vectorstore = self.vectordb.load_vdb(
            db_path, embeddings, db_type=self.retrieval_info['db_type'], collection_name=collection_name
        )
        return vectorstore

    def init_retriever(self, vectorstore: Any) -> None:
        if self.retrieval_info['rerank']:
            self.retriever = vectorstore.as_retriever(
                search_type='similarity_score_threshold',
                search_kwargs={
                    'score_threshold': self.retrieval_info['score_threshold'],
                    'k': self.retrieval_info['k_retrieved_documents'],
                },
            )
        else:
            self.retriever = vectorstore.as_retriever(
                search_type='similarity_score_threshold',
                search_kwargs={
                    'score_threshold': self.retrieval_info['score_threshold'],
                    'k': self.retrieval_info['final_k_retrieved_documents'],
                },
            )

    def get_qa_retrieval_chain(self, conversational: bool = False) -> RetrievalQAChain:
        """
        Generate a qa_retrieval chain using a language model.

        This function uses a language model, specifically a SambaNova LLM, to generate a qa_retrieval chain
        based on the input vector store of text chunks.

        Parameters:
        conversational: wether or not to use memory
        when enabled user query is reformulated using history summary

        Returns:
        RetrievalQA: A chain ready for QA without memory
        """
        assert isinstance(self.retriever, VectorStoreRetriever), (
            f'The Retriever must be VectorStoreRetriever. Got type {type(self.retriever)}'
        )
        retrievalQAChain = RetrievalQAChain(
            retriever=self.retriever,
            llm=self.llm,
            qa_prompt=load_chat_prompt(os.path.join(repo_dir, self.prompts['qa_prompt'])),
            rerank=self.retrieval_info['rerank'],
            final_k_retrieved_documents=self.retrieval_info['final_k_retrieved_documents'],
            conversational=conversational,
            summary_prompt=load_chat_prompt(os.path.join(repo_dir, self.prompts['summary_prompt'])),
            condensed_query_prompt=load_chat_prompt(os.path.join(repo_dir, self.prompts['condensed_query_prompt'])),
        )
        return retrievalQAChain
