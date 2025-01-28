import logging
from typing import Any, Dict, List, Tuple

import pandas
from crewai.tools import BaseTool
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableBinding
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from financial_assistant.prompts.retrieval_prompts import QA_RETRIEVAL_PROMPT_TEMPLATE
from financial_assistant.src.exceptions import VectorStoreException

# Main text processing, RAG, and web scraping constants
MIN_CHUNK_SIZE = 4
MAX_CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256

logger = logging.getLogger(__name__)


QA_RETRIEVAL_PROMPT_TEMPLATE = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Answer any use questions based solely on the context below:

    <context>

    {context}

    </context>

    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>

    {input}

    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
"""


def get_qa_response(
    user_query: str,
    documents: List[Document],
    rag_llm: LLM,
) -> Any:
    """
    Elaborate an answer to user request using RetrievalQA chain.

    Args:
        user_request: User request to answer.
        documents: List of documents to use for retrieval.

    Returns:
        Answer to the user request.

    Raises:
        TypeError: If `user_request` is not a string or `documents` is not a list of `langchain.schema.Document`.
    """
    if not isinstance(user_query, str):
        raise TypeError('user_request must be a string.')
    if not isinstance(documents, list):
        raise TypeError(f'documents must be a list of strings. Got {type(documents)}.')
    if not all(isinstance(doc, Document) for doc in documents):
        raise TypeError(f'All documents must be of type `langchain.schema.Document`.')

    # Get the vectostore registry
    vectorstore, retriever = get_vectorstore_retriever(documents=documents)

    # Get the QA chain from the retriever
    qa_chain = get_qa_chain(retriever, rag_llm=rag_llm)

    # Invoke the QA chain to get an answer to the user
    response = invoke_qa_chain(qa_chain=qa_chain, user_query=user_query)

    return response


def invoke_qa_chain(
    qa_chain: RunnableBinding[Dict[str, Any], Dict[str, Any]],
    user_query: str,
) -> Dict[str, Any]:
    """Invoke the chain to answer the question using RAG."""
    return qa_chain.invoke({'input': user_query})


def load_embedding_model() -> HuggingFaceEmbeddings | Embeddings:
    """Load the embedding model following the config information."""

    embeddings_cpu = HuggingFaceEmbeddings(model_name='paraphrase-mpnet-base-v2')
    return embeddings_cpu


def get_vectorstore_retriever(documents: List[Document]) -> Tuple[Chroma, VectorStoreRetriever]:
    """
    Get the retriever for a given session id and documents.

    Args:
        documents: List of documents to be used for retrieval.
        vectorstore_registry: Registry of vectorstores to be used in retrieval.
            Defaults to an empty registry.

    Returns:
        A tuple with the vectorstore registry and the current session id.

    Raisese:
        Exception: If a vectorstore and a retriever cannot be instantiated.
    """

    # Instantiate the embedding model
    embedding_model = load_embedding_model()

    # Instantiate the vectorstore with an explicit in-memory configuration
    try:
        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory=None)
    except:
        raise VectorStoreException('Could not instantiate the vectorstore.')

    if not isinstance(vectorstore, Chroma):
        raise Exception('Could not instantiate the vectorstore.')

    # Instantiate the retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={
            'k': 5,
        },
    )

    if not isinstance(retriever, VectorStoreRetriever):
        raise Exception(f'Could not retrieve the retriever.')

    return vectorstore, retriever


def get_qa_chain(retriever: VectorStoreRetriever, rag_llm: LLM) -> Any:
    """
    Get a retrieval QA chain using the provided vectorstore `as retriever`.

    Args:
        retriever: Retriever to use for the QA chain.

    Returns:
        A retrieval QA chain using the provided retriever.

    Raises:
        TypeError: If `retriever` is not of type `langchain_core.vectorstores.base.VectorStoreRetriever`.
    """
    if not isinstance(retriever, VectorStoreRetriever):
        raise TypeError(
            '`retriever` should be a `langchain_core.vectorstores.base.VectorStoreRetriever`. '
            f'Got type {type(retriever)}.'
        )

    # The Retrieval QA prompt
    retrieval_qa_chat_prompt = PromptTemplate.from_template(
        template=QA_RETRIEVAL_PROMPT_TEMPLATE,
    )

    # Create a retrieval-based QA chain
    combine_docs_chain = create_stuff_documents_chain(rag_llm, retrieval_qa_chat_prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return qa_chain


class TXTSearchToolSchema(BaseModel):
    """Input for TXTSearchTool."""

    txt: str = Field(..., description='Mandatory txt path you want to search.')


class TXTSearchTool(BaseTool):  # type: ignore
    name: str = "Search a txt's content."
    description: str = "A tool that can be used to semantic search a query from a txt's content."
    txt_path: TXTSearchToolSchema
    rag_llm: LLM

    def _run(self, search_query: str) -> Any:
        """Execute the search query and return results"""
        df = pandas.read_csv(self.txt_path.txt)
        # Convert DataFrame rows into Document objects
        documents = list()
        for _, row in df.iterrows():
            document = Document(page_content=row['text'])
            documents.append(document)

        # Get QA response
        answer = get_qa_response(user_query=search_query, documents=documents, rag_llm=self.rag_llm)['answer']

        return answer
