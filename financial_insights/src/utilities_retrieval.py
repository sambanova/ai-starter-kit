import threading
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import streamlit
import yaml
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.base import RunnableBinding
from langchain_core.vectorstores.base import VectorStoreRetriever

from financial_insights.src.tools import time_llm
from financial_insights.streamlit.constants import *
from financial_insights.streamlit.utilities_app import _get_config_info
from utils.model_wrappers.api_gateway import APIGateway


class VectorstoreRegistry:
    """A thread-safe registry for managing in-memory vectorstores."""

    def __init__(self) -> None:
        self._vectorstore_registry: Dict[str, Chroma] = {}
        self._retriever_registry: Dict[str, VectorStoreRetriever] = {}
        self._lock = threading.Lock()

    def get_vectorstore(self, session_id: str) -> Chroma | None:
        """Retrieve the vectorstore for the given session_id."""
        with self._lock:
            return self._vectorstore_registry.get(session_id)

    def set_vectorstore(self, session_id: str, vectorstore: Chroma) -> None:
        """Set the vectorstore for the given session_id."""
        with self._lock:
            self._vectorstore_registry[session_id] = vectorstore

    def delete_vectorstore(self, session_id: str) -> None:
        """Delete the vectorstore for the given session_id."""
        with self._lock:
            if session_id in self._vectorstore_registry:
                del self._vectorstore_registry[session_id]

    def get_retriever(self, session_id: str) -> VectorStoreRetriever | None:
        """Retrieve the retriever for the given session_id."""
        with self._lock:
            return self._retriever_registry.get(session_id)

    def set_retriever(self, session_id: str, retriever: VectorStoreRetriever) -> None:
        """Set the retriever for the given session_id."""
        with self._lock:
            self._retriever_registry[session_id] = retriever

    def delete_retriever(self, session_id: str) -> None:
        """Delete the retriever for the given session_id."""
        with self._lock:
            if session_id in self._retriever_registry:
                del self._retriever_registry[session_id]


def get_qa_response(user_request: str, documents: List[Document], session_id: Optional[str] = None) -> Any:
    """
    Elaborate an answer to user request using RetrievalQA chain.

    Args:
        user_request: User request to answer.
        documents: List of documents to use for retrieval.
        session_id: Optional unique identifier for the user session.

    Returns:
        Answer to the user request.

    Raises:
        TypeError: If user request is not string or documents are not list of strings.
    """
    assert isinstance(user_request, str), TypeError('user_request must be a string.')
    assert isinstance(documents, list), TypeError(f'documents must be a list of strings. Got {type(documents)}.')
    assert all(isinstance(doc, Document) for doc in documents), TypeError(
        f'All documents must be of type `langchain.schema.Document`.'
    )

    # Global instance of the registry
    vectorstore_registry = VectorstoreRegistry()

    # Get the vectostore object `as retriever``
    vectorstore_registry, session_id = get_retriever(documents, vectorstore_registry, session_id)

    # Get the QA chain from the retriever
    assert isinstance(vectorstore_registry.get_retriever(session_id), VectorStoreRetriever), f'No retriever was build for session {session_id}.'
    qa_chain = get_qa_chain(vectorstore_registry.get_retriever(session_id))  # type: ignore

    # Function to answer questions based on the input documents
    response = invoke_qa_chain(qa_chain, user_request)

    # Clean up the vectorstore from the registry after usage
    vectorstore_registry.delete_vectorstore(session_id)

    # Clean the retriever after usage
    vectorstore_registry.delete_retriever(session_id)

    return response


@time_llm
def invoke_qa_chain(qa_chain: RunnableBinding[Dict[str, Any], Dict[str, Any]], user_query: str) -> Dict[str, Any]:
    """Invoke the chain to answer the question using RAG."""
    return qa_chain.invoke({'input': user_query})


def get_retrieval_config_info() -> Tuple[Any, Any]:
    """Loads RAG json config file."""
    # Read config file
    with open(CONFIG_PATH, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    api_info = config['llm']['api']
    embedding_model_info = config['rag']['embedding_model']
    retrieval_info = config['rag']['retrieval']
    prod_mode = config['prod_mode']

    return embedding_model_info, retrieval_info


def load_embedding_model(embedding_model_info: Dict[str, Any]) -> HuggingFaceEmbeddings | Embeddings:
    """Load the embedding model following the config information."""
    if embedding_model_info['type'] == 'cpu':
        embeddings_cpu = SentenceTransformerEmbeddings(model_name='paraphrase-mpnet-base-v2')
        return embeddings_cpu
    elif embedding_model_info['type'] == 'sambastudio':
        embeddings_sambastudio = APIGateway.load_embedding_model(
            type=embedding_model_info['type'],
            batch_size=embedding_model_info['batch_size'],
            coe=embedding_model_info['coe'],
            select_expert=embedding_model_info['select_expert'],
        )
        return embeddings_sambastudio
    else:
        raise ValueError(
            f'`config.rag["embedding_model"]["type"]` can only be `cpu` or `sambastudio. '
            'Got {embedding_model_info["type"]}.'
        )


def get_retriever(
    documents: List[Document],
    vectorstore_registry: VectorstoreRegistry = VectorstoreRegistry(),
    session_id: Optional[str] = None,
) -> Tuple[VectorstoreRegistry, str]:
    """
    Get the retriever for a given session id and documents.

    Args:
        documents: List of documents to be used for retrieval.
        vectorstore_registry: Registry of vectorstores to be used in retrieval.
            Defaults to an empty registry.
        session_id: Optional unique identifier of the session.
            Defaults to None.

    Returns:
        A tuple with the vectorstore registry and the current session id.
    """
    # Generate a unique identifier for the session if not provided
    if session_id is None:
        session_id = str(uuid4())

     # Load config
    config = _get_config_info(CONFIG_PATH)

    # Check if vectorstore and retriever exist for this session
    vectorstore = vectorstore_registry.get_vectorstore(session_id)
    retriever = vectorstore_registry.get_retriever(session_id)

    if vectorstore is None:
        # Retrieve RAG config information
        embedding_model_info, retrieval_info = get_retrieval_config_info()

        # Instantiate the embedding model
        embedding_model = load_embedding_model(embedding_model_info)

        # Instantiate the vectorstore with an explicit in-memory configuration
        vectorstore_registry.set_vectorstore(
            session_id, Chroma.from_documents(documents, embedding_model, persist_directory=None)
        )

    # Get the vectorstore for this session
    vectorstore = vectorstore_registry.get_vectorstore(session_id)
    assert isinstance(vectorstore, Chroma), f'Could not retrieve vectorstore for session_id {session_id}.'

    # Instantiate the retriever
    retriever = vectorstore.as_retriever(
            search_kwargs={
                'k': retrieval_info['k_retrieved_documents'],
            },
        )

    assert isinstance(retriever, VectorStoreRetriever), f'Could not retrieve retriever for session_id {session_id}.'

    # Instantiate the retriever
    vectorstore_registry.set_retriever(session_id=session_id, retriever=retriever)

    # Delete local vectorstore and retriever
    del vectorstore, retriever

    return vectorstore_registry, session_id


def get_qa_chain(retriever: VectorStoreRetriever) -> Any:
    """
    Get a retrieval QA chain using the provided vectorstore `as retriever`.

    Args:
        retriever: Retriever to use for the QA chain.

    Returns:
        A retrieval QA chain using the provided retriever.
    """
    assert isinstance(retriever, VectorStoreRetriever), f'`retriever` should be a `langchain_core.vectorstores.base.VectorStoreRetriever`. Got type {type(retriever)}.'
    # See full prompt at https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    retrieval_qa_chat_prompt = hub.pull('langchain-ai/retrieval-qa-chat')

    # Create a retrieval-based QA chain
    combine_docs_chain = create_stuff_documents_chain(streamlit.session_state.fc.llm, retrieval_qa_chat_prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return qa_chain
