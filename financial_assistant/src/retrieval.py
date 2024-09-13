from typing import Any, Dict, List, Optional, Tuple

import streamlit
import yaml
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableBinding
from langchain_core.vectorstores.base import VectorStoreRetriever

from financial_assistant.prompts.retrieval_prompts import QA_RETRIEVAL_PROMPT_TEMPLATE
from financial_assistant.src.tools import time_llm
from financial_assistant.streamlit.constants import *
from financial_assistant.streamlit.utilities_app import _get_config_info
from utils.model_wrappers.api_gateway import APIGateway


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

    # Get the vectostore registry and the current `session_id`
    vectorstore, retriever = get_vectorstore_retriever(documents)

    # Get the QA chain from the retriever
    qa_chain = get_qa_chain(retriever)

    # Invoke the QA chain to get an answer to the user
    response = invoke_qa_chain(qa_chain, user_request)

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
            f'Got {embedding_model_info["type"]}.'
        )


def get_vectorstore_retriever(
    documents: List[Document],
) -> Tuple[Chroma, VectorStoreRetriever]:
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
    # Load config
    config = _get_config_info(CONFIG_PATH)

    # Retrieve RAG config information
    embedding_model_info, retrieval_info = get_retrieval_config_info()

    # Instantiate the embedding model
    embedding_model = load_embedding_model(embedding_model_info)

    # Instantiate the vectorstore with an explicit in-memory configuration
    try:
        vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=None)
    except:
        streamlit.error('Could not instantiate the vectorstore.')
        streamlit.stop()

    assert isinstance(vectorstore, Chroma), f'Could not instantiate the vectorstore.'

    # Instantiate the retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={
            'k': retrieval_info['k_retrieved_documents'],
        },
    )

    assert isinstance(retriever, VectorStoreRetriever), f'Could not retrieve the retriever.'

    return vectorstore, retriever


def get_qa_chain(retriever: VectorStoreRetriever) -> Any:
    """
    Get a retrieval QA chain using the provided vectorstore `as retriever`.

    Args:
        retriever: Retriever to use for the QA chain.

    Returns:
        A retrieval QA chain using the provided retriever.
    """
    assert isinstance(
        retriever, VectorStoreRetriever
    ), f'`retriever` should be a `langchain_core.vectorstores.base.VectorStoreRetriever`. Got type {type(retriever)}.'

    # The Retrieval QA prompt
    retrieval_qa_chat_prompt = PromptTemplate.from_template(
        template=QA_RETRIEVAL_PROMPT_TEMPLATE,
    )

    # Create a retrieval-based QA chain
    combine_docs_chain = create_stuff_documents_chain(streamlit.session_state.llm.llm, retrieval_qa_chat_prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return qa_chain
