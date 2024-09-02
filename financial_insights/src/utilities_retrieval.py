from typing import Any, List

import streamlit
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables.base import RunnableBinding

from financial_insights.src.tools import time_llm
from financial_insights.streamlit.constants import *
from financial_insights.streamlit.utilities_app import _get_config_info
from typing import Dict

def get_qa_response(
    user_request: str,
    documents: List[Document],
) -> Any:
    """
    Elaborate an answer to user request using RetrievalQA chain.

    Args:
        user_request: User request to answer.
        documents: List of documents to use for retrieval.

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

    # Retrieve RAG config information
    embedding_model_info, retrieval_info = get_retrieval_config_info()

    # Instantiate the embedding model
    embedding_model = load_embedding_model(embedding_model_info)

    # Instantiate the vectorstore
    vectorstore = Chroma.from_documents(documents, embedding_model)

    # Load config
    config = _get_config_info(CONFIG_PATH)

    # Instantiate the retriever
    retriever = vectorstore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            'score_threshold': retrieval_info['score_threshold'],  # type: ignore
            'k': retrieval_info['k_retrieved_documents'],  # type: ignore
        },
    )

    # See full prompt at https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    retrieval_qa_chat_prompt = hub.pull('langchain-ai/retrieval-qa-chat')

    # Create a retrieval-based QA chain
    combine_docs_chain = create_stuff_documents_chain(streamlit.session_state.fc.llm, retrieval_qa_chat_prompt)
    qa_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    # Function to answer questions based on the input documents
    response = invoke_qa_chain(qa_chain, user_request)

    return response


@time_llm
def invoke_qa_chain(qa_chain: RunnableBinding[Dict[str, Any], Dict[str, Any]], user_query: str) -> Dict[str, Any]:
    """Invoke the chain to answer the question using RAG."""
    return qa_chain.invoke({'input': user_query})


def get_retrieval_config_info(self):
    """Loads RAG json config file."""
    # Read config file
    with open(CONFIG_PATH, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    api_info = config.llm["api"]
    embedding_model_info = config.rag["embedding_model"]
    retrieval_info = config.rag["retrieval"]
    prod_mode = config["prod_mode"]
    
    return api_info, embedding_model_info, retrieval_info

def load_embedding_model(embedding_model_info):
    """Load the embedding model following the config information."""
    if embedding_model_info['type'] == 'cpu':
        embeddings = SentenceTransformerEmbeddings(model_name='paraphrase-mpnet-base-v2')
    elif embedding_model_info['type'] == 'sambastudio':
        embeddings = APIGateway.load_embedding_model(
            type=embedding_model_info["type"],
            batch_size=embedding_model_info["batch_size"],
            coe=embedding_model_info["coe"],
            select_expert=embedding_model_info["select_expert"]
            ) 
    else:
        raise ValueError(f'`config.rag["embedding_model"]["type"]` can only be `cpu` or `sambastudio. Got {embedding_model_info["type"]}.')
    return embeddings 
