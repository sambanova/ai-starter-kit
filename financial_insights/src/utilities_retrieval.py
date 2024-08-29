from typing import Any, List

import streamlit
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from financial_insights.streamlit.constants import *
from financial_insights.streamlit.utilities_app import _get_config_info


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

    # TODO
    # Add an option to use E5 from our endpoints.

    # Instantiate the embedding model
    embedding_model = SentenceTransformerEmbeddings(model_name='paraphrase-mpnet-base-v2')

    # Instantiate the vectorstore
    vectorstore = Chroma.from_documents(documents, embedding_model)

    # Load config
    config = _get_config_info(CONFIG_PATH)

    # Instantiate the retriever
    retriever = vectorstore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            'score_threshold': config['rag']['retrieval']['score_threshold'],  # type: ignore
            'k': config['rag']['retrieval']['k_retrieved_documents'],  # type: ignore
        },
    )

    # See full prompt at https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    retrieval_qa_chat_prompt = hub.pull('langchain-ai/retrieval-qa-chat')

    # Create a retrieval-based QA chain
    combine_docs_chain = create_stuff_documents_chain(streamlit.session_state.fc.llm, retrieval_qa_chat_prompt)
    qa_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    # Function to answer questions based on the input documents
    response = qa_chain.invoke({'input': user_request})

    return response
