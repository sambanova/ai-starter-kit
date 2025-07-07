from typing import Any, Dict, List, Tuple

import chromadb
import yaml
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from financial_assistant.constants import *
from financial_assistant.src.utilities import _get_config_info, get_logger, time_llm
from financial_assistant.streamlit.llm_model import sambanova_llm
from utils.model_wrappers.api_gateway import APIGateway

logger = get_logger()

RETRIEVAL_QA_PROMPT_TEMPLATE = """
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


@time_llm
def get_qa_response(user_request: str, documents: List[Document]) -> Dict[str, str | List[Document]]:
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
    if not isinstance(user_request, str):
        raise TypeError('user_request must be a string.')
    if not isinstance(documents, list):
        raise TypeError(f'documents must be a list of strings. Got {type(documents)}.')
    if not all(isinstance(doc, Document) for doc in documents):
        raise TypeError(f'All documents must be of type `langchain.schema.Document`.')

    # Get the vectostore registry
    vectorstore = get_vectorstore(documents=documents)

    # Retrieve the most relevant docs
    retrieved_docs = vectorstore.similarity_search(query=user_request, k=TOP_K)

    # Extract the content of the retrieved docs
    docs_content = '\n\n'.join(doc.page_content for doc in retrieved_docs)

    # Prompt template
    retrieval_qa_chat_prompt_template = PromptTemplate.from_template(RETRIEVAL_QA_PROMPT_TEMPLATE)

    # Prompt
    retrieval_qa_chat_prompt = retrieval_qa_chat_prompt_template.format(input=user_request, context=docs_content)

    # Call the LLM
    answer = sambanova_llm.llm.invoke(retrieval_qa_chat_prompt)

    # Build the response with the answer and the urls
    response: Dict[str, str | List[Document]] = dict()
    if isinstance(answer, AIMessage) and isinstance(answer.content, str):
        response['answer'] = answer.content
    else:
        response['answer'] = ''
    response['context'] = retrieved_docs

    return response


def get_retrieval_config_info() -> Tuple[Any, Any]:
    """Loads RAG json config file."""
    # Read config file
    with open(CONFIG_PATH, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    prod_mode = config['prod_mode']

    embedding_model_info = config['rag']['embedding_model']
    retrieval_info = config['rag']['retrieval']

    return embedding_model_info, retrieval_info


def load_embedding_model(embedding_model_info: Dict[str, Any]) -> HuggingFaceEmbeddings | Embeddings:
    """Load the embedding model following the config information."""

    if embedding_model_info['type'] == 'cpu':
        embeddings_cpu = HuggingFaceEmbeddings(model_name='paraphrase-mpnet-base-v2')
        return embeddings_cpu
    elif embedding_model_info['type'] == 'sambastudio' or embedding_model_info['type'] == 'sncloud':
        embeddings_sambastudio = APIGateway.load_embedding_model(
            type=embedding_model_info.get('type'),
            batch_size=embedding_model_info.get('batch_size'),
            bundle=embedding_model_info.get('bundle'),
            model=embedding_model_info.get('model'),
        )
        return embeddings_sambastudio
    else:
        raise ValueError(
            f'`config.rag["embedding_model"]["type"]` can only be `cpu` , `sncloud `or `sambastudio. '
            f'Got {embedding_model_info["type"]}.'
        )


def get_vectorstore(documents: List[Document]) -> Chroma:
    """
    Get the retriever for a given session id and documents.

    Args:
        documents: List of documents to be used for retrieval.

    Returns:
        The vectorstore.

    Raises:
        Exception: If a vectorstore cannot be instantiated.
    """
    # Load config
    config = _get_config_info(CONFIG_PATH)

    # Retrieve RAG config information
    embedding_model_info, retrieval_info = get_retrieval_config_info()

    # Instantiate the embedding model
    embedding_model = load_embedding_model(embedding_model_info)

    # Instantiate the vectorstore with an explicit in-memory configuration
    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache()  # type: ignore
        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory=None)
    except:
        raise Exception('Could not instantiate the vectorstore.')

    if not isinstance(vectorstore, Chroma):
        raise Exception('Could not instantiate the vectorstore.')

    return vectorstore
