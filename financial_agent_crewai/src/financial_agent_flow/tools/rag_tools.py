import logging
from typing import Any, List

import chromadb
import pandas
from crewai import LLM
from crewai.tools import BaseTool
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from financial_agent_crewai.src.financial_agent_flow.config import *

# Main text processing, RAG, and web scraping constants
MIN_CHUNK_SIZE = 4
MAX_CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256

logger = logging.getLogger(__name__)


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


class TXTSearchToolSchema(BaseModel):
    """Input for TXTSearchTool."""

    txt: str = Field(..., description='Mandatory txt path you want to search.')


class TXTSearchTool(BaseTool):  # type: ignore
    name: str = "Search a txt's content."
    description: str = "A tool that can be used to semantic search a query from a txt's content."
    txt_path: TXTSearchToolSchema
    rag_llm: Any

    def _run(self, query: str) -> Any:
        """Execute the search query and return results"""

        # Extract the datafrane from the txt path
        df = pandas.read_csv(self.txt_path.txt)

        # Convert the dataframe rows into Document objects
        documents = list()
        for _, row in df.iterrows():
            document = Document(page_content=row.values[0])
            documents.append(document)

        # Get QA response
        answer = get_qa_response(query=query, documents=documents, rag_llm=self.rag_llm)

        return answer


def get_qa_response(
    query: str,
    documents: List[Document],
    rag_llm: LLM,
) -> Any:
    """
    Elaborate an answer to user request using RetrievalQA chain.

    Args:
        query: User request to answer.
        documents: List of documents to use for retrieval.

    Returns:
        Answer to the user request.

    Raises:
        TypeError: If `query` is not a string or `documents` is not a list of `langchain.schema.Document`.
    """
    if not isinstance(query, str):
        raise TypeError('user_request must be a string.')
    if not isinstance(documents, list):
        raise TypeError(f'documents must be a list of strings. Got {type(documents)}.')
    if not all(isinstance(doc, Document) for doc in documents):
        raise TypeError(f'All documents must be of type `langchain.schema.Document`.')

    # Get the vectostore registry
    vectorstore = get_vectorstore(documents=documents)

    # Retrieve the most relevant docs
    retrieved_docs = vectorstore.similarity_search(query=query, k=NUM_RAG_SOURCES)

    # Extract the content of the retrieved docs
    docs_content = '\n\n'.join(doc.page_content for doc in retrieved_docs)

    # Prompt template
    retrieval_qa_chat_prompt_template = PromptTemplate.from_template(RETRIEVAL_QA_PROMPT_TEMPLATE)

    # Prompt
    retrieval_qa_chat_prompt = retrieval_qa_chat_prompt_template.format(input=query, context=docs_content)

    # Call the LLM
    response = rag_llm.call(retrieval_qa_chat_prompt)

    return response


def load_embedding_model() -> HuggingFaceEmbeddings | Embeddings:
    """Load the embedding model following the config information."""

    embeddings_cpu = HuggingFaceEmbeddings(model_name='paraphrase-mpnet-base-v2')
    return embeddings_cpu


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

    # Instantiate the embedding model
    embedding_model = load_embedding_model()

    # Instantiate the vectorstore with an explicit in-memory configuration
    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache()  # type: ignore
        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory=None)
    except:
        raise Exception('Could not instantiate the vectorstore.')

    if not isinstance(vectorstore, Chroma):
        raise Exception('Could not instantiate the vectorstore.')

    return vectorstore
