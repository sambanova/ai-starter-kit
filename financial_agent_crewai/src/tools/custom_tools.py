import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas
import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableBinding
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from sec_downloader import Downloader

from financial_assistant.prompts.retrieval_prompts import QA_RETRIEVAL_PROMPT_TEMPLATE
from financial_assistant.src.exceptions import VectorStoreException
from utils.model_wrappers.api_gateway import APIGateway

# Main text processing, RAG, and web scraping constants
MIN_CHUNK_SIZE = 4
MAX_CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256

# Instantiate the LLM
llm = APIGateway.load_llm(
    type='sncloud',
    streaming=False,
    bundle=True,
    do_sample=False,
    max_tokens_to_generate=1024,
    temperature=0.7,
    select_expert='Meta-Llama-3.1-70B-Instruct',
    process_prompt=False,
    sambanova_api_key=os.getenv('SAMBANOVA_API_KEY'),
)

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


class SecEdgarFilingsInput(BaseModel):
    """Tool for retrieving a financial filing from SEC Edgar and then answering the original user question."""

    query: str = Field(..., description='The original user question.')
    ticker_symbol: str = Field(..., description='The company ticker symbol.')
    filing_type: str = Field(..., description='The type of filing (either "10-K" or "10-Q").')
    filing_quarter: Optional[int] = Field(
        None, description='The quarter of the filing (1, 2, 3, or 4). Defaults to None for no quarter.'
    )
    year: int = Field(..., description='The year of the filing.')


class SecEdgarFilingsInputsList(BaseModel):
    """Tool for retrieving a financial filing from SEC Edgar and then answering the original user question."""

    inputs_list: List[SecEdgarFilingsInput] = Field(..., description='The list of filing metadata.')


class SecEdgarFilingRetriever(BaseTool):  # type: ignore
    """Tool for retrieving a financial filing from SEC Edgar and then answering the original user question."""

    name: str = 'SEC Edgar Filing Retriever'
    description: str = 'Retrieve a financial filing from SEC Edgar and then answer the original user question.'
    filing_metadata: SecEdgarFilingsInput

    def _run(self) -> Path:
        # Retrieve the filing text from SEC Edgar
        try:
            downloader = Downloader(os.getenv('SEC_API_ORGANIZATION'), os.getenv('SEC_API_EMAIL'))
        except requests.exceptions.HTTPError:
            raise Exception('Please submit your SEC EDGAR details (organization and email) in the sidebar first.')

        # Extract today's year
        current_year = datetime.datetime.now().date().year

        # Extract the delta time, i.e. the number of years between the current year and the year of the filing
        delta = current_year - self.filing_metadata.year

        # Quarterly filing retrieval
        if self.filing_metadata.filing_type == '10-Q':
            if self.filing_metadata.filing_quarter is not None:
                if not isinstance(self.filing_metadata.filing_quarter, int):
                    raise TypeError('The quarter must be an integer.')
                if self.filing_metadata.filing_quarter not in [
                    1,
                    2,
                    3,
                    4,
                ]:
                    raise ValueError('The quarter must be between 1 and 4.')
                delta = (current_year - self.filing_metadata.year + 1) * 3
            else:
                raise ValueError('The quarter must be provided for 10-Q filing.')

        # Yearly filings
        elif self.filing_metadata.filing_type == '10-K':
            delta = current_year - self.filing_metadata.year + 1
        else:
            raise ValueError('The filing type must be either "10-K" or "10-Q".')

        response_dict: Dict[str, str] = dict()

        # Parse filings
        # Extract the metadata of the filings
        from sec_downloader.types import RequestedFilings

        metadata = downloader.get_filing_metadatas(
            RequestedFilings(
                ticker_or_cik=self.filing_metadata.ticker_symbol,
                form_type=self.filing_metadata.filing_type,
                limit=delta,
            )
        )[0]

        html_text = downloader.download_filing(url=metadata.primary_doc_url)

        # Convert html to text
        soup = BeautifulSoup(html_text, 'html.parser')

        # Extract text from the parsed HTML
        text = soup.get_text(separator=' ', strip=True)

        # Instantiate the text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=[
                r'\n\n',  # Split on double newlines (paragraphs)
                r'(?<=[.!?])\s+(?=[A-Z])',  # Split on sentence boundaries
                r'\n',  # Split on single newlines
                r'\s+',  # Split on whitespace
                r'',  # Split on characters as a last resort
            ],
            is_separator_regex=True,
        )

        # Split the text into chunks
        chunks = splitter.split_text(text)

        # Save chunks to csv
        df = pandas.DataFrame(chunks, columns=['text'])

        # Specify the directory and file path
        directory = Path(
            'financial_agent_crewai',
            'cache',
            'output',
            'filings',
        )

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        filename = directory / (
            f"filing_id_{self.filing_metadata.filing_type.replace('-', '')}_{self.filing_metadata.filing_quarter}_"
            + f'{self.filing_metadata.ticker_symbol}_{self.filing_metadata.year}.csv'
        )

        # Save chunks as csv
        df.to_csv(filename, index=False)

        # Return the file name
        return filename


def get_qa_response(user_request: str, documents: List[Document]) -> Any:
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
    vectorstore, retriever = get_vectorstore_retriever(documents)

    # Get the QA chain from the retriever
    qa_chain = get_qa_chain(retriever)

    # Invoke the QA chain to get an answer to the user
    response = invoke_qa_chain(qa_chain, user_request)

    return response


def invoke_qa_chain(qa_chain: RunnableBinding[Dict[str, Any], Dict[str, Any]], user_query: str) -> Dict[str, Any]:
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


def get_qa_chain(retriever: VectorStoreRetriever) -> Any:
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
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return qa_chain


class TXTSearchToolSchema(BaseModel):
    """Input for TXTSearchTool."""

    txt: str = Field(..., description='Mandatory txt path you want to search.')


class TXTSearchTool(BaseTool):  # type: ignore
    name: str = "Search a txt's content."
    description: str = "A tool that can be used to semantic search a query from a txt's content."
    txt_path: TXTSearchToolSchema

    def _run(self, search_query: str) -> Any:
        """Execute the search query and return results"""
        df = pandas.read_csv(self.txt_path.txt)
        # Convert DataFrame rows into Document objects
        documents = list()
        for _, row in df.iterrows():
            document = Document(page_content=row['text'])
            documents.append(document)

        # Get QA response
        answer = get_qa_response(user_request=search_query, documents=documents)['answer']

        return answer
