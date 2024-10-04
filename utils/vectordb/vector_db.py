# Define the script's usage example
USAGE_EXAMPLE = """
Example usage:

To process input *.txt files at input_path and save the vector db output at output_db:
python create_vector_db.py input_path output_db --chunk_size 100 --chunk_overlap 10

Required arguments:
- input_path: Path to the input dir containing the .txt files
- output_path: Path to the output vector db.

Optional arguments:
- --chunk_size: Size of the chunks (default: None).
- --chunk_overlap: Overlap between chunks (default: None).
"""

import argparse
import logging
import os
import sys
from typing import Any, List, Optional

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredURLLoader
from langchain_community.vectorstores import FAISS, Chroma, Qdrant

vectordb_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(vectordb_dir, '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))

sys.path.append(repo_dir)
sys.path.append(utils_dir)

import uuid

from utils.model_wrappers.api_gateway import APIGateway

EMBEDDING_MODEL = 'intfloat/e5-large-v2'
NORMALIZE_EMBEDDINGS = True
VECTORDB_LOG_FILE_NAME = 'vector_db.log'

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG)
    format='%(asctime)s [%(levelname)s] - %(message)s',  # Define the log message format
    handlers=[
        logging.StreamHandler(),  # Output logs to the console
        logging.FileHandler(VECTORDB_LOG_FILE_NAME),
    ],
)

# Create a logger object
logger = logging.getLogger(__name__)


class VectorDb:
    """
    A class for creating, updating and loading FAISS or Chroma vector databases,
    to use them with retrieval augmented generation tasks with langchain

    Args:
        None

    Attributes:
        None

    Methods:
        load_files: Load files from an input directory as langchain documents
        get_text_chunks: Get text chunks from a list of documents
        get_token_chunks: Get token chunks from a list of documents
        create_vector_store: Create a vector store from chunks and an embedding model
        load_vdb: load a previous stored vector database
        update_vdb: Update an existing vector store with new chunks
        create_vdb: Create a vector database from the raw files in a specific input directory
    """

    def __init__(self) -> None:
        self.collection_id = str(uuid.uuid4())
        self.vector_collections = set()

    def load_files(
        self,
        input_path: str,
        recursive: bool = False,
        load_txt: bool = True,
        load_pdf: bool = False,
        urls: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Load files from input location

        Args:
            input_path : input location of files
            recursive (bool, optional): flag to load files recursively. Defaults to False.
            load_txt (bool, optional): flag to load txt files. Defaults to True.
            load_pdf (bool, optional): flag to load pdf files. Defaults to False.
            urls (list, optional): list of urls to load. Defaults to None.

        Returns:
            list: list of documents
        """
        docs = []
        text_loader_kwargs = {'autodetect_encoding': True}
        if input_path is not None:
            if load_txt:
                loader = DirectoryLoader(
                    input_path, glob='*.txt', recursive=recursive, show_progress=True, loader_kwargs=text_loader_kwargs
                )
                docs.extend(loader.load())
            if load_pdf:
                loader = DirectoryLoader(
                    input_path, glob='*.pdf', recursive=recursive, show_progress=True, loader_kwargs=text_loader_kwargs
                )
                docs.extend(loader.load())
        if urls:
            loader = UnstructuredURLLoader(urls=urls)
            docs.extend(loader.load())

        logger.info(f'Total {len(docs)} files loaded')

        return docs

    def get_text_chunks(self, docs: list, chunk_size: int, chunk_overlap: int, meta_data: list = None) -> list:
        """Gets text chunks. If metadata is not None, it will create chunks with metadata elements.

        Args:
            docs (list): list of documents or texts. If no metadata is passed, this parameter is a list of documents.
            If metadata is passed, this parameter is a list of texts.
            chunk_size (int): chunk size in number of characters
            chunk_overlap (int): chunk overlap in number of characters
            metadata (list, optional): list of metadata in dictionary format. Defaults to None.

        Returns:
            list: list of documents
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )

        if meta_data is None:
            logger.info(f'Splitter: splitting documents')
            chunks = text_splitter.split_documents(docs)
        else:
            logger.info(f'Splitter: creating documents with metadata')
            chunks = text_splitter.create_documents(docs, meta_data)

        logger.info(f'Total {len(chunks)} chunks created')

        return chunks

    def get_token_chunks(self, docs: List[Any], chunk_size: int, chunk_overlap: int, tokenizer: Any) -> List[Any]:
        """Gets token chunks. If metadata is not None, it will create chunks with metadata elements.

        Args:
            docs (list): list of documents or texts. If no metadata is passed, this parameter is a list of documents.
            If metadata is passed, this parameter is a list of texts.
            chunk_size (int): chunk size in number of tokens
            chunk_overlap (int): chunk overlap in number of tokens

        Returns:
            list: list of documents
        """

        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        logger.info(f'Splitter: splitting documents')
        chunks = text_splitter.split_documents(docs)

        logger.info(f'Total {len(chunks)} chunks created')

        return chunks

    def create_vector_store(
        self,
        chunks: list,
        embeddings: Any,
        db_type: str,
        output_db: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Any:
        """Creates a vector store

        Args:
            chunks (list): list of chunks
            embeddings (HuggingFaceInstructEmbeddings): embedding model
            db_type (str): vector db type
            output_db (str, optional): output path to save the vector db. Defaults to None.
        """
        if collection_name is None:
            collection_name = f'collection_{self.collection_id}'
            logger.info(f'This is the collection name: {collection_name}')

        if db_type == 'faiss':
            vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
            if output_db:
                vector_store.save_local(output_db)

        elif db_type == 'chroma':
            if output_db:
                vector_store = Chroma()
                vector_store.delete_collection()
                vector_store = Chroma.from_documents(
                    documents=chunks, embedding=embeddings, persist_directory=output_db, collection_name=collection_name
                )
            else:
                vector_store = Chroma()
                vector_store.delete_collection()
                vector_store = Chroma.from_documents(
                    documents=chunks, embedding=embeddings, collection_name=collection_name
                )
            self.vector_collections.add(collection_name)

        elif db_type == 'qdrant':
            if output_db:
                vector_store = Qdrant.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    path=output_db,
                    collection_name='test_collection',
                )
            else:
                vector_store = Qdrant.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name='test_collection',
                )

        logger.info(f'Vector store saved to {output_db}')

        return vector_store

    def load_vdb(
        self,
        persist_directory: Optional[str],
        embedding_model: Any,
        db_type: str = 'chroma',
        collection_name: Optional[str] = None,
    ) -> Any:
        if db_type == 'faiss':
            vector_store = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
        elif db_type == 'chroma':
            if collection_name:
                vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding_model,
                    collection_name=collection_name,
                )
            else:
                vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        elif db_type == 'qdrant':
            # TODO: Implement Qdrant loading
            pass
        else:
            raise ValueError(f'Unsupported database type: {db_type}')

        return vector_store

    def update_vdb(
        self,
        chunks: List[Any],
        embeddings: Any,
        db_type: str,
        input_db: Optional[str] = None,
        output_db: Optional[str] = None,
    ) -> None:
        if db_type == 'faiss':
            vector_store = FAISS.load_local(input_db, embeddings, allow_dangerous_deserialization=True)
            new_vector_store = self.create_vector_store(chunks, embeddings, db_type, None)
            vector_store.merge_from(new_vector_store)
            if output_db:
                vector_store.save_local(output_db)

        elif db_type == 'chroma':
            # TODO implement update method for chroma
            pass
        elif db_type == 'qdrant':
            # TODO implement update method for qdrant
            pass

        return vector_store

    def create_vdb(
        self,
        input_path: str,
        chunk_size: int,
        chunk_overlap: int,
        db_type: str,
        output_db: Optional[str] = None,
        recursive: Optional[bool] = False,
        tokenizer: Optional[Any] = None,
        load_txt: bool = True,
        load_pdf: bool = False,
        urls: Optional[List[str]] = None,
        embedding_type: str = 'cpu',
        batch_size: Optional[int] = None,
        coe: Optional[bool] = None,
        select_expert: Optional[str] = None,
    ) -> Any:
        docs = self.load_files(input_path, recursive=recursive, load_txt=load_txt, load_pdf=load_pdf, urls=urls)

        if tokenizer is None:
            chunks = self.get_text_chunks(docs, chunk_size, chunk_overlap)
        else:
            chunks = self.get_token_chunks(docs, chunk_size, chunk_overlap, tokenizer)

        embeddings = APIGateway.load_embedding_model(
            type=embedding_type, batch_size=batch_size, coe=coe, select_expert=select_expert
        )

        vector_store = self.create_vector_store(chunks, embeddings, db_type, output_db)

        return vector_store


def dir_path(path: str) -> Any:
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')


# Parse the arguments
def parse_arguments() -> Any:
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-input_path', type=dir_path, help='path to input directory')
    parser.add_argument('--chunk_size', type=int, help='chunk size for splitting')
    parser.add_argument('--chunk_overlap', type=int, help='chunk overlap for splitting')
    parser.add_argument('-output_path', type=dir_path, help='path to input directory')

    return parser.parse_args()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data with optional chunking')

    # Required arguments
    parser.add_argument('--input_path', type=str, help='Path to the input directory')
    parser.add_argument('--output_db', type=str, help='Path to the output vectordb')

    # Optional arguments
    parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size (default: 1000)')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Chunk overlap (default: 200)')
    parser.add_argument(
        '--db_type',
        type=str,
        default='faiss',
        help='Type of vector store (default: faiss)',
    )
    args = parser.parse_args()

    vectordb = VectorDb()

    vectordb.create_vdb(
        args.input_path,
        args.output_db,
        args.chunk_size,
        args.chunk_overlap,
        args.db_type,
    )
