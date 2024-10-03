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
from typing import Any

from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Qdrant
from langchain_chroma import Chroma

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG)
    format='%(asctime)s [%(levelname)s] - %(message)s',  # Define the log message format
    handlers=[
        logging.StreamHandler(),  # Output logs to the console
        logging.FileHandler('create_vector_db.log'),
    ],
)

# Create a logger object
logger = logging.getLogger(__name__)


# Parse the arguments
def parse_arguments() -> Any:
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-input_path', type=dir_path, help='path to input directory')
    parser.add_argument('--chunk_size', type=int, help='chunk size for splitting')
    parser.add_argument('--chunk_overlap', type=int, help='chunk overlap for splitting')
    parser.add_argument('-output_path', type=dir_path, help='path to input directory')

    return parser.parse_args()


# Check valid path
def dir_path(path: str) -> Any:
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')


def main(input_path: str, output_db: str, chunk_size: int, chunk_overlap: int, db_type: str) -> Any:
    # Load files from input_location
    loader = DirectoryLoader(input_path, glob='*.txt')
    docs = loader.load()
    logger.info(f'Total {len(docs)} files loaded')

    # get the text chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f'Total {len(chunks)} chunks created')

    # create vector store
    encode_kwargs = {'normalize_embeddings': True}
    embedding_model = 'BAAI/bge-large-en'
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model,
        embed_instruction='',  # no instruction is needed for candidate passages
        query_instruction='Represent this sentence for searching relevant passages: ',
        encode_kwargs=encode_kwargs,
    )
    logger.info(
        f'Processing embeddings using {embedding_model}. This could take time depending on the number of chunks ...'
    )

    if db_type == 'faiss':
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        # save vectorstore
        vectorstore.save_local(output_db)
    elif db_type == 'chromadb':
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=output_db)
    elif db_type == 'qdrant':
        vectorstore = Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings,
            path=output_db,
            collection_name='test_collection',
        )
    elif db_type == 'qdrant-server':
        url = 'http://localhost:6333/'
        vectorstore = Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=url,
            prefer_grpc=True,
            collection_name='anaconda',
        )

    logger.info(f'Vector store saved to {output_db}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data with optional chunking')

    # Required arguments
    parser.add_argument('input_path', type=str, help='Path to the input directory')
    parser.add_argument('output_db', type=str, help='Path to the output vectordb')

    # Optional arguments
    parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size (default: 1000)')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Chunk overlap (default: 200)')
    parser.add_argument(
        '--db_type',
        type=str,
        default='faiss',
        help='Type of vectorstore (default: faiss)',
    )

    args = parser.parse_args()
    main(
        args.input_path,
        args.output_db,
        args.chunk_size,
        args.chunk_overlap,
        args.db_type,
    )
