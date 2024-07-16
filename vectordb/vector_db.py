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

from langchain_community.document_loaders import DirectoryLoader, UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import SambaStudioEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma, Qdrant

EMBEDDING_MODEL = "intfloat/e5-large-v2"
NORMALIZE_EMBEDDINGS = True
VECTORDB_LOG_FILE_NAME = "vector_db.log"

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG)
    format="%(asctime)s [%(levelname)s] - %(message)s",  # Define the log message format
    handlers=[
        logging.StreamHandler(),  # Output logs to the console
        logging.FileHandler(VECTORDB_LOG_FILE_NAME),
    ],
)

# Create a logger object
logger = logging.getLogger(__name__)


class VectorDb():
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
        load_embedding_model: Load a sambastudio or a huggingface embedding model
        create_vector_store: Create a vector store from chunks and an embedding model
        load_vdb: load a previous stored vector database 
        update_vdb: Update an existing vector store with new chunks
        create_vdb: Create a vector database from the raw files in a specific input directory 
    """
    def __init__(self) -> None:
        pass

    def load_files(self, input_path, recursive=False, load_txt=True, load_pdf=False, urls = None) -> list:
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
        docs=[]
        text_loader_kwargs={'autodetect_encoding': True}
        if input_path is not None:
            if load_txt:
                loader = DirectoryLoader(input_path, glob="*.txt", recursive=recursive, show_progress=True, loader_kwargs=text_loader_kwargs)
                docs.extend(loader.load())
            if load_pdf:
                loader = DirectoryLoader(input_path, glob="*.pdf", recursive=recursive, show_progress=True, loader_kwargs=text_loader_kwargs)
                docs.extend(loader.load())
        if urls:
            loader = UnstructuredURLLoader(urls=urls)
            docs.extend(loader.load())
        
        logger.info(f"Total {len(docs)} files loaded")

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
            logger.info(f"Splitter: splitting documents")
            chunks = text_splitter.split_documents(docs)
        else:
            logger.info(f"Splitter: creating documents with metadata")
            chunks = text_splitter.create_documents(docs, meta_data)

        logger.info(f"Total {len(chunks)} chunks created")

        return chunks

    def get_token_chunks(self, docs: list, chunk_size: int, chunk_overlap: int, tokenizer) -> list:
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

        logger.info(f"Splitter: splitting documents")
        chunks = text_splitter.split_documents(docs)

        logger.info(f"Total {len(chunks)} chunks created")

        return chunks

    def load_embedding_model(self, type = "cpu", batch_size = None ,coe = False, select_expert = None):
        """Loads embedding model
        Args:
            type (str): wether to use sambastudio embedding model or in local cpu model
        Returns:
            langchain embedding model
        """
        
        if type == "sambastudio":
            if coe:
                if batch_size is None:
                    batch_size = 1
                embeddings = SambaStudioEmbeddings(
                    batch_size=batch_size,
                    model_kwargs = {
                        "select_expert":select_expert
                        }
                    )
            else:
                if batch_size is None:
                    batch_size = 32
                embeddings = SambaStudioEmbeddings(
                    batch_size=batch_size
                )
        elif type == "cpu":
            encode_kwargs = {"normalize_embeddings": NORMALIZE_EMBEDDINGS}
            embedding_model = EMBEDDING_MODEL
            embeddings = HuggingFaceInstructEmbeddings(
                model_name=embedding_model,
                embed_instruction="",  # no instruction is needed for candidate passages
                query_instruction="Represent this sentence for searching relevant passages: ",
                encode_kwargs=encode_kwargs,
            )
        else:
            raise ValueError(f"{type} is not a valid embedding model type")

        return embeddings

    def create_vector_store(self, chunks: list, embeddings: HuggingFaceInstructEmbeddings, db_type: str,
                            output_db: str = None):
        """Creates a vector store

        Args:
            chunks (list): list of chunks
            embeddings (HuggingFaceInstructEmbeddings): embedding model
            db_type (str): vector db type
            output_db (str, optional): output path to save the vector db. Defaults to None.
        """

        if db_type == "faiss":
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            if output_db:
                vector_store.save_local(output_db)

        elif db_type == "chroma":
            if output_db:
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=output_db
                )
            else:
                try:
                    vector_store = Chroma()
                    vector_store.delete_collection()
                    print("colelction deleted")
                except Exception as e:
                    print("colelction not deleted")
                    pass
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

        elif db_type == "qdrant":
            if output_db:
                vector_store = Qdrant.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    path=output_db,
                    collection_name="test_collection",
                )
            else:
                vector_store = Qdrant.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name="test_collection",
                )

        logger.info(f"Vector store saved to {output_db}")

        return vector_store

    def load_vdb(self, persist_directory, embedding_model, db_type="chroma"):

        if db_type == "faiss":
            vector_store = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
        elif db_type == "chroma":
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        elif db_type == "qdrant":
            # TODO: vector_store = Qdrant...
            pass

        return vector_store

    def update_vdb(self, chunks: list, embeddings, db_type: str, input_db: str = None,
                   output_db: str = None):

        if db_type == "faiss":
            vector_store = FAISS.load_local(input_db, embeddings, allow_dangerous_deserialization=True)
            new_vector_store = self.create_vector_store(chunks, embeddings, db_type, None)
            vector_store.merge_from(new_vector_store)
            if output_db:
                vector_store.save_local(output_db)

        elif db_type == "chroma":
            # TODO implement update method for chroma
            pass
        elif db_type == "qdrant":
            # TODO implement update method for qdrant
            pass

        return vector_store

    def create_vdb(self, input_path, chunk_size, chunk_overlap, db_type, output_db=None, recursive=False, tokenizer=None, load_txt=True, load_pdf=False, urls=None, embedding_type="cpu"):

        docs = self.load_files(input_path, recursive=recursive, load_txt=load_txt, load_pdf=load_pdf, urls=urls)

        if tokenizer is None:
            chunks = self.get_text_chunks(docs, chunk_size, chunk_overlap)
        else:
            chunks = self.get_token_chunks(docs, chunk_size, chunk_overlap, tokenizer)

        embeddings = self.load_embedding_model(type=embedding_type)

        vector_store = self.create_vector_store(chunks, embeddings, db_type, output_db)

        return vector_store


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


# Parse the arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument("-input_path", type=dir_path, help="path to input directory")
    parser.add_argument("--chunk_size", type=int, help="chunk size for splitting")
    parser.add_argument("--chunk_overlap", type=int, help="chunk overlap for splitting")
    parser.add_argument("-output_path", type=dir_path, help="path to input directory")

    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data with optional chunking")

    # Required arguments
    parser.add_argument("--input_path", type=str, help="Path to the input directory")
    parser.add_argument("--output_db", type=str, help="Path to the output vectordb")

    # Optional arguments
    parser.add_argument(
        "--chunk_size", type=int, default=1000, help="Chunk size (default: 1000)"
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=200, help="Chunk overlap (default: 200)"
    )
    parser.add_argument(
        "--db_type",
        type=str,
        default="faiss",
        help="Type of vector store (default: faiss)",
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
