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

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma, Qdrant

EMBEDDING_MODEL = "hkunlp/instructor-large"
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
    
    def __init__(self) -> None:
        pass
    
    def load_files(self, input_path: str) -> list:
        """Load files from input location

        Args:
            input_path (str): input location of files

        Returns:
            list: list of documents
        """
        
        loader = DirectoryLoader(input_path, glob="*.txt")
        docs = loader.load()
        logger.info(f"Total {len(docs)} files loaded")
        
        return docs
    
    def get_text_chunks(self, docs: list, chunk_size: int, chunk_overlap: int, meta_data: list = None) -> list:
        """Gets text chunks. If metadata is not None, it will create chunks with metadata elements.

        Args:
            docs (list): list of documents or texts. If no metadata is passed, this parameter is a list of documents.
            If metadata is passed, this parameter is a list of texts.
            chunk_size (int): chunk size in number of tokens
            chunk_overlap (int): chunk overlap in number of tokens
            metadata (list, optional): list of metadata in dictionary format. Defaults to None.

        Returns:
            list: list of documents 
        """
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        
        if meta_data is None:
            chunks = text_splitter.split_documents(docs)
        else:
            print("splitter: create_documents")
            chunks = text_splitter.create_documents(docs, meta_data)
            
        logger.info(f"Total {len(chunks)} chunks created")
        
        return chunks
    
    def load_embedding_model(self) -> HuggingFaceInstructEmbeddings:
        """Loads embedding model

        Returns:
            HuggingFaceInstructEmbeddings: a type of HF instruct embedding model
        """
        encode_kwargs = {"normalize_embeddings": NORMALIZE_EMBEDDINGS}
        embedding_model = EMBEDDING_MODEL
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=embedding_model,
            embed_instruction="",  # no instruction is needed for candidate passages
            query_instruction="Represent this sentence for searching relevant passages: ",
            encode_kwargs=encode_kwargs,
        )
        
        logger.info(
            f"Processing embeddings using {embedding_model}. This could take time depending on the number of chunks ..."
        )
        
        return embeddings
    
    def create_vector_store(self, chunks: list, embeddings: HuggingFaceInstructEmbeddings, db_type: str, output_db: str = None ):
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
        
    def load_vdb(self, persist_directory, embedding_model, db_type = "chroma"):

        if db_type == "faiss":
             vector_store = FAISS.load_local(persist_directory, embedding_model)
        elif db_type == "chroma":
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        elif db_type == "qdrant":
            # TODO: vector_store = Qdrant...
            pass
        
        return vector_store
    
    def update_vdb(self, chunks: list, embeddings: HuggingFaceInstructEmbeddings, db_type: str, input_db: str = None, output_db: str = None):
        
        embeddings = self.load_embedding_model()
        
        if db_type == "faiss":
             vector_store = FAISS.load_local(input_db, embeddings)
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
    
    def create_vdb(self, input_path, chunk_size, chunk_overlap, db_type, output_db=None):
        
        docs = self.load_files(input_path)

        chunks = self.get_text_chunks(docs, chunk_size, chunk_overlap)

        embeddings = self.load_embedding_model()

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
