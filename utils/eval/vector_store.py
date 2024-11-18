from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    """
    A class responsible for managing vector stores.

    Attributes:
        None

    Methods:
        load_vectordb: Loads a vector database based on the provided type and collection name.
    """

    @staticmethod
    def load_vectordb(embeddings: Embeddings, db_type: str, collection_name: str) -> Chroma:
        """
        Loads a vector database based on the provided type and collection name.

        Args:
            embeddings (Embeddings): The embeddings to use for the vector store.
            db_type (str): The type of vector store to load. Currently, only 'chroma' is supported.
            collection_name (str): The name of the collection to load.

        Returns:
            Chroma: The loaded vector store.

        Raises:
            ValueError: If the provided db_type is not 'chroma'.
        """
        if db_type == 'chroma':
            vectordb = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory='./chroma_langchain_db',
            )
            return vectordb
        else:
            raise ValueError(f'{db_type} is not a valid vector store type')
