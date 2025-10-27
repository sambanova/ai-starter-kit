import os
import sys

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_sambanova import ChatSambaNova, SambaNovaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)

from typing import List, Tuple

from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings

from utils.eval.eval_utils import custom_parser
from utils.eval.schemas import EmbeddingsSchema, SNCloudSchema, VectorDBSchema
from utils.eval.vector_store import VectorStoreManager


class RAGChain:
    """
    A class representing a Retrieval-Augmented Generation (RAG) chain.

    Attributes:
        model_type (str): The type of model to use.
        model_name (str): The name of the model to use.
        temperature (float): The temperature for the model.
        max_tokens (int): The maximum number of tokens for the model.
        embeddings (object): The embeddings model.
        vectordb (object): The vector database.
        rag_chain (object): The RAG chain.
    """

    def __init__(
        self, llm_params: SNCloudSchema, embeddings_params: EmbeddingsSchema, vectordb_params: VectorDBSchema
    ) -> None:
        self.llm_params = llm_params
        self.embeddings_params = embeddings_params
        self.vectordb_params = vectordb_params
        self.embeddings, self.vectordb = self._init_vectordb()
        self.rag_chain = self._init_chain()

    def upload_docs(self, path: str) -> None:
        """
        Uploads documents to the vector database.

        Args:
            path (str): The path to the documents.

        Raises:
            FileNotFoundError: If the path does not exist.
        """

        if not os.path.exists(path):
            raise FileNotFoundError('Path does not exist')

        loader = PyPDFLoader(path)
        docs = loader.load()
        self.vectordb.add_documents(docs)

    def predict(self, query: str) -> str:
        """
        Makes a prediction using the RAG chain.

        Args:
            query (str): The query to make a prediction on.

        Returns:
            str: The prediction.

        Raises:
            ValueError: If query is empty.
        """

        return self.rag_chain.invoke(query)

    def retrieve(self, query: str) -> List[str]:
        retriever = self.vectordb.as_retriever()
        docs = retriever.invoke(query)
        return [i.page_content for i in docs]

    def _init_vectordb(self) -> Tuple[Embeddings, Chroma]:
        """
        Initializes the vector database.

        Returns:
            Tuple: A tuple containing the embeddings model and the vector database.
        """
        try:
            embeddings = SambaNovaEmbeddings(**self.embeddings_params.model_dump())
            vectordb = VectorStoreManager.load_vectordb(embeddings=embeddings, **self.vectordb_params.model_dump())
            return embeddings, vectordb
        except Exception as e:
            raise Exception(f'Failed to initialize vector database: {e}') from e

    def _init_chain(self) -> None:
        """
        Initializes the RAG chain.

        Returns:
            object: The RAG chain.
        """

        try:
            prompt = hub.pull('rlm/rag-prompt')
            retriever = self.vectordb.as_retriever()
            llm = ChatSambaNova(**self.llm_params.model_dump())

            rag_chain = prompt | llm | custom_parser

            rag_chain_with_source = RunnableParallel({'context': retriever, 'question': RunnablePassthrough()}).assign(
                response=rag_chain
            )

            return rag_chain_with_source
        except Exception as e:
            raise Exception(f'Failed to initialize RAG chain: {e}') from e
