import sys

sys.path.append("../")
import os

import qdrant_client
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate

# import langchain
# langchain.debug = True
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain.vectorstores import Qdrant
from llama_hub.sec_filings.base import SECFilingsLoader

from src.models.sambanova_endpoint import SambaNovaEndpoint

EMBEDDING_MODEL = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: "
)


class SecFilingQa:
    def __init__(self, config: dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None

    def init_embeddings(self) -> None:
        self.embedding = EMBEDDING_MODEL

    def init_models(self) -> None:
        self.llm = SambaNovaEndpoint(
            model_kwargs={"do_sample": True, "temperature": 0.1}
        )

    def vector_db_sec_docs(self, force_reload: bool = False) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        persist_directory = self.config.get("persist_directory", None)
        ticker = self.config.get("ticker", None)
        vector_db_url = self.config.get("vector_db_url", None)
        client = qdrant_client.QdrantClient(path=None, url=vector_db_url)
        self.vectordb = Qdrant(
            client=client,
            collection_name="edgar",
            embeddings=self.embedding,
        )

    def retreival_qa_chain(self, ticker: str):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        self.retriever = self.vectordb.as_retriever(
            search_kwargs={"k": 2, "filter": {"ticker": ticker}}
        )
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            input_key="question",
            output_key="response",
            return_source_documents=True,
        )
        custom_prompt_template = """Use the following pieces of context about company annual/quarterly report filing to answer the question at the end. If the answer to the question cant be extracted from given CONTEXT than  say I do not have information regarding this.
        {context}

        Question: {question}
        Helpful Answer:"""
        CUSTOMPROMPT = PromptTemplate(
            template=custom_prompt_template, input_variables=["context", "question"]
        )
        ## Inject custom prompt
        self.qa.combine_documents_chain.llm_chain.prompt = CUSTOMPROMPT

    def answer_sec(self, question: str) -> str:
        """
        Answer the question
        """
        resp = self.qa(question)
        ans = resp["response"]
        resp = f"{ans}\n"
        # https://github.com/streamlit/streamlit/issues/868#issuecomment-965819742
        resp = resp.replace("\n", "  \n")
        return resp
