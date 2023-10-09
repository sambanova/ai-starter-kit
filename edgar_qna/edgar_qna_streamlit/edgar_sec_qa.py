import sys

sys.path.append("../../")
import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate

# import langchain
# langchain.debug = True
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
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
        if persist_directory and os.path.exists(persist_directory) and not force_reload:
            ## Load from the persist db
            self.vectordb = Chroma(
                persist_directory=persist_directory, embedding_function=self.embedding
            )
        else:
            # 1. Get SEC data
            last_n = 1
            try:
                loader = SECFilingsLoader(
                    tickers=[ticker], amount=last_n, filing_type="10-K"
                )
                loader.load_data()
            except Exception as ex:
                raise Exception(
                    f"Failed to fetch data for {ticker} from Edgar database"
                ) from ex
            sec_dir = f"data/{ticker}"
            dir_loader = DirectoryLoader(
                sec_dir, glob="**/*.json", loader_cls=TextLoader
            )
            documents = dir_loader.load()
            ## 2. Split the texts
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)

            ## 3. Create Embeddings and add to chroma store
            ##TODO: Validate if self.embedding is not None
            self.vectordb = Chroma.from_documents(
                documents=texts,
                embedding=self.embedding,
                persist_directory=persist_directory,
            )

    def retreival_qa_chain(self):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 2})
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
