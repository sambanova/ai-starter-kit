import sys

sys.path.append("../../")
import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# import langchain
# langchain.debug = True
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from sec_edgar_downloader import Downloader

from xbrl import XBRLParser
from dotenv import load_dotenv
load_dotenv('../export.env')

from src.models.sambanova_endpoint import SambaNovaEndpoint

EMBEDDING_MODEL = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: "
)
EMAIL = 'mlengineer@samba.ai'
COMPANY = 'samba'
DATA_DIRECTORY = './data'
PERSIST_TSLA = 'chroma_db/tsla'

class SecFilingQa:
    def __init__(self, config: dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None

    def init_embeddings(self) -> None:
        # print("initializng embeddings")
        self.embedding = EMBEDDING_MODEL

    def init_models(self) -> None:
        # print("initializng models")
        self.llm = SambaNovaEndpoint(
            model_kwargs={"do_sample": True, "temperature": 0.1}
        )

    def vector_db_sec_docs(self, force_reload: bool = False) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        # print("creating/loading vector db")
        persist_directory = self.config.get("persist_directory", None)
        ticker = self.config.get("ticker", None)

        if persist_directory and os.path.exists(persist_directory) and not force_reload:
            # Load from the persist db
            self.vectordb = Chroma(
                persist_directory=persist_directory, embedding_function=self.embedding
            )
        else:
            # 1. Get SEC data
            last_n = 1
            try:
                dl = Downloader(COMPANY, EMAIL, DATA_DIRECTORY)
                dl.get("10-K", ticker, limit=last_n)
            except Exception as ex:
                raise Exception(
                    f"Failed to fetch data for {ticker} from Edgar database"
                ) from ex
            sec_dir = f"{DATA_DIRECTORY}/sec-edgar-filings/{ticker}"
            dir_loader = DirectoryLoader(
                sec_dir, glob="**/*.txt", loader_cls=TextLoader
            )
            documents = dir_loader.load()
            
            # Parse XBRL document and get text
            xbrl_parser = XBRLParser()
            xbrl_texts = []
            for document in documents:
                xbrl_doc = xbrl_parser.parse(document.metadata['source'])
                p_span_tags = xbrl_doc.find_all(lambda x: x.name == 'p' and x.find('span'))
                xbrl_text = ' '.join(tag.get_text() for tag in p_span_tags)  
                xbrl_texts.append(xbrl_text)
            
            all_document_texts = '\n\n'.join(xbrl_texts)
            complete_document =  Document(page_content=all_document_texts, metadata={})
        
            # 2. Split the texts
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200, chunk_overlap=240
            )
            text_chunks = text_splitter.split_documents([complete_document])

            # 3. Create Embeddings and add to chroma store
            #TODO: Validate if self.embedding is not None
            self.vectordb = Chroma.from_documents(
                documents=text_chunks,
                embedding=self.embedding,
                persist_directory=persist_directory,
            )

    def retreival_qa_chain(self):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        # print("defining retriever")
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})
        # docs = self.retriever.get_relevant_documents("What are the biggest discussed risks?")
        # print(len(docs))
        # print(docs)
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            input_key="question",
            output_key="response",
            return_source_documents=True,
            # verbose=True,
        )

        custom_prompt_template = """
        <s>[INST] <<SYS>>\nYou're an expert in filing reports of many companies\n<</SYS>>\n 
        Given the following context enclosed in backticks regarding a company annual/quarterly report filing:
        ```
        {context}
        ```
        Consider the question:  
        {question}
        Answer the question using only the information from the context. If the answer to the question can't be extracted from the preovious context, then say "I do not have information regarding this".
        Helpful Answer: [/INST]"""
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

if __name__ == '__main__':
    config = {'persist_directory': PERSIST_TSLA, 'ticker': 'tsla'}
    sec_qa = SecFilingQa(config)
    sec_qa.init_embeddings()
    sec_qa.init_models()
    sec_qa.vector_db_sec_docs()