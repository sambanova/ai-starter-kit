import pandas as pd
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain.vectorstores import Qdrant

SNP500_CSV_FILE = "ticker_to_download.csv"


URL = "127.0.0.1:6333"
DATA_DIR = "data"


def update_meta_data(documents):
    docs = []
    for doc in documents:
        metadata = doc.metadata
        ticker, year, form = metadata["source"].split("/")[-3:]
        metadata["ticker"] = ticker
        metadata["year"] = year
        metadata["form"] = form
        # skip 10-q for now
        if form.startswith("10-Q"):
            continue
        docs.append(doc)
    return docs


def main():
    df = pd.read_csv(SNP500_CSV_FILE)
    embedding = HuggingFaceInstructEmbeddings(
        query_instruction="Represent the query for retrieval: "
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    for i, ticker in enumerate(sorted(df.Symbol)):
        # for i, ticker in enumerate(tickers):
        #     # if i
        #     if i <= 3:
        #         continue
        try:
            ticker = ticker.lower()
            sec_dir = f"{DATA_DIR}/{ticker}"
            dir_loader = DirectoryLoader(
                sec_dir, glob="**/*.json", loader_cls=TextLoader
            )
            documents = dir_loader.load()
            documents = update_meta_data(documents)
            documents = text_splitter.split_documents(documents)
            vectordb = Qdrant.from_documents(
                documents,
                embedding,
                url=URL,
                path=None,
                api_key=None,
                collection_name="edgar",
                force_recreate=False,
            )
        except Exception as ex:
            print(ticker, ex)
            # i+=1
        # i+=1
        print(i, ticker)
        if i > 10:
            return


if __name__ == "__main__":
    main()
