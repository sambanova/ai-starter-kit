import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
import sys
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# Use embeddings As Part of Langchain
from langchain.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from src.models.sambanova_endpoint import SambaNovaEndpoint

# load dot_env

load_dotenv(os.path.join(repo_dir,".env"))

try:
    from snsdk import SnSdk
except ImportError:
    snsdk_installed = False


class SNSDKModel(Embeddings):
    def __init__(self, url_domain, project_id, endpoint_id, key):
        api_key_path = os.path.join(os.environ["HOME"], ".snapi/secret.txt")
        with open(api_key_path) as file:
            api_key = file.read().strip()
        self.sdk = SnSdk(host_url=url_domain, access_key=api_key)
        self.sdk.http_session.verify = False
        self.project_id = project_id
        self.endpoint_id = endpoint_id
        self.key = key

    def embed_documents(self, texts, batch_size=32, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        embeddings = []
        for sentence in texts:
            sentence = "document: " + sentence
            responses = self.sdk.nlp_predict(
                self.project_id, self.endpoint_id, self.key, sentence
            )
            embedding = responses["data"][0]
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text, batch_size=32, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        sentence = "query: " + text
        responses = self.sdk.nlp_predict(
            self.project_id, self.endpoint_id, self.key, sentence
        )
        embedding = responses["data"][0]
        return embedding


def main():
    # snsdk_model retuns a langchain Embedding Object which can be used within langchain
    snsdk_model = SNSDKModel(
        url_domain=os.environ.get("EMBED_BASE_URL"),
        project_id=os.environ.get("EMBED_PROJECT_ID"),
        endpoint_id=os.environ.get("EMBED_ENDPOINT_ID"),
        key=os.environ.get("EMBED_API_KEY"),
    )

    # An Example Using Raw Text and Cosine Similarity
    documents = [
        "25 backpacks to take to work or school in 2023",
        "How an 11th-century monastery reclaimed artifacts from the US — and discovered a hoard of treasures in the process",
        "Retail sales rose 0.7% in September, much stronger than estimate",
        "80% of companies plan to adopt AI in the next year. Here’s how it's already helping M&A",
        "AI startup SambaNova Systems reveals new SN40 chip",
        "U.S. wraps up fiscal year with a budget deficit near $1.7 trillion, up 23%",
        "U.S. soccer team leads in the World Cup",
    ]

    query = "What is the current state of the United States budget?"

    document_encodings = np.asarray(snsdk_model.embed_documents(documents))
    query_encoding = snsdk_model.embed_query(query)
    similarity_doc = cosine_similarity(
        document_encodings, np.asarray(query_encoding).reshape(1, -1)
    )

    print(f"Document encodings are {document_encodings}")
    print(f"================================")
    print(f"================================")
    print(f"Similarity doc is {similarity_doc}")

    # A Small Example to see how SNS embeddings can be integrated within Langchain Workflow
    # Let's set embeddings to equal or snsdk_model to keep with the langchain convention
    embeddings = snsdk_model

    llm = SambaNovaEndpoint(
        model_kwargs={
            "do_sample": True,
            "temperature": 0.01,
            "max_tokens_to_generate": 512,
        },
    )

    loader = WebBaseLoader("https://docs.smith.langchain.com")

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = Chroma.from_documents(documents, embeddings)

    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
    print(response["answer"])


if __name__ == "__main__":
    main()
