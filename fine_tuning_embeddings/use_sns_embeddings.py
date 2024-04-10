
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)



import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from utils.sambanova_endpoint import SambaNovaEndpoint, SambaverseEndpoint, SambaNovaEmbeddingModel
import yaml

# Use embeddings As Part of Langchain
from langchain.vectorstores import Chroma

CONFIG_PATH = os.path.join(current_dir,'config.yaml')

with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
api_info = config["api"]
llm_info = config["llm"]


# load dot_env
load_dotenv(os.path.join(current_dir,".env"))

try:
    from snsdk import SnSdk
except ImportError:
    snsdk_installed = False


def main():
    # snsdk_model retuns a langchain Embedding Object which can be used within langchain
    snsdk_model = SambaNovaEmbeddingModel()

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


    if api_info == "sambaverse":
        llm = SambaverseEndpoint(
            sambaverse_model_name=llm_info["sambaverse_model_name"],
            sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
            model_kwargs={
                "do_sample": False, 
                "max_tokens_to_generate": llm_info["max_tokens_to_generate"],
                "temperature": llm_info["temperature"],
                "process_prompt": True,
                "select_expert": llm_info["smabaverse_select_expert"]
                #"stop_sequences": { "type":"str", "value":""},
                # "repetition_penalty": {"type": "float", "value": "1"},
                # "top_k": {"type": "int", "value": "50"},
                # "top_p": {"type": "float", "value": "1"}
            }
        )
    
    elif api_info == "sambastudio":
        llm = SambaNovaEndpoint(
            model_kwargs={
                "do_sample": True, 
                "temperature": llm_info["temperature"],
                "max_tokens_to_generate": llm_info["max_tokens_to_generate"],
                            #"stop_sequences": { "type":"str", "value":""},
            # "repetition_penalty": {"type": "float", "value": "1"},
            # "top_k": {"type": "int", "value": "50"},
            # "top_p": {"type": "float", "value": "1"}
            }
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
