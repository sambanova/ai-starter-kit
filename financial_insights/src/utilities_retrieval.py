import logging
import os
import sys
from typing import Any, List

import streamlit
from langchain.chains import RetrievalQA
from langchain.prompts import load_prompt
from langchain.schema import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from financial_insights.streamlit.utilities_app import _get_config_info

logging.basicConfig(level=logging.INFO)
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)


TEMP_DIR = 'financial_insights/streamlit/cache/'
SOURCE_DIR = 'financial_insights/streamlit/cache/sources/'
CONFIG_PATH = 'financial_insights/config.yaml'


def get_qa_response(
    documents: List[Document],
    user_request: str,
) -> Any:
    # Set up the embedding model and vector store
    embedding_model = SentenceTransformerEmbeddings(model_name='paraphrase-mpnet-base-v2')
    vectorstore = Chroma.from_documents(documents, embedding_model)

    # Load config
    config = _get_config_info(CONFIG_PATH)

    # Load retrieval prompt
    prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama30b-web_crawling_data_retriever.yaml'))
    retriever = vectorstore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            'score_threshold': config['tools']['rag']['retrieval']['score_threshold'],  # type: ignore
            'k': config['tools']['rag']['retrieval']['k_retrieved_documents'],  # type: ignore
        },
    )

    # Create a retrieval-based QA chain
    qa_chain = RetrievalQA.from_llm(
        llm=streamlit.session_state.fc.llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        verbose=True,
        input_key='question',
        output_key='answer',
        prompt=prompt,
    )

    # Function to answer questions based on the news data
    response = qa_chain.invoke({'question': user_request})

    return response
