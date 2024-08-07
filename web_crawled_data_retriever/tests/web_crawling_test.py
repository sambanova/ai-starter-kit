import os
import sys
import yaml
import logging

current_dir = os.getcwd()
kit_dir = current_dir
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))
print(kit_dir)
print(repo_dir)

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import streamlit as st
from src.web_crawling_retriever import WebCrawlingRetrieval
from dotenv import load_dotenv

load_dotenv(os.path.join(repo_dir,'.env'))
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

def get_config_info():
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        web_crawling_params = config["web_crawling"]
        
        return  web_crawling_params

def set_retrieval_qa_chain(documents=None, config=None, save=False):
    if config is None:
        config = {}
    if documents is None:
        documents = []
    web_crawling_retrieval = WebCrawlingRetrieval(documents, config)
    web_crawling_retrieval.init_llm_model()
    if save:
        web_crawling_retrieval.create_and_save_local(input_directory=config.get("input_directory",None) , persist_directory=config.get("persist_directory"), update=config.get("update",False))
    else:
        web_crawling_retrieval.create_load_vector_store(force_reload=config.get("force_reload",False), update = config.get("update",False))
    web_crawling_retrieval.retrieval_qa_chain()
    return web_crawling_retrieval

# Read config
web_crawling_params = get_config_info()
print(web_crawling_params)

# Scrap sites
filtered_sites = ["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "telagram.me", "reddit.com", "whatsapp.com", "wa.me"]
base_urls_list=["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/", "https://sambanova.ai/"]

crawler = WebCrawlingRetrieval()
docs, sources = crawler.web_crawl(base_urls_list, depth=1)
print(sources) # urls

# Chunk the text and create a vectorstore
db_path = None
config = config ={"force_reload":True}
conversation = set_retrieval_qa_chain(docs, config=config)

# Initialize the LLM and retrieval chain, and ask a question
user_question = "which kinds of memory can an agent have?"
response = conversation.qa_chain.invoke({"question": user_question})
print(f'Response ={response["answer"]}')





