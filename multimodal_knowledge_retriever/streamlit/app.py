import os
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import streamlit as st
from multimodal_knowledge_retriever.src.multimodal import MultimodalRetrieval

logging.basicConfig(level=logging.INFO)
logging.info("URL: http://localhost:8501")

def handle_userinput(user_question):
    pass

def main():
    multimodalRetrieval = MultimodalRetrieval()
    
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )
    
    st.title(":orange[SambaNova] Multimodal Assistant")
    user_question = st.chat_input("Ask questions about your data")
    handle_userinput(user_question)
    
    with st.sidebar:
        st.title("Setup")

if __name__ == "__main__":
    main()