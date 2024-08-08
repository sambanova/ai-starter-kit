import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import streamlit as st
from src.edgar_sec import SecFiling
from dotenv import load_dotenv

PERSIST_DIRECTORY = os.path.join(kit_dir,"data/vectordbs")

load_dotenv(os.path.join(repo_dir,'.env'))

st.set_page_config(
    page_title="Q&A Bot for Edgar SEC filings",
    page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    layout="wide",
    initial_sidebar_state="auto",
)
if "config" not in st.session_state:
    st.session_state["config"] = {}
if "sec_qa" not in st.session_state:
    st.session_state["sec_qa"] = None


@st.cache_resource
def set_retrieval_qa_chain(config):
    sec_qa = SecFiling(config=config)
    sec_qa.init_llm_model()
    sec_qa.create_load_vector_store()
    sec_qa.retrieval_qa_chain()
    return sec_qa


with st.sidebar.form(key="Form1"):
    ticker = st.text_input(
        "Specify company Ticker to do QnA on most recent [annual report (10-K)](https://www.sec.gov/edgar/searchedgar/companysearch)",
        "TSLA",
    )
    ticker = ticker.lower()
    submitted1 = st.form_submit_button(label="Submit")

    if submitted1 and ticker:
        st.session_state["config"] = {
            "persist_directory": None,
            "tickers": [ticker],
        }
        with st.spinner(text="Fetching Edgar data..."):
            config = st.session_state["config"]
            config["persist_directory"] = f"{PERSIST_DIRECTORY}/{ticker}"
            st.session_state["config"] = config

            st.session_state["sec_qa"] = set_retrieval_qa_chain(st.session_state["config"])
        st.write("Edgar data Ingested")


st.title("Edgar Assistant")
st.write("Powered by [SambaNova](https://sambanova.ai/) LLM")

question = st.text_input(
    "Ask a question", "What does the report say are the main risks to this company?"
)

if st.button("Get Answer", key="button2"):
    with st.spinner(text="Asking LLM..."):
        sec_qa = st.session_state.get("sec_qa")
        if sec_qa is not None:
            response = sec_qa.answer_sec(question)
            st.write(response['answer'])
        else:
            st.write("Please load SEC data first.")
