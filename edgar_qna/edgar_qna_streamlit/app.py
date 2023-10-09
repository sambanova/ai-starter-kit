import streamlit as st
from edgar_sec_qa import SecFilingQa

st.set_page_config(
    page_title="Q&A Bot for Edgar SEC filings",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto",
)
if "config" not in st.session_state:
    st.session_state["config"] = {}
if "sec_qa" not in st.session_state:
    st.session_state["sec_qa"] = None


@st.cache_resource
def load_edgar_data(config):
    sec_qa = SecFilingQa(config=config)
    sec_qa.init_embeddings()
    sec_qa.init_models()
    sec_qa.vector_db_sec_docs()
    sec_qa.retreival_qa_chain()
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
            "ticker": ticker,
        }
        with st.spinner(text="Fetching Edgar data..."):
            config = st.session_state["config"]
            config["persist_directory"] = f"chroma_db/{ticker}"
            st.session_state["config"] = config

            st.session_state["sec_qa"] = load_edgar_data(st.session_state["config"])
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
            result = sec_qa.answer_sec(question)
            st.write(result)
        else:
            st.write("Please load SEC data first.")
