import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import streamlit as st
from search_assistant.src.search_assistant import SearchAssistant

def handle_user_input(user_question):
    if user_question:
        with st.spinner("Processing..."):
            
            if st.session_state.method=="rag_query":
                response = st.session_state.search_assistant.retrieval_call(user_question)
            elif st.session_state.method=="basic_query":
                print(st.session_state.tool)
                response = st.session_state.search_assistant.basic_call(
                    query = user_question,
                    search_method = st.session_state.tool[0],
                    max_results = st.session_state.max_results,
                    search_engine = st.session_state.search_engine)
            st.session_state.chat_history.append(user_question)
            st.session_state.chat_history.append(response["answer"])

    for ques, ans, in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
    ):
        with st.chat_message("user"):
            st.write(f"{ques}")

        with st.chat_message(
            "ai",
            avatar="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
        ):
            st.write(f"{ans}")
    #TODO: add sources
             
def main():
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/wp-content/uploads/2021/05/logo_icon-footer.svg",
    )
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "tool" not in st.session_state:
        st.session_state.tool = "serpapi"
    if "search_engine" not in st.session_state:
        st.session_state.search_engine="google"
    if "max_results" not in st.session_state:
        st.session_state.max_results = 5
    if "method" not in st.session_state:
        st.session_state.method = "basic_query"
    if "sources_history" not in st.session_state:
        st.session_state.sources_history = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "search_assistant" not in st.session_state:
        st.session_state.search_assistant = None
    if "query" not in st.session_state:   
        st.session_state.query = None
    if "input_disabled" not in st.session_state:
        st.session_state.input_disabled = True
    
    st.title(":orange[SambaNova] Search Assistant")
    
    with st.sidebar:
        st.title("**Setup**")
        
        tool=st.radio("Select Search Tool to use", ["serpapi","serper","openserp"])
        if tool == "serpapi":
            st.session_state.tool = ["serpapi"]
            st.session_state.search_engine=st.selectbox("Method for retrieval", ["google","bing"])
        elif tool == "serper":
            st.session_state.tool = ["serper"]
            st.session_state.search_engine=st.selectbox("Method for retrieval", ["google"])
        elif tool == "openserp":
            st.session_state.tool = ["openserp"]
            st.session_state.search_engine=st.selectbox("Method for retrieval", ["google","baidu"])
            
        st.session_state.max_results=st.slider("Max number of results to retrieve", 1, 20, 5)
        
        st.markdown("Method for retrieval")
        method = st.selectbox("Method for retrieval", ["Search and answer","Search and scrape sites"])
        if method=="Search and scrape sites":
            st.session_state.query = st.text_input("Query")
            
        if st.button("set"):
            st.session_state.search_assistant=SearchAssistant()
            with st.spinner("setting searchAssistant" if method=="Search and answer" else "searching and scraping sites"):
                if method=="Search and scrape sites":
                    st.session_state.method="rag_query"
                    if not st.session_state.query:
                        st.error("Please enter a query")
                    else:
                        st.session_state.search_assistant.search_and_scrape(
                            query=st.session_state.query,
                            search_method=st.session_state.tool[0],
                            max_results=st.session_state.max_results,
                            search_engine=st.session_state.search_engine)
                        with st.expander("Scraped sites", expanded=True):
                            st.write(st.session_state.search_assistant.urls)
                        st.session_state.input_disabled=False
                        st.toast("Search done and knowledge base updated you can chat now")
                elif method=="Search and answer":
                    st.session_state.method=="basic_query"
                    st.session_state.input_disabled=False
                    st.toast("Settings updated you can chat now")
            
    user_question = st.chat_input("Ask questions about data in provided sites", disabled=st.session_state.input_disabled)
    handle_user_input(user_question)

if __name__ == '__main__':
    main()