import os
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import streamlit as st
from enterprise_knowledge_retriever.src.document_retrieval import DocumentRetrieval
from enterprise_knowledge_retriever.src.langgraph_rag import RAG

CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
PROMPTS_PATH = os.path.join(kit_dir,'prompts')
PERSIST_DIRECTORY = os.path.join(kit_dir,f"data/my-vector-db")

logging.basicConfig(level=logging.INFO)
logging.info("URL: http://localhost:8501")

def handle_userinput(user_question):
    if user_question:
        with st.spinner("Processing..."):
            response = st.session_state.conversation.call_rag(user_question)
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response["answer"])

        # List of sources
        sources =set([
            f'{sd.metadata["filename"]}'
            for sd in response["source_documents"]
        ])
        # Create a Markdown string with each source on a new line as a numbered list with links
        sources_text = ""
        for index, source in enumerate(sources, start=1):
            # source_link = f'<a href="about:blank">{source}</a>'
            source_link = source
            sources_text += (
                f'<font size="2" color="grey">{index}. {source_link}</font>  \n'
            )
        st.session_state.sources_history.append(sources_text)

    for ques, ans, source in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
        st.session_state.sources_history,
    ):
        with st.chat_message("user"):
            st.write(f"{ques}")

        with st.chat_message(
            "ai",
            avatar="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
        ):
            st.write(f"{ans}")
            if st.session_state.show_sources:
                with st.expander("Sources"):
                    st.markdown(
                        f'<font size="2" color="grey">{source}</font>',
                        unsafe_allow_html=True,
                    )

def main(): 
    documentRetrieval =  DocumentRetrieval()
    *_, embedding_model_info ,retrieval_info, _ = documentRetrieval.get_config_info()

    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "sources_history" not in st.session_state:
        st.session_state.sources_history = []
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    with st.sidebar:
        st.title("Setup")
        st.markdown("**1. Pick a datasource**")
        datasource = st.selectbox(
            "", ("Upload files (create new vector db)", "Use existing vector db")
        )
        if "Upload" in datasource:
            docs = st.file_uploader(
                "Add files", accept_multiple_files=True, type=[".eml", ".html", ".json", ".md", ".msg", ".rst", ".rtf", ".txt", ".xml", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic", ".csv", ".doc", ".docx", ".epub", ".odt", ".pdf", ".ppt", ".pptx", ".tsv", ".xlsx"]
            )
            st.markdown("**2. Process your documents and create vector store**")
            st.markdown(
                "**Note:** Depending on the size and number of your documents, this could take several minutes"
            )
            st.markdown("Create database")
            if st.button("Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    text_chunks = documentRetrieval.parse_doc(docs)
                    # get the text chunks
                    # text_chunks = documentRetrieval.get_text_chunks_with_metadata(docs=raw_text, meta_data=meta_data)
                    # create vector store
                    embeddings = documentRetrieval.load_embedding_model()
                    st.session_state.embeddings = embeddings
                    vectorstore = documentRetrieval.create_vector_store(text_chunks, embeddings, output_db=None)
                    st.session_state.vectorstore = vectorstore
                    # instantiate retriever
                    # search_kwargs = {"k": lg_configs["retrieval_configs"]["top_k"]}
                    # st.session_state.retriever = st.session_state.vectorstore.as_retriever()

                    # instantiate rag
                    rag = RAG(
                    config_path=CONFIG_PATH,
                    embeddings = st.session_state.embeddings,
                    vectorstore=st.session_state.vectorstore
                    )

                    lg_configs = rag.load_config(CONFIG_PATH)
                    print(lg_configs) 
                    
                    # Initialize chains
                    rag.init_llm()
                    rag.init_qa_chain()
                    rag.init_final_generation()
                    # Build nodes
                    workflow = rag.create_rag_nodes()
                    # Build graph
                    rag.build_rag_graph(workflow)

                    # create conversation chain
                    st.session_state.conversation = rag 
                    st.toast(f"File uploaded! Go ahead and ask some questions",icon='🎉')
            st.markdown("[Optional] Save database for reuse")
            save_location = st.text_input("Save location", "./data/my-vector-db").strip()
            if st.button("Process and Save database"):
                with st.spinner("Processing"):
                    # get pdf text
                    text_chunks = documentRetrieval.parse_doc(docs)
                    # get the text chunks
                    #text_chunks = documentRetrieval.get_text_chunks_with_metadata(docs=raw_text, meta_data=meta_data)
                    # create vector store
                    embeddings = documentRetrieval.load_embedding_model()
                    st.session_state.embeddings = embeddings
                    vectorstore = documentRetrieval.create_vector_store(text_chunks, embeddings, save_location)
                    st.session_state.vectorstore = vectorstore
                    # instantiate retriever
                    # search_kwargs = {"k": lg_configs["retrieval_configs"]["top_k"]}
                    # retriever = vectorstore.as_retriever()
                    # st.session_state.retriever = retriever
                    
                    # instantiate rag
                    rag = RAG(
                    config_path=CONFIG_PATH,
                    embeddings = st.session_state.embeddings,
                    vectorstore=st.session_state.vectorstore
                    )

                    lg_configs = rag.load_config(CONFIG_PATH)
                    print(lg_configs) 
                    
                    # Initialize chains
                    rag.init_llm()
                    rag.init_qa_chain()
                    rag.init_final_generation()
                    # Build nodes
                    workflow = rag.create_rag_nodes()
                    # Build graph
                    rag.build_rag_graph(workflow)

                    # create RAG chain
                    st.session_state.conversation = rag 
                    st.toast(f"File uploaded and saved to {PERSIST_DIRECTORY}! Go ahead and ask some questions",icon='🎉')

        else:
            db_path = st.text_input(
                f"Absolute path to your DB folder",
                placeholder="E.g., /Users/<username>/path/to/your/vectordb",
            ).strip()
            st.markdown("**2. Load your datasource and create vectorstore**")
            st.markdown(
                "**Note:** Depending on the size of your vector database, this could take a few seconds"
            )
            if st.button("Load"):
                with st.spinner("Loading vector DB..."):
                    if db_path == "":
                        st.error("You must provide a provide a path", icon="🚨")
                    else:
                        if os.path.exists(db_path):
                            # load the vectorstore
                            embeddings = documentRetrieval.load_embedding_model()
                            st.session_state.embeddings = embeddings
                            vectorstore = documentRetrieval.load_vdb(db_path, embeddings)
                            st.toast("Database loaded")

                            # assign vectorstore to session
                            st.session_state.vectorstore = vectorstore
                            
                            rag = RAG(
                            config_path=CONFIG_PATH,
                            embeddings = st.session_state.embeddings,
                            vectorstore=st.session_state.vectorstore
                            )

                            lg_configs = rag.load_config(CONFIG_PATH)
                            print(lg_configs) 
                            
                            # Initialize chains
                            rag.init_llm()
                            rag.init_qa_chain()
                            rag.init_final_generation()
                            # Build nodes
                            workflow = rag.create_rag_nodes()
                             # Build graph
                            rag.build_rag_graph(workflow)

                            st.session_state.conversation = st.session_state.conversation = rag 
                        else:
                            st.error("database not present at " + db_path, icon="🚨")

        st.markdown("**3. Ask questions about your data!**")

        with st.expander("Additional settings", expanded=True):
            st.markdown("**Interaction options**")
            st.markdown(
                "**Note:** Toggle these at any time to change your interaction experience"
            )
            show_sources = st.checkbox("Show sources", value=True, key="show_sources")

            st.markdown("**Reset chat**")
            st.markdown(
                "**Note:** Resetting the chat will clear all conversation history"
            )
            if st.button("Reset conversation"):
                # reset create conversation chain
                st.session_state.conversation = []
                st.session_state.chat_history = []

                lg_configs = rag.load_config(CONFIG_PATH)
                print(lg_configs) 
                # Initialize chains
                rag.init_llm()
                rag.init_qa_chain()
                # Build nodes
                workflow = rag.create_rag_nodes()
                # Build graph
                rag.build_rag_graph(workflow)

                # create RAG chain
                st.session_state.conversation = rag
                st.toast(
                    "Conversation reset. The next response will clear the history on the screen"
                )


    st.title(":orange[SambaNova] Analyst Assistant")
    user_question = st.chat_input("Ask questions about your data")
    handle_userinput(user_question)

if __name__ == "__main__":
    main()
