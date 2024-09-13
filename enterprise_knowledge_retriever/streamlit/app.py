import os
import sys
import logging
import yaml
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from enterprise_knowledge_retriever.src.document_retrieval import DocumentRetrieval
from utils.visual.env_utils import env_input_fields, initialize_env_variables, are_credentials_set, save_credentials
from utils.vectordb.vector_db import VectorDb



CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir,f"data/my-vector-db")

logging.basicConfig(level=logging.INFO)
logging.info("URL: http://localhost:8501")

def handle_userinput(user_question):
    if user_question:
        try:
            with st.spinner("Processing..."):
                response = st.session_state.conversation.invoke({"question":user_question})
            st.session_state.chat_history.append(user_question)
            st.session_state.chat_history.append(response["answer"])

            sources = set([
                f'{sd.metadata["filename"]}'
                for sd in response["source_documents"]
            ])
            sources_text = ""
            for index, source in enumerate(sources, start=1):
                source_link = source
                sources_text += (
                    f'<font size="2" color="grey">{index}. {source_link}</font>  \n'
                )
            st.session_state.sources_history.append(sources_text)
        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")

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

def initialize_document_retrieval():
    if are_credentials_set():
        try:
            return DocumentRetrieval()
        except Exception as e:
            st.error(f"Failed to initialize DocumentRetrieval: {str(e)}")
            return None
    return None

def main(): 
   

    with open(CONFIG_PATH, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    prod_mode = config.get('prod_mode', False)
    default_collection = 'ekr_default_collection'

    initialize_env_variables(prod_mode)

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
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = True
    if 'document_retrieval' not in st.session_state:
        st.session_state.document_retrieval = None

    st.title(":orange[SambaNova] Analyst Assistant")

    with st.sidebar:
        st.title("Setup")

        #Callout to get SambaNova API Key
        st.markdown(
            "Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)"
        )

        if not are_credentials_set():
            url, api_key = env_input_fields()
            if st.button("Save Credentials", key="save_credentials_sidebar"):
                message = save_credentials(url, api_key, prod_mode)
                st.success(message)
                st.rerun()
        else:
            st.success("Credentials are set")
            if st.button("Clear Credentials", key="clear_credentials"):
                save_credentials("", "", prod_mode)
                st.rerun()

        if are_credentials_set():
            if st.session_state.document_retrieval is None:
                st.session_state.document_retrieval = initialize_document_retrieval()

        if st.session_state.document_retrieval is not None:
            st.markdown("**1. Pick a datasource**")
            

            # Conditionally set the options based on prod_mode
            datasource_options = ["Upload files (create new vector db)"]
            if not prod_mode:
                datasource_options.append("Use existing vector db")
            
            datasource = st.selectbox("", datasource_options)

            if "Upload" in datasource:
                if config.get('pdf_only_mode', False):
                    docs = st.file_uploader(
                        "Add PDF files", accept_multiple_files=True, type=["pdf"]
                    )
                else:
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
                        try:
                            text_chunks = st.session_state.document_retrieval.parse_doc(docs)
                            embeddings = st.session_state.document_retrieval.load_embedding_model()
                            collection_name = default_collection if not prod_mode else None
                            vectorstore = st.session_state.document_retrieval.create_vector_store(text_chunks, embeddings, output_db=None, collection_name=collection_name)
                            st.session_state.vectorstore = vectorstore
                            st.session_state.document_retrieval.init_retriever(vectorstore)
                            st.session_state.conversation = st.session_state.document_retrieval.get_qa_retrieval_chain()
                            st.toast(f"File uploaded! Go ahead and ask some questions", icon='🎉')
                            st.session_state.input_disabled = False
                        except Exception as e:
                            st.error(f"An error occurred while processing: {str(e)}")
                
                if not prod_mode:
                    st.markdown("[Optional] Save database for reuse")
                    save_location = st.text_input("Save location", "./data/my-vector-db").strip()
                    if st.button("Process and Save database"):
                        with st.spinner("Processing"):
                            try:
                                text_chunks = st.session_state.document_retrieval.parse_doc(docs)
                                embeddings = st.session_state.document_retrieval.load_embedding_model()
                                vectorstore = st.session_state.document_retrieval.create_vector_store(text_chunks, embeddings, output_db=save_location, collection_name=default_collection)
                                st.session_state.vectorstore = vectorstore
                                st.session_state.document_retrieval.init_retriever(vectorstore)
                                st.session_state.conversation = st.session_state.document_retrieval.get_qa_retrieval_chain()
                                st.toast(f"File uploaded and saved to {save_location} with collection '{default_collection}'! Go ahead and ask some questions", icon='🎉')
                                st.session_state.input_disabled = False
                            except Exception as e:
                                st.error(f"An error occurred while processing and saving: {str(e)}")

            elif not prod_mode and "Use existing" in datasource:
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
                            st.error("You must provide a path", icon="🚨")
                        else:
                            if os.path.exists(db_path):
                                try:
                                    embeddings = st.session_state.document_retrieval.load_embedding_model()
                                    collection_name = default_collection if not prod_mode else None
                                    vectorstore = st.session_state.document_retrieval.load_vdb(db_path, embeddings, collection_name=collection_name)
                                    st.toast(f"Database loaded{'with collection ' + default_collection if not prod_mode else ''}")
                                    st.session_state.vectorstore = vectorstore
                                    st.session_state.document_retrieval.init_retriever(vectorstore)
                                    st.session_state.conversation = st.session_state.document_retrieval.get_qa_retrieval_chain()
                                    st.session_state.input_disabled = False
                                except Exception as e:
                                    st.error(f"An error occurred while loading the database: {str(e)}")
                            else:
                                st.error("Database not present at " + db_path, icon="🚨")
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
                    st.session_state.chat_history = []
                    st.session_state.sources_history = []
                    st.toast(
                        "Conversation reset. The next response will clear the history on the screen"
                    )

    user_question = st.chat_input("Ask questions about your data", disabled=st.session_state.input_disabled)
    handle_userinput(user_question)



if __name__ == "__main__":
    main()