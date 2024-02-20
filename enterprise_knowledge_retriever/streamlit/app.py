import os
import sys

sys.path.append("../")
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, load_prompt
from PyPDF2 import PdfReader

from utils.sambanova_endpoint import SambaNovaEndpoint
from vectordb.vector_db import VectorDb

from dotenv import load_dotenv
load_dotenv('../export.env')

DB_TYPE = "chroma"
PERSIST_DIRECTORY = f"data/vectordbs/{DB_TYPE}_default"

def get_pdf_text_and_metadata(pdf_doc):
    """Extract text and metadata from pdf document

    Args:
        pdf_doc: path to pdf document

    Returns:
        list, list: list of extracted text and metadata per page
    """
    text = []
    metadata = []
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text.append(page.extract_text())
        metadata.append({"filename": pdf_doc.name, "page": pdf_reader.get_page_number(page)})
    return text, metadata


def get_data_for_splitting(pdf_docs):
    """Extract text and metadata from all the pdf files

    Args:
        pdf_docs (list): list of pdf files

    Returns:
        list, list: list of extracted text and metadata per file
    """
    files_data = []
    files_metadatas = []
    for file in pdf_docs:
        text, meta = get_pdf_text_and_metadata(file)
        files_data.extend(text)
        files_metadatas.extend(meta)
    return files_data, files_metadatas


def get_qa_retrieval_chain(vectorstore):
    """
    Generate a qa_retrieval chain using a language model.

    This function uses a language model, specifically a SambaNovaEndpoint, to generate a qa_retrieval chain
    based on the input vector store of text chunks.

    Parameters:
    vectorstore (Chroma): A Vector Store containing embeddings of text chunks used as context
                          for generating the conversation chain.

    Returns:
    RetrievalQA: A chain ready for QA without memory
    """
    llm = SambaNovaEndpoint(
        model_kwargs={
            "do_sample": False,
            "temperature": 0.0,
            "max_tokens_to_generate": 2500,
        }
    )

    qa_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        input_key="question",
        output_key="answer",
    )
    
    customprompt = load_prompt("prompts/llama7b-knowledge_retriever-custom_qa_prompt.yaml")

    ## Inject custom prompt
    qa_chain.combine_documents_chain.llm_chain.prompt = customprompt
    return qa_chain


def get_conversational_qa_retrieval_chain(vectorstore):
    """
    Generate a conversational retrieval qa chain using a language model.

    This function uses a language model, specifically a SambaNovaEndpoint, to generate a conversational_qa_retrieval chain
    based on the chat history and the relevant retrieved content from the input vector store of text chunks.

    Parameters:
    vectorstore (Chroma): A Vector Store containing embeddings of text chunks used as context
                                    for generating the conversation chain.

    Returns:
    RetrievalQA: A chain ready for QA with memory
    """


def handle_userinput(user_question):
    if user_question:
        with st.spinner("Processing..."):
            response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response["answer"])

        # List of sources
        sources = [
            f'{sd.metadata["filename"]} (page {sd.metadata["page"]})'
            for sd in response["source_documents"]
        ]
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
            avatar="https://sambanova.ai/wp-content/uploads/2021/05/logo_icon-footer.svg",
        ):
            st.write(f"{ans}")
            if st.session_state.show_sources:
                with st.expander("Sources"):
                    st.markdown(
                        f'<font size="2" color="grey">{source}</font>',
                        unsafe_allow_html=True,
                    )

def main():
    load_dotenv()
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/wp-content/uploads/2021/05/logo_icon-footer.svg",
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # if "show_sources" not in st.session_state:
    #     st.session_state.show_sources = False
    if "sources_history" not in st.session_state:
        st.session_state.sources_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.title(":orange[SambaNova] Analyst Assistant")
    user_question = st.chat_input("Ask questions about your data")
    handle_userinput(user_question)

    vectordb = VectorDb()
    with st.sidebar:
        st.title("Setup")
        st.markdown("**1. Pick a datasource**")
        datasource = st.selectbox(
            "", ("Upload PDFs (create new vector db)", "Use existing vector db")
        )
        if "Upload" in datasource:
            pdf_docs = st.file_uploader(
                "Add PDF files", accept_multiple_files=True, type="pdf"
            )
            st.markdown("**2. Process your documents and create vector store**")
            st.markdown(
                "**Note:** Depending on the size and number of your documents, this could take several minutes"
            )
            st.markdown("Create database")
            if st.button("Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text, meta_data = get_data_for_splitting(pdf_docs)
                    # get the text chunks
                    text_chunks = vectordb.get_text_chunks(docs=raw_text, chunk_size=1000, chunk_overlap=200, meta_data=meta_data)
                    # create vector store
                    embeddings = vectordb.load_embedding_model()
                    vectorstore = vectordb.create_vector_store(text_chunks, embeddings, output_db=None, db_type=DB_TYPE)
                    st.session_state.vectorstore = vectorstore
                    # create conversation chain
                    st.session_state.conversation = get_qa_retrieval_chain(
                        st.session_state.vectorstore
                    )
                    st.toast(f"File uploaded! Go ahead and ask some questions",icon='🎉')
            st.markdown("[Optional] Save database for reuse")
            save_location = st.text_input("Save location", "./data/my-vector-db").strip()
            if st.button("Process and Save database"):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text, meta_data = get_data_for_splitting(pdf_docs)
                    # get the text chunks
                    text_chunks = vectordb.get_text_chunks(docs=raw_text, chunk_size=1000, chunk_overlap=200, meta_data=meta_data)
                    # create vector store
                    embeddings = vectordb.load_embedding_model()
                    vectorstore = vectordb.create_vector_store(text_chunks, embeddings, output_db=save_location, db_type=DB_TYPE)
                    st.session_state.vectorstore = vectorstore
                    # create conversation chain
                    st.session_state.conversation = get_qa_retrieval_chain(
                        st.session_state.vectorstore
                    ) 
                    st.toast(f"File uploaded and saved to {PERSIST_DIRECTORY}! Go ahead and ask some questions",icon='🎉')

        else:
            db_path = st.text_input(
                f"Absolute path to your {DB_TYPE} DB folder",
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
                            embeddings = vectordb.load_embedding_model()
                            vectorstore = vectordb.load_vdb(db_path, embeddings, db_type=DB_TYPE)
                            st.toast("Database loaded")

                            # assign vectorstore to session
                            st.session_state.vectorstore = vectorstore

                            # create conversation chain
                            st.session_state.conversation = get_qa_retrieval_chain(
                                st.session_state.vectorstore
                            )
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
                st.session_state.conversation = get_qa_retrieval_chain(
                    st.session_state.vectorstore
                )
                st.session_state.chat_history = []
                st.toast(
                    "Conversation reset. The next response will clear the history on the screen"
                )


if __name__ == "__main__":
    main()
