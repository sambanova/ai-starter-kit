import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import yaml
import streamlit as st
from dotenv import load_dotenv
from typing import Tuple   
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, load_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from utils.sambanova_endpoint import SambaNovaEndpoint, SambaverseEndpoint
from vectordb.vector_db import VectorDb

import fitz
from data_extraction.src.multi_column import column_boxes

load_dotenv(os.path.join(repo_dir,'export.env'))


DB_TYPE = "chroma"
PERSIST_DIRECTORY = os.path.join(kit_dir,f"data/vectordbs/{DB_TYPE}_default")
K_RETRIEVED_DOCUMENTS = 3
SCORE_TRESHOLD = 0.6
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS_TO_GENERATE = 1200
CONFIG_PATH = os.path.join(kit_dir,'config.yaml')

def get_config_info() -> Tuple[str, list]:
    """
    Loads json config file
    """
    # Read config file
    with open(CONFIG_PATH, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    api_info = config["api"]
    loader = config["loader"]
    
    return api_info, loader

def get_pdf_text_and_metadata_pypdf2(pdf_doc, extra_tags=None):
    """Extract text and metadata from pdf document with pypdf2 loader

    Args:
        pdf_doc: path to pdf document

    Returns:
        list, list: list of extracted text and metadata per page
    """
    text = []
    metadata = []
    pdf_reader = PdfReader(pdf_doc)
    doc_name = pdf_doc.name
    for page in pdf_reader.pages:
        #page_number =pdf_reader.get_page_number(page)+1
        text.append(page.extract_text())
        metadata.append({"filename": doc_name})#, "page": page_number})
    return text, metadata


def get_pdf_text_and_metadata_fitz(pdf_doc):    
    """Extract text and metadata from pdf document with fitz loader

    Args:
        pdf_doc: path to pdf document

    Returns:
        list, list: list of extracted text and metadata per page
    """
    text = []
    metadata = []
    temp_folder = os.path.join(kit_dir,"data/tmp")
    temp_file = os.path.join(temp_folder,"file.pdf")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    with open(temp_file, "wb") as f:
        f.write(pdf_doc.getvalue())
    docs = fitz.open(temp_file)  
    for page, page in enumerate(docs):
        full_text = ''
        bboxes = column_boxes(page, footer_margin=100, no_image_text=True)
        for rect in bboxes:
            full_text += page.get_text(clip=rect, sort=True)
        text.append(full_text)
        metadata.append({"filename": pdf_doc.name})
    return text, metadata

def get_pdf_text_and_metadata_unstructured(pdf_doc):
    """Extract text and metadata from pdf document with unstructured loader

    Args:
        pdf_doc: path to pdf document

    Returns:
        list, list: list of extracted text and metadata per page
    """
    text = []
    metadata = []
    temp_folder = os.path.join(kit_dir,"data/tmp")
    temp_file = os.path.join(temp_folder,"file.pdf")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    with open(temp_file, "wb") as f:
        f.write(pdf_doc.getvalue())
    loader = UnstructuredPDFLoader(temp_file)
    docs_unstructured = loader.load()
    for doc in docs_unstructured:
        text.append(doc.page_content)
        metadata.append({"filename": pdf_doc.name})
    return text, metadata

def get_data_for_splitting(pdf_docs):
    """Extract text and metadata from all the pdf files

    Args:
        pdf_docs (list): list of pdf files

    Returns:
        list, list: list of extracted text and metadata per file
    """
    _, loader = get_config_info()
    files_data = []
    files_metadatas = []
    for i in range(len(pdf_docs)):
        if loader == "unstructured":
            text, meta = get_pdf_text_and_metadata_unstructured(pdf_docs[i])
        elif loader == "pypdf2":
            text, meta = get_pdf_text_and_metadata_pypdf2(pdf_docs[i])
        elif loader == "fitz":
            text, meta = get_pdf_text_and_metadata_fitz(pdf_docs[i])
        files_data.extend(text)
        files_metadatas.extend(meta)
    return files_data, files_metadatas


def get_text_chunks_with_metadata(docs, chunk_size, chunk_overlap, meta_data):
    """Gets text chunks. .

    Args:
    doc (list): list of strings with text to split 
    chunk_size (int): chunk size in number of tokens
    chunk_overlap (int): chunk overlap in numb8er of tokens
    metadata (list, optional): list of metadata in dictionary format.

    Returns:
        list: list of documents 
    """
    chunks_list = []
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
    for doc, meta in zip(docs, meta_data):
        chunks = text_splitter.create_documents([doc], [meta])
        for chunk in chunks:
            chunk.page_content = f"Source: {meta['filename'].split('.')[0]}, Text: \n{chunk.page_content}\n"
            chunks_list.append(chunk)
    return chunks_list

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
    api_info, _ = get_config_info()
    
    if api_info == "sambaverse":
        llm = SambaverseEndpoint(
                sambaverse_model_name="Meta/llama-2-70b-chat-hf",
                sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
                model_kwargs={
                    "do_sample": False, 
                    "max_tokens_to_generate": LLM_MAX_TOKENS_TO_GENERATE,
                    "temperature": LLM_TEMPERATURE,
                    "process_prompt": True,
                    "select_expert": "llama-2-70b-chat-hf"
                    #"stop_sequences": { "type":"str", "value":""},
                    # "repetition_penalty": {"type": "float", "value": "1"},
                    # "top_k": {"type": "int", "value": "50"},
                    # "top_p": {"type": "float", "value": "1"}
                }
            )
        
    elif api_info == "sambastudio":
        llm = SambaNovaEndpoint(
            model_kwargs={
                "do_sample": False,
                "temperature": LLM_TEMPERATURE,
                "max_tokens_to_generate": LLM_MAX_TOKENS_TO_GENERATE,
                #"stop_sequences": { "type":"str", "value":""},
                # "repetition_penalty": {"type": "float", "value": "1"},
                # "top_k": {"type": "int", "value": "50"},
                # "top_p": {"type": "float", "value": "1"}
            }
        )
        
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": SCORE_TRESHOLD, "k": K_RETRIEVED_DOCUMENTS},
    )
    qa_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        input_key="question",
        output_key="answer",
    )
    
    customprompt = load_prompt(os.path.join(kit_dir, "prompts/llama7b-knowledge_retriever-custom_qa_prompt.yaml"))

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
                    text_chunks = get_text_chunks_with_metadata(docs=raw_text, chunk_size=1200, chunk_overlap=150, meta_data=meta_data)
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
                    text_chunks = get_text_chunks_with_metadata(docs=raw_text, chunk_size=1200, chunk_overlap=150, meta_data=meta_data)
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
