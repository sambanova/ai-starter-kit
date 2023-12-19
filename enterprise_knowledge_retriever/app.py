import os
import sys

sys.path.append("../")
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader

from src.models.sambanova_endpoint import SambaNovaEndpoint


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
        metadata.append({"filename": pdf_doc, "page": pdf_reader.get_page_number(page)})
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


def get_text_chunks(text, metadata):
    """Chunks the text

    Args:
        text (list): text data
        metadata (list): metadata

    Returns:
        list: chunks of text based on the RecursiveCharacterTextSplitter
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.create_documents(text, metadata)
    return chunks


def build_vectorstore(text_chunks):
    """
    Create and return a Vector Store for a collection of text chunks.

    This function generates a vector store using the FAISS library, which allows efficient similarity search
    over a collection of text chunks by representing them as embeddings.

    Parameters:
    text_chunks (list of str): A list of text chunks or sentences to be stored and indexed for similarity search.

    Returns:
    FAISSVectorStore: A Vector Store containing the embeddings of the input text chunks,
                     suitable for similarity search operations.
    """
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="BAAI/bge-large-en",
        embed_instruction="",  # no instruction is needed for candidate passages
        query_instruction="Represent this sentence for searching relevant passages: ",
        encode_kwargs=encode_kwargs,
    )
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore


def load_vectorstore(faiss_location):
    """
    Loads an existing vector store generated with the FAISS library
    Parameters:
    faiss_location (str): Path to the vector store
    Returns:
    FAISSVectorStore: A Vector Store containing the embeddings of the input text chunks, suitable for similarity search operations.
    """
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="BAAI/bge-large-en",
        embed_instruction="",  # no instruction is needed for candidate passages
        query_instruction="Represent this sentence for searching relevant passages: ",
        encode_kwargs=encode_kwargs,
    )
    vectorstore = FAISS.load_local(faiss_location, embeddings)
    return vectorstore


def get_qa_retrieval_chain(vectorstore):
    """
    Generate a qa_retrieval chain using a language model.

    This function uses a language model, specifically a SambaNovaEndpoint, to generate a qa_retrieval chain
    based on the input vector store of text chunks.

    Parameters:
    vectorstore (FAISSVectorStore): A Vector Store containing embeddings of text chunks used as context
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

    ## Inject custom prompt
    qa_chain.combine_documents_chain.llm_chain.prompt = get_custom_prompt()
    return qa_chain


def get_conversational_qa_retrieval_chain(vectorstore):
    """
    Generate a conversational retrieval qa chain using a language model.

    This function uses a language model, specifically a SambaNovaEndpoint, to generate a conversational_qa_retrieval chain
    based on the chat history and the relevant retrieved content from the input vector store of text chunks.

    Parameters:
    vectorstore (FAISSVectorStore): A Vector Store containing embeddings of text chunks used as context
                                    for generating the conversation chain.

    Returns:
    RetrievalQA: A chain ready for QA with memory
    """


def get_custom_prompt():
    """
    Generate a custom prompt template for contextual question answering.

    This function creates and returns a custom prompt template that instructs the model on how to answer a question
    based on the provided context. The template includes placeholders for the context and question to be filled in
    when generating prompts.

    Returns:
    PromptTemplate: A custom prompt template for contextual question answering.
    """

    # custom_prompt_template = """Use the following pieces of context to answer the question at the end.
    # If the answer is not in context for answering, say that you don't know, don't try to make up an answer or provide an answer not extracted from provided context.
    # Cross check if the answer is contained in provided context. If not than say "I do not have information regarding this."

    # {context}

    # Question: {question}
    # Helpful Answer:"""

    # llama prompt template
    custom_prompt_template = """[INST]<<SYS>> You are a helpful assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If the answer is not in the context, say that you don't know. Cross check if the answer is contained in provided context. If not than say "I do not have information regarding this.
    Do not use images or emojis in your answer. Keep the answer conversational and professional.<</SYS>>

    {context} 
    
    Question: {question} 
    Helpful answer: [/INST]"""

    CUSTOMPROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return CUSTOMPROMPT


def handle_userinput(user_question):
    if user_question:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response["answer"])

        # List of sources
        sources = [
            f'{sd.metadata["filename"].name} (page {sd.metadata["page"]})'
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
            st.markdown("**2. Process your documents**")
            st.markdown(
                "**Note:** Depending on the size and number of your documents, this could take several minutes"
            )

            if st.button("Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text, meta_data = get_data_for_splitting(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text, meta_data)

                    # create vector store
                    vectorstore = build_vectorstore(text_chunks)

                    # assign vectorstore to session
                    st.session_state.vectorstore = vectorstore

                    # create conversation chain
                    st.session_state.conversation = get_qa_retrieval_chain(
                        st.session_state.vectorstore
                    )

            st.markdown("**[Optional] Save database for reuse**")
            save_location = st.text_input("Save location", "./my-vector-db").strip()
            if st.button("Save database"):
                if st.session_state.vectorstore is not None:
                    st.session_state.vectorstore.save_local(save_location)
                    st.toast("Database saved to " + save_location)
                else:
                    st.error(
                        "You need to process your files before saving the database"
                    )

        else:
            db_path = st.text_input(
                "Absolute path to your FAISS Vector DB folder",
                placeholder="E.g., /Users/<username>/Downloads/<vectordb_folder>",
            ).strip()
            st.markdown("**2. Load your datasource**")
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
                            vectorstore = load_vectorstore(db_path)
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
