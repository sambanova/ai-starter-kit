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


def get_vectorstore(text_chunks):
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


def get_conversation_chain(vectorstore):
    """
    Generate a conversation chain using a language model.

    This function uses a language model, specifically a SambaNovaEndpoint, to generate a conversation chain
    based on the input vector store of text chunks.

    Parameters:
    vectorstore (FAISSVectorStore): A Vector Store containing embeddings of text chunks used as context
                                    for generating the conversation chain.

    Returns:
    ConversationalRetrievalChain: A chain ready for QA
    """
    llm = SambaNovaEndpoint(
        model_kwargs={
            "do_sample": False,
            "temperature": 0.0,
            "max_tokens_to_generate": 1500,
        }
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    return conversation_chain


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
    llm = SambaNovaEndpoint(model_kwargs={"do_sample": False, "temperature": 0.0})

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


def get_custom_prompt():
    """
    Generate a custom prompt template for contextual question answering.

    This function creates and returns a custom prompt template that instructs the model on how to answer a question
    based on the provided context. The template includes placeholders for the context and question to be filled in
    when generating prompts.

    Returns:
    PromptTemplate: A custom prompt template for contextual question answering.
    """

    custom_prompt_template = """Use the following pieces of context to answer the question at the end. 
    If the answer is not in context for answering, say that you don't know, don't try to make up an answer or provide an answer not extracted from provided context. 
    Cross check if the answer is contained in provided context. If not than say "I do not have information regarding this."

    {context}

    Question: {question}
    Helpful Answer:"""
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
        sources_text = "Citations:  \n"
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
        st.markdown(f":red[Q: {ques}]")
        st.write(f"A: {ans}")

        if st.session_state.show_sources:
            # Use Markdown with inline HTML for the entire list
            st.markdown(
                f'<font size="2" color="grey">{source}</font>',
                unsafe_allow_html=True,
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False
    if "sources_history" not in st.session_state:
        st.session_state.sources_history = []

    st.header(":orange[SambaNova] analyst assitant")
    user_question = st.chat_input("Upload your PDFs and ask questions about them")
    handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Get answers from your docs")
        st.markdown("**1. Upload your PDFs**")
        pdf_docs = st.file_uploader("", accept_multiple_files=True)
        st.markdown("**2. Click process**")
        st.write(
            "Note: Depending on the size and number of your documents, this could take several minutes"
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text, meta_data = get_data_for_splitting(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text, meta_data)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_qa_retrieval_chain(vectorstore)
        st.markdown("**3. Ask questions about your PDFs!**")
        show_sources = st.checkbox("show sources", value=False, key="show_sources")


if __name__ == "__main__":
    main()
