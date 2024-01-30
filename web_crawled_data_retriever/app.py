import os
import sys
sys.path.append("../")

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
from src.models.sambanova_endpoint import SambaNovaEndpoint

nest_asyncio.apply()

def load_htmls(urls):
    docs=[]
    for url in urls:
        #print(url)
        loader = AsyncHtmlLoader(url, verify_ssl=False)
        docs.extend(loader.load())
    return docs

def link_filter(all_links, excluded_links):
    clean_excluded_links=set()
    for excluded_link in excluded_links:
        parsed_link=urlparse(excluded_link)
        clean_excluded_links.add(parsed_link.netloc + parsed_link.path)
    filtered_links = set()
    for link in all_links:
        # Check if the link contains any of the excluded links
        if not any(excluded_link in link for excluded_link in clean_excluded_links):
            filtered_links.add(link)
    return filtered_links

def find_links(docs, excluded_links=None):
    if excluded_links is None:
        excluded_links = []
    all_links = set()  
    excluded_link_suffixes = {".ico", ".svg", ".jpg", ".png", ".jpeg", "."}
    
    for doc in docs:
        page_content = doc.page_content
        base_url = doc.metadata["source"]
        excluded_links.append(base_url)
        soup = BeautifulSoup(page_content, 'html.parser')
        # Identify the main content section (customize based on HTML structure)
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        
        if main_content:
            links = main_content.find_all('a', href=True)
            for link in links:
                href = link['href']
                # Check if the link is not an anchor link and not in the excluded links or suffixes
                if (
                    not href.startswith(('#', 'data:', 'javascript:')) and
                    not any(href.endswith(suffix) for suffix in excluded_link_suffixes)
                ):
                    full_url, _ = urldefrag(urljoin(base_url, href))
                    all_links.add(full_url)
                    
    all_links=link_filter(all_links, set(excluded_links))
    return all_links

def clean_docs(docs):
    html2text_transformer = Html2TextTransformer()
    docs=html2text_transformer.transform_documents(documents=docs)
    return docs

def web_crawl(urls, excluded_links=None, depth = 1):
    if excluded_links == None:
        excluded_links = []
    excluded_links.extend(["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "telagram.me", "reddit.com", "whatsapp.com", "wa.me"])
    if depth > 3:
        depth = 3
    scrapped_urls=[]
    raw_docs=[]
    for _ in range(depth):
        scraped_docs = load_htmls(urls)
        scrapped_urls.extend(urls)
        urls=find_links(scraped_docs, excluded_links)
        excluded_links.extend(scrapped_urls)
        raw_docs.extend(scraped_docs)
    docs=clean_docs(scraped_docs)
    return docs, scrapped_urls

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(text)
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

def get_custom_prompt():
    """
    Generate a custom prompt template for contextual question answering.

    This function creates and returns a custom prompt template that instructs the model on how to answer a question
    based on the provided context. The template includes placeholders for the context and question to be filled in
    when generating prompts.

    Returns:
    PromptTemplate: A custom prompt template for contextual question answering.
    """
    # llama prompt template
    custom_prompt_template = """<s>[INST] <<SYS>>\n"Use the following pieces of context to answer the question at the end. 
        If the answer is not in context for answering, say that you don't know, don't try to make up an answer or provide an answer not extracted from provided context. 
        Cross check if the answer is contained in provided context. If not than say "I do not have information regarding this." 

        context
        {context}
        end of context
        <</SYS>>

        Question: {question}
        Helpful Answer: [/INST]"""

    CUSTOMPROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    
    return CUSTOMPROMPT

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
    
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 4},
        )

    qa_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        input_key="question",
        output_key="answer",
    )
    ## Inject custom prompt
    qa_chain.combine_documents_chain.llm_chain.prompt = get_custom_prompt()
    return qa_chain

def handle_userinput(user_question):
    if user_question:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response["answer"])

        # List of sources
        sources = set(f'{doc.metadata["source"]}'for doc in response["source_documents"])
        # Create a Markdown string with each source on a new line as a numbered list with links
        sources_text = ""
        for index, source in enumerate(sources, start=1):
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
    if "sources_history" not in st.session_state:
        st.session_state.sources_history = []
    if "base_urls_list" not in st.session_state:
        st.session_state.base_urls_list = []
    if "full_urls_list" not in st.session_state:
        st.session_state.full_urls_list = [] 
    if "docs" not in st.session_state:
        st.session_state.docs = []              
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.title(":orange[SambaNova] Web Crawling Assistant")
    user_question = st.chat_input("Ask questions about data in provided sites")
    handle_userinput(user_question)

    with st.sidebar:
        st.title("Setup")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**1. Pick a datasource**")
        datasource = st.selectbox(
            "", ("Select websites(create new vector db)", "Use existing vector db")
        )
        
        if "websites" in datasource:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**2. Include the Urls to crawl**")
            col_a1, col_a2 = st.columns((3,2))
            # Single-line input for adding URLs
            new_url = col_a1.text_input("Add URL:", "")
            col_a2.markdown(
                """
                <style>
                    div[data-testid="column"]:nth-of-type(2)
                    {
                        padding-top: 4%;
                    } 
                </style>
                """,unsafe_allow_html=True
            )
            if col_a2.button("Include URL") and new_url:
                st.session_state.base_urls_list.append(new_url)
            # Display the list of URLs
            with st.expander(f"{len(st.session_state.base_urls_list)} Selected URLs",expanded=False):
                st.write(st.session_state.base_urls_list)
            if st.button("Clear List"):
                st.session_state.base_urls_list = []
                st.session_state.full_urls_list = []
                st.experimental_rerun()
             
            st.markdown("<hr>", unsafe_allow_html=True)   
            st.markdown("**3. Choose the crawling depth**")
            depth = st.number_input("Depth for web crawling:", min_value=1, max_value=2, value=1)
            if st.button("Scrape sites"):
                with st.spinner("Processing"):
                    # get pdf text
                    st.session_state.docs, sources = web_crawl(st.session_state.base_urls_list, depth=depth)
                    st.session_state.full_urls_list=sources
                    st.experimental_rerun()
            with st.expander(f"{len(st.session_state.full_urls_list)} crawled URLs",expanded=False):
                st.write(st.session_state.full_urls_list)
                
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**4. Load sites and create vectorstore**")
            if st.button("Process"):
                with st.spinner("Processing"):
                    # get the text chunks
                    text_chunks = get_text_chunks(st.session_state.docs)
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
            st.markdown("<hr>", unsafe_allow_html=True)
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
                            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Ask questions about your data!**")

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
