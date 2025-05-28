import logging
import os
import sys

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain.docstore.document import Document

from web_crawled_data_retriever.src.web_crawling_retriever import WebCrawlingRetrieval

load_dotenv(os.path.join(repo_dir, '.env'))
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

logging.basicConfig(level=logging.INFO)
logging.info('URL: http://localhost:8501')


def get_config_info() -> Any:
    """
    Loads json config file
    """
    # Read config file
    with open(CONFIG_PATH, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    web_crawling_params = config['web_crawling']

    return web_crawling_params


def set_retrieval_qa_chain(documents: List[Document] = [], config: Dict[str, Any] = {}, save: bool = False) -> Any:
    web_crawling_retrieval = WebCrawlingRetrieval(documents, config)
    web_crawling_retrieval.init_llm_model()
    if save:
        web_crawling_retrieval.create_and_save_local(
            input_directory=config.get('input_directory', None),
            persist_directory=config.get('persist_directory', None),
            update=config.get('update', False),
        )
    else:
        web_crawling_retrieval.create_load_vector_store(
            force_reload=config.get('force_reload', False), update=config.get('update', False)
        )
    web_crawling_retrieval.retrieval_qa_chain()
    return web_crawling_retrieval


def handle_userinput(user_question: str) -> None:
    """
    Handle user input and generate a response with sources, also update chat UI in streamlit app
    Args:
        user_question (str): The user's question or input.
    Returns:
        None
    """
    if user_question:
        with st.spinner('Processing...'):
            response = st.session_state.conversation.qa_chain.invoke({'question': user_question})
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response['answer'])

        # List of sources
        sources = set(f'{doc.metadata["source"]}' for doc in response['source_documents'])
        # Create a Markdown string with each source on a new line as a numbered list with links
        sources_text = ''
        for index, source in enumerate(sources, start=1):
            source_link = source
            sources_text += f'<font size="2" color="grey">{index}. {source_link}</font>  \n'

        st.session_state.sources_history.append(sources_text)

    for ques, ans, source in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
        st.session_state.sources_history,
    ):
        with st.chat_message('user'):
            st.write(f'{ques}')

        with st.chat_message(
            'ai',
            avatar=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
        ):
            st.write(f'{ans}')
            if st.session_state.show_sources:
                with st.expander('Sources'):
                    st.markdown(
                        f'<font size="2" color="grey">{source}</font>',
                        unsafe_allow_html=True,
                    )


def main() -> None:
    web_crawling_params = get_config_info()

    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
    )

    # set session state variables
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'sources_history' not in st.session_state:
        st.session_state.sources_history = []
    if 'base_urls_list' not in st.session_state:
        st.session_state.base_urls_list = []
    if 'full_urls_list' not in st.session_state:
        st.session_state.full_urls_list = []
    if 'docs' not in st.session_state:
        st.session_state.docs = []
    if 'vectorstore' not in st.session_state:
        st.session_state.db_path = None
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = True

    st.title(':orange[SambaNova] Web Crawling Assistant')

    # setup of the application
    with st.sidebar:
        st.title('Setup')
        # selecction of datasource urls, or vector database folder
        datasource = st.selectbox(
            '**1. Pick a datasource**',
            (
                'Select websites(create new vector db)',
                'Use existing vector db',
                'Add new websites to an existing vector db',
            ),
        )
        # input text area for urls
        if datasource is not None and ('Select websites' in datasource or 'Add new' in datasource):
            if datasource is not None and 'Add new' in datasource:
                st.session_state.db_path = st.text_input(
                    'Absolute path to your FAISS Vector DB folder',
                    placeholder='E.g., /Users/<username>/Downloads/<vectordb_folder>',
                ).strip()
                st.markdown('<hr>', unsafe_allow_html=True)
                st.markdown('**Load your datasource**')
                st.markdown('**Note:** Depending on the size of your vector database, this could take a few seconds')
                if st.button('Load'):
                    with st.spinner('Loading vector DB...'):
                        if st.session_state.db_path == '':
                            st.error('You must provide a provide a path', icon='🚨')
                        else:
                            if os.path.exists(st.session_state.db_path):
                                config = {'persist_directory': st.session_state.db_path}
                                # create conversation chain
                                st.session_state.conversation = set_retrieval_qa_chain(config=config)
                            else:
                                st.error('database not present at ' + st.session_state.db_path, icon='🚨')
                        st.toast('Database loaded successfully')
            # Select
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('**2. Include the Urls to crawl**')
            col_a1, col_a2 = st.columns((3, 2))
            # Single-line input for adding URLs
            new_url = col_a1.text_input(
                'Add URL:',
                '',
                help='Each time you press Add URL button, the url will be included in URLs to crawl list',
            )
            col_a2.markdown(
                """
                <style>
                    div[data-testid="column"]:nth-of-type(2)
                    {
                        padding-top: 4%;
                    } 
                </style>
                """,
                unsafe_allow_html=True,
            )
            if col_a2.button('Include URL') and new_url:
                st.session_state.base_urls_list.append(new_url)
            # Display the list of URLs
            with st.expander(f'{len(st.session_state.base_urls_list)} Selected URLs', expanded=False):
                st.write(st.session_state.base_urls_list)
            if st.button('Clear List'):
                st.session_state.base_urls_list = []
                st.session_state.full_urls_list = []
                st.rerun()
            # selection of crawling depth and crawling process
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('**3. Choose the crawling depth**')
            depth = st.number_input(
                'Depth for web crawling:',
                min_value=1,
                max_value=web_crawling_params['max_depth'],
                value=1,
                help=f"""Maximum depth to crawl limited to {web_crawling_params["max_depth"]} / Maximum crawled sites
                  limited to {web_crawling_params["max_scraped_websites"]}""",
            )
            if st.button('Scrape sites'):
                with st.spinner('Processing'):
                    # get pdf text
                    crawler = WebCrawlingRetrieval()
                    st.session_state.docs, sources = crawler.web_crawl(
                        st.session_state.base_urls_list, depth=int(depth)
                    )
                    st.session_state.full_urls_list = sources
                    st.rerun()
            with st.expander(f'{len(st.session_state.full_urls_list)} crawled URLs', expanded=False):
                st.write(st.session_state.full_urls_list)
            if 'Select websites' in datasource:
                # Processing of crawled documents, storing them in vector database and creating retrival chain
                st.markdown('<hr>', unsafe_allow_html=True)
                st.markdown('**4. Load sites and create vectorstore**')
                st.markdown('Create database')
                if st.button('Process'):
                    with st.spinner('Processing'):
                        # create conversation chain
                        st.session_state.db_path = None
                        config_conversation = {'force_reload': True}
                        st.session_state.conversation = set_retrieval_qa_chain(
                            st.session_state.docs, config=config_conversation
                        )
                        # storing vector database in disk
                    st.session_state.input_disabled = False
                st.markdown('[Optional] Save database for reuse')
                save_location = st.text_input('Save location', './data/my-vector-db').strip()
                if st.button('Process and Save database'):
                    with st.spinner('Processing'):
                        config = {
                            'persist_directory': save_location,
                        }
                        st.session_state.conversation = set_retrieval_qa_chain(
                            st.session_state.docs, config=config, save=True
                        )
                        st.toast('Database with new sites saved in ' + save_location)
                    st.session_state.input_disabled = False

            elif 'Add new' in datasource:
                st.markdown('<hr>', unsafe_allow_html=True)
                st.markdown('**4. Load sites and update vectorstore**')
                st.markdown('Create database')
                if st.button('Process'):
                    with st.spinner('Processing'):
                        config_update = {'persist_directory': st.session_state.db_path, 'update': True}
                        st.session_state.conversation = set_retrieval_qa_chain(
                            st.session_state.docs, config=config_update
                        )
                    st.session_state.input_disabled = False
                st.markdown('[Optional] Save database with new scraped sites for reuse')
                save_location = st.text_input('Save location', './data/my-vector-db').strip()
                if st.button('Process and Save database'):
                    with st.spinner('Processing'):
                        config_save = {
                            'input_directory': st.session_state.db_path,
                            'persist_directory': save_location,
                            'update': True,
                        }
                        st.session_state.conversation = set_retrieval_qa_chain(
                            st.session_state.docs, config=config_save, save=True
                        )
                        st.toast('Database with new sites saved in ' + save_location)
                    st.session_state.input_disabled = False

        # if vector database folder selected as datasource
        elif datasource is not None and 'Use existing' in datasource:
            st.session_state.db_path = st.text_input(
                'Absolute path to your FAISS Vector DB folder',
                placeholder='E.g., /Users/<username>/Downloads/<vectordb_folder>',
            ).strip()
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('**Load your datasource**')
            st.markdown('**Note:** Depending on the size of your vector database, this could take a few seconds')
            if st.button('Load'):
                with st.spinner('Loading vector DB...'):
                    if st.session_state.db_path == '':
                        st.error('You must provide a provide a path', icon='🚨')
                    else:
                        if os.path.exists(st.session_state.db_path):
                            config = {'persist_directory': st.session_state.db_path}
                            # create conversation chain
                            st.session_state.conversation = set_retrieval_qa_chain(config=config)
                            st.session_state.input_disabled = False
                        else:
                            st.error('database not present at ' + st.session_state.db_path, icon='🚨')
                    st.toast('Database loaded successfully')

        # show sources and reset conversation controls
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('**Ask questions about your data!**')

        with st.expander('Additional settings', expanded=True):
            st.markdown('**Interaction options**')
            st.markdown('**Note:** Toggle these at any time to change your interaction experience')
            show_sources = st.checkbox('Show sources', value=True, key='show_sources')

            st.markdown('**Reset chat**')
            st.markdown('**Note:** Resetting the chat will clear all conversation history and not updated documents.')
            if st.button('Reset conversation'):
                # reset create conversation chain
                if st.session_state.db_path:
                    config = {'persist_directory': st.session_state.db_path}
                    st.session_state.conversation = set_retrieval_qa_chain(config=config)
                else:
                    st.session_state.conversation = set_retrieval_qa_chain(documents=st.session_state.docs)
                st.session_state.chat_history = []
                st.toast('Conversation reset. The next response will clear the history on the screen')
    user_question = st.chat_input(
        'Ask questions about data in provided sites', disabled=st.session_state.input_disabled
    )
    if user_question is not None:
        handle_userinput(user_question)


if __name__ == '__main__':
    main()
