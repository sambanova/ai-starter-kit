import base64
import logging
import os
import shutil
import sys
import time
import uuid
from threading import Thread

import schedule
import streamlit as st
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import concurrent.futures
from typing import Any, Optional

from search_assistant.src.search_assistant import SearchAssistant
from utils.events.mixpanel import MixpanelEvents
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
APP_DESCRIPTION_PATH = os.path.join(kit_dir, 'streamlit', 'app_description.yaml')
# Minutes for scheduled cache deletion
EXIT_TIME_DELTA = 30

logging.basicConfig(level=logging.INFO)
logging.info('URL: http://localhost:8501')


def load_config() -> Any:
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def load_app_description() -> Any:
    with open(APP_DESCRIPTION_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


config = load_config()
st_description = load_app_description()
prod_mode = config.get('prod_mode', False)
additional_env_vars = config.get('additional_env_vars', None)


def delete_temp_dir(temp_dir: str) -> None:
    """Delete the temporary directory and its contents."""

    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logging.info(f'Temporary directory {temp_dir} deleted.')
        except:
            logging.info(f'Could not delete temporary directory {temp_dir}.')


def schedule_temp_dir_deletion(temp_dir: str, delay_minutes: int) -> None:
    """Schedule the deletion of the temporary directory after a delay."""

    schedule.every(delay_minutes).minutes.do(delete_temp_dir, temp_dir).tag(temp_dir)

    def run_scheduler() -> None:
        while schedule.get_jobs(temp_dir):
            schedule.run_pending()
            time.sleep(1)

    # Run scheduler in a separate thread to be non-blocking
    Thread(target=run_scheduler, daemon=True).start()


def handle_user_input(user_question: Optional[str]) -> None:
    if user_question:
        with st.spinner('Processing...'):
            if st.session_state.method == 'rag_query':
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    related_querys_future = executor.submit(
                        st.session_state.search_assistant.get_relevant_queries,
                        query=f'{st.session_state.query} - {user_question}',
                    )
                response = st.session_state.search_assistant.retrieval_call(user_question)
                sources = set(f'{doc.metadata["source"]}' for doc in response['source_documents'])
                st.session_state.related_queries_history.append(related_querys_future.result())
            elif st.session_state.method == 'basic_query':
                reformulated_query = st.session_state.search_assistant.reformulate_query_with_history(user_question)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    related_querys_future = executor.submit(
                        st.session_state.search_assistant.get_relevant_queries, query=reformulated_query
                    )
                response = st.session_state.search_assistant.basic_call(
                    query=user_question,
                    reformulated_query=reformulated_query,
                    search_method=st.session_state.tool[0],
                    max_results=st.session_state.max_results,
                    search_engine=st.session_state.search_engine,
                    conversational=True,
                )
                sources = set(response['sources'])
                st.session_state.related_queries_history.append(related_querys_future.result())

        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response['answer'])

        sources_text = ''
        for index, source in enumerate(sources, start=1):
            source_link = source
            sources_text += f'<font size="2" color="grey">{index}. {source_link}</font>  \n'

        st.session_state.sources_history.append(sources_text)

    # Show the app description only if chat history is empty
    if len(st.session_state.chat_history) == 0:
        with st.chat_message(
            'ai',
            avatar=os.path.join(repo_dir, 'images', 'icon.svg'),
        ):
            st.write(st_description.get('app_overview'))

    for ques, ans, source, related_queries in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
        st.session_state.sources_history,
        st.session_state.related_queries_history,
    ):
        with st.chat_message('user'):
            st.write(f'{ques}')

        with st.chat_message(
            'ai',
            avatar=os.path.join(repo_dir, 'images', 'icon.svg'),
        ):
            st.markdown(
                f'{ans}',
                unsafe_allow_html=True,
            )
            if st.session_state.show_sources:
                with st.popover('Sources', use_container_width=False):
                    sources_lines = source.split('\n')[:-1]
                    for i in range(len(sources_lines) // 3 + 1):
                        columns = st.columns(3)
                        for j in range(len(columns)):
                            if i * 3 + j >= len(sources_lines):
                                break
                            columns[j].container(border=True).markdown(
                                f'<font size="2" color="grey">{sources_lines[i * 3 + j]}</font>',
                                unsafe_allow_html=True,
                            )
            if related_queries:
                with st.expander('**Related questions**', expanded=False):
                    for question in related_queries:
                        st.markdown(
                            f'[{question}](https://www.google.com/search?q={question.replace(" ", "+")})',
                        )


def main() -> None:
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'icon.svg'),
    )

    # set buttons style
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #250E36;  /* Button background */
            color: #FFFFFF;             /* Button text color */
        }
        div.stButton > button:hover, div.stButton > button:focus  {
            background-color: #4E22EB;  /* Button background */
            color: #FFFFFF;             /* Button text color */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load Inter font from Google Fonts and apply globally
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">

        <style>
            /* Apply Exile font to all elements on the page */
            html, body, [class^="css"] :not(.material-icons) {
                font-family: 'Inter', sans-serif !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # add title and icon
    col1, col2, col3 = st.columns([4, 1, 4])
    with col2:
        st.image(os.path.join(repo_dir, 'images', 'search_assistant_icon.png'))
    st.markdown(
        """
        <style>
            .kit-title {
                text-align: center;
                color: #250E36 !important;
                font-size: 3.0em;
                font-weight: bold;
                margin-bottom: 0.5em;
            }
        </style>
        <div class="kit-title">Search Assistant</div>
    """,
        unsafe_allow_html=True,
    )

    initialize_env_variables(prod_mode, additional_env_vars)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'sources_history' not in st.session_state:
        st.session_state.sources_history = []
    if 'related_queries_history' not in st.session_state:
        st.session_state.related_queries_history = []
    if 'related_questions' not in st.session_state:
        st.session_state.related_questions = []
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True
    if 'search_assistant' not in st.session_state:
        st.session_state.search_assistant = None
    if 'tool' not in st.session_state:
        st.session_state.tool = 'serpapi'
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = 'google'
    if 'max_results' not in st.session_state:
        st.session_state.max_results = 5
    if 'method' not in st.session_state:
        st.session_state.method = 'basic_query'
    if 'query' not in st.session_state:
        st.session_state.query = None
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = True
    if 'st_session_id' not in st.session_state:
        st.session_state.st_session_id = str(uuid.uuid4())
    if 'session_temp_subfolder' not in st.session_state:
        st.session_state.session_temp_subfolder = st.session_state.st_session_id + '_db'
    if 'mp_events' not in st.session_state:
        st.session_state.mp_events = MixpanelEvents(
            os.getenv('MIXPANEL_TOKEN'),
            st_session_id=st.session_state.st_session_id,
            kit_name='search_assistant',
            track=prod_mode,
        )
        st.session_state.mp_events.demo_launch()

    with st.sidebar:
        # Inject HTML to display the logo in the sidebar at 70% width
        logo_path = os.path.join(repo_dir, 'images', 'dark-logo.png')
        with open(logo_path, 'rb') as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        st.sidebar.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded}" style="width:60%; display: block; max-width:100%;">
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.title('**Setup**')

        # Callout to get SambaNova API Key
        st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')

        st.markdown('Get your SerpApi key [here]( https://serpapi.com)')

        if not are_credentials_set(additional_env_vars):
            api_key, additional_vars = env_input_fields(additional_env_vars)
            if st.button('Save Credentials'):
                message = save_credentials(api_key, additional_vars, prod_mode)
                st.session_state.mp_events.api_key_saved()
                st.success(message)
                st.rerun()
        else:
            st.success('Credentials are set')
            if st.button('Clear Credentials'):
                save_credentials('', {var: '' for var in (additional_env_vars or [])}, prod_mode)
                st.rerun()

        if are_credentials_set(additional_env_vars):
            if prod_mode:
                sambanova_api_key = st.session_state.SAMBANOVA_API_KEY
                serpapi_api_key = st.session_state.SERPAPI_API_KEY
                st.session_state.tool = ['serpapi']
                st.session_state.search_engine = 'google'
            else:
                if 'SAMBANOVA_API_KEY' in st.session_state:
                    sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY') or st.session_state.SAMBANOVA_API_KEY
                else:
                    sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')
                serpapi_api_key = os.environ.get('SERPAPI_API_KEY')
                tool = st.radio('Select Search Tool to use', ['serpapi', 'serper', 'openserp'])
                if tool == 'serpapi':
                    st.session_state.tool = ['serpapi']
                    st.session_state.search_engine = st.selectbox('Search engine to use', ['google', 'bing'])
                elif tool == 'serper':
                    st.session_state.tool = ['serper']
                    st.session_state.search_engine = st.selectbox('Search engine to use', ['google'])
                elif tool == 'openserp':
                    st.session_state.tool = ['openserp']
                    st.session_state.search_engine = st.selectbox('Search engine to use', ['google', 'baidu'])

            st.session_state.max_results = st.slider('Max number of results to retrieve', 1, 20, 5)

            st.markdown('Method for retrieval')
            method = st.selectbox('Method for retrieval', ['Search and answer', 'Search and scrape sites'])
            if method == 'Search and scrape sites':
                st.session_state.query = st.text_input('Query')

            if st.button('set'):
                st.session_state.search_assistant = SearchAssistant(sambanova_api_key, serpapi_api_key)
                with st.spinner(
                    'setting searchAssistant' if method == 'Search and answer' else 'searching and scraping sites'
                ):
                    if method == 'Search and scrape sites':
                        st.session_state.method = 'rag_query'
                        if not st.session_state.query:
                            st.error('Please enter a query')
                        else:
                            # Create the temporal folder to this session if it doesn't exist
                            temp_folder = os.path.join(kit_dir, 'data', 'tmp', st.session_state.session_temp_subfolder)
                            scraper_state = st.session_state.search_assistant.search_and_scrape(
                                query=st.session_state.query,
                                search_method=st.session_state.tool[0],
                                max_results=st.session_state.max_results,
                                search_engine=st.session_state.search_engine,
                                persist_directory=temp_folder,
                            )
                            if prod_mode:
                                schedule_temp_dir_deletion(temp_folder, EXIT_TIME_DELTA)
                                st.toast(
                                    f'your session will be active for the next {EXIT_TIME_DELTA} minutes, '
                                    'after this time files will be deleted'
                                )
                            if scraper_state is not None:
                                st.error(scraper_state.get('message'))
                            st.session_state.input_disabled = False
                            st.session_state.mp_events.input_submitted('scrape_and_ingest_sites')
                            st.toast('Search done and knowledge base updated you can chat now')
                    elif method == 'Search and answer':
                        st.session_state.method = 'basic_query'
                        st.session_state.input_disabled = False
                        st.toast('Settings updated you can chat now')
            if st.session_state.search_assistant:
                if st.session_state.search_assistant.urls:
                    with st.expander('Scraped sites', expanded=True):
                        st.write(st.session_state.search_assistant.urls)

            with st.expander('Additional settings', expanded=True):
                st.markdown('**Interaction options**')
                st.markdown('**Note:** Toggle these at any time to change your interaction experience')
                st.session_state.show_sources = st.checkbox('Show sources', value=True)

                st.markdown('**Reset chat**')
                st.markdown(
                    '**Note:** Resetting the chat will clear all conversation history and not updated documents.'
                )
                if st.button('Reset conversation'):
                    st.session_state.chat_history = []
                    st.session_state.sources_history = []
                    st.session_state.related_queries_history = []
                    if st.session_state.search_assistant:
                        st.session_state.search_assistant.init_memory()
                    st.rerun()

    user_question = st.chat_input(
        'Ask questions about data in provided sites', disabled=st.session_state.input_disabled
    )
    if user_question is not None:
        st.session_state.mp_events.input_submitted('chat_input')
    handle_user_input(user_question)


if __name__ == '__main__':
    main()
