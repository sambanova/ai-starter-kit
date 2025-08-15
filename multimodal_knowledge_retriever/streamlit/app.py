import base64
import logging
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from threading import Thread
from typing import Any, Callable, Generator, Optional

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import schedule
import streamlit as st

from multimodal_knowledge_retriever.src.multimodal_rag import MultimodalRetrieval
from utils.events.mixpanel import MixpanelEvents
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

logging.basicConfig(level=logging.INFO)
logging.info('URL: http://localhost:8501')

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
APP_DESCRIPTION_PATH = os.path.join(kit_dir, 'streamlit', 'app_description.yaml')
ADDITIONAL_ENV_VARS: list[str] = []
# Available models in dropdown menu
LVLM_MODELS = [
    'Llama-4-Maverick-17B-128E-Instruct',
]
# Available models in dropdown menu
LLM_MODELS = [
    'Llama-4-Maverick-17B-128E-Instruct',
    'Meta-Llama-3.3-70B-Instruct',
    'DeepSeek-R1-Distill-Llama-70B',
    'DeepSeek-R1',
    'DeepSeek-V3-0324',
    'Meta-Llama-3.1-8B-Instruct',
]
# Minutes for scheduled cache deletion
EXIT_TIME_DELTA = 30


def load_config() -> Any:
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def load_app_description() -> Any:
    with open(APP_DESCRIPTION_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


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


@contextmanager
def st_capture(output_func: Callable[[str], None]) -> Generator[StringIO, None, None]:
    """
    context manager to catch stdout and send it to an output streamlit element

    Args:
        output_func (function to write terminal output in

    Yields:
        Generator:
    """
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string: str) -> int:
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write  # type: ignore
        yield stdout


def handle_user_input(user_question: Optional[str]) -> None:
    if user_question:
        with st.spinner('Processing...'):
            response = st.session_state.multimodal_retriever.call(user_question)
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response['answer'])

        # List of sources
        sources = set(
            [
                f"""{sd.metadata["filename"]} {" - Page "+str(sd.metadata.get("page_number")) 
                            if sd.metadata.get("page_number")else " - "+sd.metadata["file_directory"].split("/")[-1]}"""
                for sd in response['source_documents']
            ]
        )
        image_sources = [
            os.path.join(sd.metadata['file_directory'], sd.metadata['filename'])
            for sd in response['source_documents']
            if sd.metadata['filename'].endswith(('.png', '.jpeg', '.jpg'))
        ]
        # Create a Markdown string with each source on a new line as a numbered list with links
        sources_text = ''
        for index, source in enumerate(sources, start=1):
            # source_link = f'<a href="about:blank">{source}</a>'
            source_link = source
            sources_text += f'<font size="2" color="grey">{index}. {source_link}</font>  \n'

        st.session_state.sources_history.append(sources_text)
        st.session_state.image_sources_history.append(image_sources)

    for ques, ans, source, image_source in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
        st.session_state.sources_history,
        st.session_state.image_sources_history,
    ):
        with st.chat_message('user'):
            st.write(f'{ques}')

        with st.chat_message(
            'ai',
            avatar=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
        ):
            formatted_ans = ans.replace('$', '\$')
            st.write(f'{formatted_ans}')
            if st.session_state.show_sources:
                c1, c2 = st.columns(2)
                with c1.expander('Sources'):
                    st.markdown(
                        f'<font size="2" color="grey">{source}</font>',
                        unsafe_allow_html=True,
                    )
                if image_source:
                    with c2.expander('Images'):
                        for image in image_source:
                            st.image(image)

    # show overview message when chat history is empty
    if len(st.session_state.chat_history) == 0:
        with st.chat_message(
            'ai',
            avatar=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
        ):
            st.write(load_app_description().get('app_overview'))


def initialize_multimodal_retrieval() -> Optional[MultimodalRetrieval]:
    if are_credentials_set():
        try:
            return MultimodalRetrieval(conversational=True)
        except Exception as e:
            st.error(f'Failed to initialize MultimodalRetrieval: {str(e)}')
            return None
    return None


def main() -> None:
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
    )
    
    # set buttons style
    st.markdown("""
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
        """, unsafe_allow_html=True)
    
    # Load Inter font from Google Fonts and apply globally
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">

        <style>
            /* Apply Exile font to all elements on the page */
            * {
                font-family: 'Inter', sans-serif !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # add title and icon
    col1, col2, col3 = st.columns([4, 1, 4])
    with col2:
        st.image(os.path.join(repo_dir, 'images', 'multimodal_icon.png'))
    st.markdown("""
        <style>
            .kit-title {
                text-align: center;
                color: #250E36 !important;
                font-size: 3.0em;
                font-weight: bold;
                margin-bottom: 0.5em;
            }
        </style>
        <div class="kit-title">Multimodal Assistant</div>
    """, unsafe_allow_html=True)
        
    config = load_config()

    prod_mode = config.get('prod_mode', False)
    llm_type = 'SambaStudio' if config.get('llm', {}).get('type') == 'sambastudio' else 'SambaNova Cloud'

    initialize_env_variables(prod_mode, ADDITIONAL_ENV_VARS)

    if 'multimodal_retriever' not in st.session_state:
        st.session_state.multimodal_retriever = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'sources_history' not in st.session_state:
        st.session_state.sources_history = []
    if 'image_sources_history' not in st.session_state:
        st.session_state.image_sources_history = []
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = True
    if 'st_session_id' not in st.session_state:
        st.session_state.st_session_id = str(uuid.uuid4())
    if 'session_temp_subfolder' not in st.session_state:
        if prod_mode:
            st.session_state.session_temp_subfolder = 'upload_' + st.session_state.st_session_id
        else:
            st.session_state.session_temp_subfolder = None
    if 'mp_events' not in st.session_state:
        st.session_state.mp_events = MixpanelEvents(
            os.getenv('MIXPANEL_TOKEN'),
            st_session_id=st.session_state.st_session_id,
            kit_name='multimodal_knowledge_retriever',
            track=prod_mode,
        )
        st.session_state.mp_events.demo_launch()

    with st.sidebar:
        
        # Inject HTML to display the logo in the sidebar at 70% width
        logo_path = os.path.join(repo_dir, 'images', 'SambaNova-dark-logo-1.png')
        with open(logo_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        st.sidebar.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded}" style="width:60%; display: block; max-width:100%;">
            </div>
        """, unsafe_allow_html=True)
        
        st.title('Setup')

        st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')

        if not are_credentials_set(ADDITIONAL_ENV_VARS):
            api_key, additional_variables = env_input_fields(ADDITIONAL_ENV_VARS, mode=llm_type)
            if st.button('Save Credentials', key='save_credentials_sidebar'):
                message = save_credentials(api_key, additional_variables, prod_mode)
                st.session_state.mp_events.api_key_saved()
                st.rerun()
                st.success(message)

        else:
            st.success('Credentials are set')
            if st.button('Clear Credentials', key='clear_credentials'):
                save_credentials('', ADDITIONAL_ENV_VARS, prod_mode)  # type: ignore
                st.rerun()

        if are_credentials_set(ADDITIONAL_ENV_VARS):
            if st.session_state.multimodal_retriever is None:
                st.session_state.multimodal_retriever = initialize_multimodal_retrieval()

        if st.session_state.multimodal_retriever is not None:
            st.markdown('**1. Upload your files**')
            docs = st.file_uploader('Add your files', accept_multiple_files=True, type=['pdf', 'jpg', 'jpeg', 'png'])
            st.markdown('**2. Set ingestion steps**')
            table_summaries = st.toggle('Use summarized Tables for retrieval', value=True)
            text_summaries = st.toggle('Use summarized Text for retrieval', value=False)
            st.caption('**Note** *If not enabled retrieval will be done over raw text and table contents*')
            st.markdown('**3. Set retrieval steps**')
            raw_image_retrieval = st.toggle('Answer over raw images', value=True)
            st.caption(
                '**Note** *If selected the kit will use raw images to generate the answers, \
                if not, image summaries will be used instead*'
            )
            # hard setting of llm and lvlm (overwrites models from config.yaml)
            st.markdown('**Optional Set a specific multimodal model and LLM**')
            lvlm_model = st.selectbox('Select the multimodal model to use', LVLM_MODELS, 0)
            llm_model = st.selectbox('Select the LLM to use', LLM_MODELS, 0)
            if st.button('set_model'):
                st.session_state.multimodal_retriever.set_lvlm(lvlm_model)
                st.session_state.multimodal_retriever.set_llm(llm_model)
                # set again qa chain with out ingestion in case step 4 was done previously to avoid re ingestion
                if st.session_state.multimodal_retriever.qa_chain is not None:
                    if raw_image_retrieval:
                        st.session_state.multimodal_retriever.set_retrieval_chain(image_retrieval_type='raw')
                    else:
                        st.session_state.multimodal_retriever.set_retrieval_chain(image_retrieval_type='summary')
                st.toast('Models updated')
            st.markdown('**4. Process your documents and create an in memory vector store**')
            st.caption('**Note:** Depending on the size and number of your documents, this could take several minutes')
            if st.button('Process'):
                if docs:
                    st.session_state.mp_events.input_submitted('document_ingest')
                    with st.status('Processing this could take a while...', expanded=True):
                        if prod_mode:
                            schedule_temp_dir_deletion(
                                os.path.join(kit_dir, 'data', st.session_state.session_temp_subfolder), EXIT_TIME_DELTA
                            )
                            st.toast(
                                """your session will be active for the next 30 minutes, after this time files and
                                vectorstores will be deleted"""
                            )
                        execution_scratchpad_output = st.empty()
                        with st_capture(execution_scratchpad_output.write):
                            st.session_state.multimodal_retriever.st_ingest(
                                docs,
                                table_summaries,
                                text_summaries,
                                raw_image_retrieval,
                                st.session_state.session_temp_subfolder,
                            )
                        st.toast('Vector DB successfully created!')
                        st.session_state.input_disabled = False
                        st.rerun()
                else:
                    st.error('You must provide at least one document', icon='🚨')
            st.markdown('**5. Ask questions about your data!**')
            with st.expander('Additional settings', expanded=True):
                st.markdown('**Reset chat**')
                st.markdown('**Note:** Resetting the chat will clear all conversation history')
                if st.button('Reset conversation'):
                    st.session_state.chat_history = []
                    st.session_state.sources_history = []
                    st.session_state.image_sources_history = []
                    st.session_state.multimodal_retriever.init_memory()
                    st.toast('Conversation reset. The next response will clear the history on the screen')
                    st.rerun()
    
    user_question = st.chat_input('Ask questions about your data', disabled=st.session_state.input_disabled)
    if user_question is not None:
        st.session_state.mp_events.input_submitted('chat_input')
    handle_user_input(user_question)


if __name__ == '__main__':
    main()
