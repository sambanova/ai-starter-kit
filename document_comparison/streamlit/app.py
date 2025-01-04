import logging
import os
import shutil
import sys
import uuid
from typing import Any, List, Optional, Dict

import streamlit as st
import yaml
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.events.mixpanel import MixpanelEvents
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials
from utils.parsing.sambaparse import parse_doc_universal

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
APP_DESCRIPTION_PATH = os.path.join(kit_dir, 'streamlit', 'app_description.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir, f'data/my-vector-db')

logging.basicConfig(level=logging.INFO)
logging.info('URL: http://localhost:8501')

def load_config() -> Any:
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

def load_app_description() -> Any:
    with open(APP_DESCRIPTION_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

def save_files_user(docs: List[UploadedFile]) -> str:
    """
    Save all user uploaded files in Streamlit to the tmp dir with their file names

    Args:
        docs (List[UploadFile]): A list of uploaded files in Streamlit

    Returns:
        str: path where the files are saved.
    """
    temp_folder = os.path.join(kit_dir, 'data/tmp')
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    else:
        # If there are already files there, delete them
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    # Save all selected files to the tmp dir with their file names
    for doc in docs:
        assert hasattr(doc, 'name'), 'doc has no attribute name.'
        assert callable(doc.getvalue), 'doc has no method getvalue.'
        temp_file = os.path.join(temp_folder, doc.name)
        with open(temp_file, 'wb') as f:
            f.write(doc.getvalue())

    return temp_folder

def handle_userinput(user_question: Optional[str]) -> None:
    if user_question:
        try:
            with st.spinner('Processing...'):
                response = st.session_state.conversation.invoke({'question': user_question})
            st.session_state.chat_history.append(user_question)
            st.session_state.chat_history.append(response['answer'])

            sources = set([f'{sd.metadata["filename"]}' for sd in response['source_documents']])
            sources_text = ''
            for index, source in enumerate(sources, start=1):
                source_link = source
                sources_text += f'<font size="2" color="grey">{index}. {source_link}</font>  \n'
            st.session_state.sources_history.append(sources_text)
        except Exception as e:
            st.error(f'An error occurred while processing your question: {str(e)}')

    for ques, ans, source in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
        st.session_state.sources_history,
    ):
        with st.chat_message('user'):
            st.write(f'{ques}')

        with st.chat_message(
            'ai',
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            st.write(f'{ans}')
            if st.session_state.show_sources:
                with st.expander('Sources'):
                    st.markdown(
                        f'<font size="2" color="grey">{source}</font>',
                        unsafe_allow_html=True,
                    )

    # show overview message when chat history is empty
    if len(st.session_state.chat_history) == 0:
        with st.chat_message(
            'ai',
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            st.write(load_app_description().get('app_overview'))

def initialize_document_analyzer(prod_mode: bool) -> Optional[DocumentAnalyzer]:
    if prod_mode:
        sambanova_api_key = st.session_state.SAMBANOVA_API_KEY
    else:
        if 'SAMBANOVA_API_KEY' in st.session_state:
            sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY') or st.session_state.SAMBANOVA_API_KEY
        else:
            sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')
    if are_credentials_set():
        try:
            return DocumentAnalyzer(sambanova_api_key=sambanova_api_key)
        except Exception as e:
            st.error(f'Failed to initialize DocumentAnalyzer: {str(e)}')
            return None
    return None

def get_document_text(pdf_only_mode=False, document_name="Document 1"):
    datasource_options = ['Choose a file', 'Enter plain text']
    datasource = st.selectbox('', datasource_options)

    if isinstance(datasource, str):
        if 'Choose' in datasource:
            if pdf_only_mode:
                document_path = st.file_uploader('Add PDF files', accept_multiple_files=False, type=['pdf'])
            else:
                document_path = st.file_uploader(
                    'Add files',
                    accept_multiple_files=False,
                    type=[
                        '.txt',
                        '.doc',
                        '.docx',
                        '.pdf',
                        '.csv',
                        '.tsv',
                    ],
                )
            if st.button('Parse Document 1'):
                document_text, _, _ = parse_doc_universal(
                        doc=document_path, lite_mode=pdf_only_mode
                    )
                document_text = "\n".join(document_text)
        else:
            document_text = st.text_input(f'Copy {document_name} text here', value=st.session_state.get(document_name, ''), type="default")
        st.session_state.documents[document_name] = document_text
    return document_text

def main() -> None:
    config = load_config()

    prod_mode = config.get('prod_mode', False)
    llm_type = 'SambaStudio' if config['llm']['api'] == 'sambastudio' else 'SambaNova Cloud'
    conversational = config['retrieval'].get('conversational', False)
    default_collection = 'ekr_default_collection'

    initialize_env_variables(prod_mode)

    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )

    # if 'conversation' not in st.session_state:
    #     st.session_state.conversation = None
    # if 'chat_history' not in st.session_state:
    #     st.session_state.chat_history = []
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True
    if 'sources_history' not in st.session_state:
        st.session_state.sources_history = []
    if 'document_analyzer' not in st.session_state:
        st.session_state.document_analyzer = None
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = True
    if 'st_session_id' not in st.session_state:
        st.session_state.st_session_id = str(uuid.uuid4())
    if 'mp_events' not in st.session_state:
        st.session_state.mp_events = MixpanelEvents(
            os.getenv('MIXPANEL_TOKEN'),
            st_session_id=st.session_state.st_session_id,
            kit_name='enterprise_knowledge_retriever',
            track=prod_mode,
        )
        st.session_state.mp_events.demo_launch()
    if 'documents' not in st.session_state:
        st.session_state.documents = {}

    st.title(':orange[SambaNova] Document Comparison')

    # Callout to get SambaNova API Key
    st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')

    if not are_credentials_set():
        api_key, additional_vars = env_input_fields(mode=llm_type)
        if st.button('Save Credentials', key='save_credentials_sidebar'):
            message = save_credentials(api_key, additional_vars, prod_mode)
            st.session_state.mp_events.api_key_saved()
            st.success(message)
            st.rerun()
    else:
        st.success('Credentials are set')
        if st.button('Clear Credentials', key='clear_credentials'):
            save_credentials('', '', prod_mode)  # type: ignore
            st.rerun()

    if are_credentials_set():
        if st.session_state.document_analyzer is None:
            st.session_state.document_analyzer = initialize_document_analyzer(prod_mode)

    pdf_only_mode = config.get('pdf_only_mode', False)

    st.markdown('**1. Select Document 1**')
    document_1_text = get_document_text(pdf_only_mode, document_name="Document 1")

    st.markdown('**2. Select Document 2**')
    document_2_text = get_document_text(pdf_only_mode, document_name="Document 2")

    st.markdown('**3. Enter your instruction here:**')
        
    user_question = st.chat_input('Describe the differences between the contents of the above two documents')
    if user_question is not None:
        st.session_state.mp_events.input_submitted('chat_input')
    handle_userinput(user_question)

class DocumentAnalyzer:
    def __init__(self, sambanova_api_key: str) -> None:
        self.get_config_info()
        self.sambanova_api_key = sambanova_api_key
        self.set_llm()

    def get_config_info(self):
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        self.llm_info = config['llm']
        self.pdf_only_mode = config['pdf_only_mode']

    def set_llm(self):
        self.llm = ChatSambaNovaCloud(
                sambanova_api_key=self.sambanova_api_key,
                model=self.llm_info['model'],
                max_tokens=self.llm_info['max_tokens'],
                temperature=self.llm_info['temperature'],
                top_k=self.llm_info['top_k'],
                top_p=self.llm_info['top_p'],
                streaming=self.llm_info['streaming'],
                stream_options={'include_usage':True}
            )

if __name__ == '__main__':
    main()
