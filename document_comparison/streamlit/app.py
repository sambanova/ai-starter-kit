import logging
import os
import sys
import time
import uuid
from typing import Any, Optional

import streamlit as st
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from document_comparison.src.document_analyzer import DocumentAnalyzer
from utils.events.mixpanel import MixpanelEvents
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

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


def handle_userinput(instruction: str) -> None:
    doc1_title = st.session_state.document_titles[0]
    doc2_title = st.session_state.document_titles[1]
    messages = st.session_state.document_analyzer.generate_prompt_messages(
        instruction,
        doc1_title,
        st.session_state.documents[doc1_title],
        doc2_title,
        st.session_state.documents[doc2_title],
    )
    start = time.time()

    with st.chat_message('user'):
        st.write(instruction)

    try:
        with st.spinner('Processing...'):
            completion, usage = st.session_state.document_analyzer.get_analysis(messages)
    except Exception as e:
        st.error(f'An error occurred while processing your instruction: {str(e)}')
    latency = time.time() - start

    with st.chat_message(
        'ai',
        avatar=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
    ):
        st.write(completion)
        st.markdown(
            '<font size="2" color="grey">Latency: %.1fs | Throughput: %d t/s | TTFT: %.2fs | Output Tokens: %d</font>'
            % (latency, usage['completion_tokens_per_sec'], usage['time_to_first_token'], usage['completion_tokens']),
            unsafe_allow_html=True,
        )


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


def get_document_text(pdf_only_mode: bool = False, document_name: str = 'Document 1', prod_mode: bool = True) -> str:
    st.markdown('Do you want to enter plain text or upload a file?')
    datasource_options = ['Enter plain text', 'Upload a file']
    datasource = st.selectbox(
        'File entry option', datasource_options, key='SB - ' + document_name, label_visibility='collapsed'
    )
    document_text = ''
    if isinstance(datasource, str):
        if 'Upload' in datasource:
            if pdf_only_mode:
                doc = st.file_uploader(
                    'Add PDF files', accept_multiple_files=False, type=['pdf'], key='FU - ' + document_name
                )
            else:
                doc = st.file_uploader(
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
                    key='FU - ' + document_name,
                )
            if doc:
                document_text = st.session_state.document_analyzer.parse_document(
                    doc, st.session_state.session_temp_subfolder, pdf_only_mode
                )
                logging.info(f'{document_name} parsed. Length of text = {len(document_text)}')
                st.markdown(f'Your document has been parsed and deleted from the remote server.')
        else:
            document_text = st.text_area(
                f'Enter {document_name} text here and hit Command + Enter to save your input',
                value=st.session_state.get(document_name, ''),
                key='TA - ' + document_name,
                height=400,
            )
        if document_text != '':
            st.session_state.documents[document_name] = document_text
        if document_name in st.session_state.documents:
            token_count = st.session_state.document_analyzer.get_token_count(st.session_state.documents[document_name])
            st.markdown(f'{document_name} token count: {token_count}')
    return document_text


def initialize_application_template() -> None:
    # st.markdown('#### Application Template')
    app_templates = st.session_state.document_analyzer.templates
    selected_app_template = st.selectbox('Application Template', app_templates.keys(), key='SB - App Template')  # type: str
    st.session_state.selected_app_template = selected_app_template
    if 'document_1_title' in app_templates[selected_app_template]:
        st.session_state.document_titles[0] = app_templates[selected_app_template]['document_1_title']
    else:
        st.session_state.document_titles[0] = 'Document 1'

    if 'document_2_title' in app_templates[selected_app_template]:
        st.session_state.document_titles[1] = app_templates[selected_app_template]['document_2_title']
    else:
        st.session_state.document_titles[1] = 'Document 2'
    st.session_state.prompts = app_templates[selected_app_template]['prompts']


def main() -> None:
    config = load_config()

    prod_mode = config.get('prod_mode', False)
    llm_type = 'SambaStudio' if config['llm']['api'] == 'sambastudio' else 'SambaNova Cloud'

    initialize_env_variables(prod_mode)

    st.set_page_config(
        page_title='AI Starter Kit', 
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'), 
        layout='wide'
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
    if 'session_temp_subfolder' not in st.session_state:
        st.session_state.session_temp_subfolder = 'upload_' + st.session_state.st_session_id
    if 'mp_events' not in st.session_state:
        st.session_state.mp_events = MixpanelEvents(
            os.getenv('MIXPANEL_TOKEN'),
            st_session_id=st.session_state.st_session_id,
            kit_name='document_comparison',
            track=prod_mode,
        )
        st.session_state.mp_events.demo_launch()
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'document_titles' not in st.session_state:
        st.session_state.document_titles = {}
    if 'selected_app_template' not in st.session_state:
        st.session_state.selected_app_template = None

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

        initialize_application_template()

        doc1_title = st.session_state.document_titles[0]
        st.markdown(f'#### 1. {doc1_title}')
        get_document_text(pdf_only_mode, document_name=doc1_title, prod_mode=prod_mode)

        doc2_title = st.session_state.document_titles[1]
        st.markdown(f'#### 2. {doc2_title}')
        get_document_text(pdf_only_mode, document_name=doc2_title, prod_mode=prod_mode)
        # document_name = st.text_input('name', value=)

        st.markdown('#### 3. Provide your comparison instruction')
        template_default = '<Type out your own instruction below>'
        template_options = [template_default] + st.session_state.prompts
        template = st.selectbox('Templates', template_options, key='SB - templates')  # type: str
        user_instruction = st.chat_input(template)

        if user_instruction is None and template != template_default:
            user_instruction = template

        if (
            user_instruction is not None
            and st.session_state.document_titles[0] in st.session_state.documents
            and st.session_state.document_titles[1] in st.session_state.documents
        ):
            if st.session_state.selected_app_template is not None:
                st.session_state.mp_events.input_submitted(st.session_state.selected_app_template)
            handle_userinput(user_instruction)


if __name__ == '__main__':
    main()
