import logging
import os
import shutil
import sys
import uuid
from typing import Any, List, Optional, Dict
import time
import json
import streamlit as st
import yaml
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

from threading import Thread
import schedule

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

EXIT_TIME_DELTA = 30

logging.basicConfig(level=logging.INFO)
logging.info('URL: http://localhost:8501')

def load_config() -> Any:
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

def load_app_description() -> Any:
    with open(APP_DESCRIPTION_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

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
        self.system_message = config['system_message']
        self.max_retries = config['max_retries']        
        with open(os.path.join(repo_dir, config['templates']), "r") as ifile:
            self.templates = json.load(ifile)


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
    
    def get_analysis(self, prompt):
        messages = [
            ["system", self.system_message],
            ["user", prompt]
        ]

        retries = 0
        error_message = ""
        while retries < self.max_retries:
            try:
                response = self.llm.invoke(messages)
                completion = response.content.strip()
                usage = response.response_metadata["usage"]
                break
            except Exception as e:
                retries += 1
                time.sleep(10)
                error_message = str(e)
                pass
        
        if retries == self.max_retries:            
            completion = f"The model endpoint returned the following error: {error_message}"
            usage = None
        return completion, usage

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


def save_files_user(doc: UploadedFile, schedule_deletion: bool = True) -> str:
    """
    Save all user uploaded files in Streamlit to the tmp dir with their file names

    Args:
        docs (List[UploadFile]): A list of uploaded files in Streamlit.
        schedule_deletion (bool): wether or not to schedule the deletion of the uploaded files
            temporal folder. default to True.

    Returns:
        str: path where the files are saved.
    """

    # Create the temporal folder to this session if it doesn't exist
    temp_folder = os.path.join(kit_dir, 'data', 'tmp', st.session_state.session_temp_subfolder, doc.name)
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

    assert hasattr(doc, 'name'), 'doc has no attribute name.'
    assert callable(doc.getvalue), 'doc has no method getvalue.'
    temp_file = os.path.join(temp_folder, doc.name)
    with open(temp_file, 'wb') as f:
        f.write(doc.getvalue())

    if schedule_deletion:
        schedule_temp_dir_deletion(temp_folder, EXIT_TIME_DELTA)
        st.toast(
            """your session will be active for the next 30 minutes, after this time files 
            will be deleted"""
        )

    return temp_folder

def generate_prompt(instruction):
    doc1_title =  st.session_state.document_titles[0]
    doc2_title =  st.session_state.document_titles[1]
    return f"""-----Begin {doc1_title}-----
{st.session_state.documents[doc1_title]}
-----End {doc1_title}-----
-----Begin {doc2_title}-----
{st.session_state.documents[doc2_title]}
-----End {doc2_title}-----
{instruction}
"""    

def handle_userinput(instruction: str) -> None:   
    prompt = generate_prompt(instruction)
    start = time.time()    
    try:
        with st.spinner('Processing...'):
            completion, usage = st.session_state.document_analyzer.get_analysis(prompt)        
    except Exception as e:
        st.error(f'An error occurred while processing your instruction: {str(e)}')
    latency = time.time()-start
    
    with st.chat_message('user'):
        st.write(instruction)

    with st.chat_message(
        'ai',
        avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    ):
        st.write(completion)
        st.markdown(
                        '<font size="2" color="grey">Latency: %.1fs | Throughput: %d t/s | TTFT: %.2fs | Output Tokens: %d</font>'%(
                            latency,
                            usage["completion_tokens_per_sec"],
                            usage["time_to_first_token"],
                            usage["completion_tokens"]
                        ),
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

def get_document_text(pdf_only_mode=False, document_name="Document 1", prod_mode=True):
    st.markdown('Do you want to enter the text or upload a file?')
    datasource_options = ['Enter plain text', 'Upload a file']
    datasource = st.selectbox('', datasource_options, key="SB - " + document_name)
    document_text = ""
    if isinstance(datasource, str):
        if 'Upload' in datasource:
            if pdf_only_mode:
                doc = st.file_uploader('Add PDF files', accept_multiple_files=False, type=['pdf'], key="FU - " + document_name)
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
                    key="FU - " + document_name
                )
            if st.button(f'Parse {document_name}', key="Button - " + document_name):
                temp_folder = save_files_user(doc, schedule_deletion=prod_mode)
                document_text, _, _ = parse_doc_universal(
                        doc=temp_folder, lite_mode=pdf_only_mode
                    )
                document_text = "\n".join(document_text)
                logging.info(f'{document_name} parsed. Length of text = {len(document_text)}')
        else:            
            document_text = st.text_area(f'Enter {document_name} text here and hit Command + Enter to save your input', value=st.session_state.get(document_name, ''), key="TA - " + document_name)
        if document_text != "":
            st.session_state.documents[document_name] = document_text
        if document_name in st.session_state.documents:
            token_count = len(tokenizer.encode(st.session_state.documents[document_name]))
            st.markdown(f"{document_name} token count: {token_count}")            
    return document_text

def initialize_application_template():
    #st.markdown('#### Application Template') 
    app_templates = st.session_state.document_analyzer.templates
    selected_app_template = st.selectbox('Application Template', app_templates.keys(), key="SB - App Template")
    st.session_state.selected_app_template = selected_app_template
    if "document_1_title" in app_templates[selected_app_template]:
        st.session_state.document_titles[0] = app_templates[selected_app_template]["document_1_title"]
    else:
        st.session_state.document_titles[0] = "Document 1"

    if "document_2_title" in app_templates[selected_app_template]:
        st.session_state.document_titles[1] = app_templates[selected_app_template]["document_2_title"]
    else:
        st.session_state.document_titles[1] = "Document 2"
    st.session_state.prompts = app_templates[selected_app_template]["prompts"]
    

def main() -> None:
    config = load_config()

    prod_mode = config.get('prod_mode', False)
    llm_type = 'SambaStudio' if config['llm']['api'] == 'sambastudio' else 'SambaNova Cloud'

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
        st.selected_app_template = None    

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
    #document_name = st.text_input('name', value=)

    st.markdown('#### 3. Provide your comparison instruction')    
    template_default = "<Type out your own instruction below>"
    template_options = [template_default] + st.session_state.prompts
    template = st.selectbox('Templates', template_options, key="SB - templates")    
    user_instruction = st.chat_input(template)
    
    if user_instruction is None and template != template_default:
        user_instruction = template

    if user_instruction is not None \
        and st.session_state.document_titles[0] in st.session_state.documents \
        and st.session_state.document_titles[1] in st.session_state.documents:
        st.session_state.mp_events.input_submitted('chat_input')       
        handle_userinput(user_instruction)



if __name__ == '__main__':
    main()
