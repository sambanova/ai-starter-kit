import logging
import os
import shutil
import sys
import time
import uuid
from threading import Thread
from typing import Optional

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import schedule
import streamlit as st

from multimodal_knowledge_retriever.src.multimodal_rag import MultimodalRetrieval
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

logging.basicConfig(level=logging.INFO)
logging.info('URL: http://localhost:8501')

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
ADDITIONAL_ENV_VARS = ['LVLM_BASE_URL', 'LVLM_API_KEY']
# Available models in dropdown menu
LVLM_MODELS = [
    'meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo',
    'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
    'llava-v1.5-7b-4096-preview',
]
# Available models in dropdown menu
LLM_MODELS = ['Meta-Llama-3.1-70B-Instruct', 'Meta-Llama-3.1-405B-Instruct', 'Meta-Llama-3.1-8B-Instruct']
# Minutes for scheduled cache deletion
EXIT_TIME_DELTA = 30


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


def handle_user_input(user_question: str) -> None:
    if user_question:
        with st.spinner('Processing...'):
            response = st.session_state.qa_chain(user_question)
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
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            st.write(f'{ans}')
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


def initialize_multimodal_retrieval() -> Optional[MultimodalRetrieval]:
    if are_credentials_set():
        try:
            return MultimodalRetrieval()
        except Exception as e:
            st.error(f'Failed to initialize MultimodalRetrieval: {str(e)}')
            return None
    return None


def main() -> None:
    with open(CONFIG_PATH, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    prod_mode = config.get('prod_mode', False)

    initialize_env_variables(prod_mode, ADDITIONAL_ENV_VARS)

    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )

    if 'multimodal_retriever' not in st.session_state:
        st.session_state.multimodal_retriever = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
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
    if 'session_temp_subfolder' not in st.session_state:
        if prod_mode:
            st.session_state.session_temp_subfolder = 'upload_' + str(uuid.uuid4())
        else:
            st.session_state.session_temp_subfolder = None

    st.title(':orange[SambaNova] Multimodal Assistant')
    user_question = st.chat_input('Ask questions about your data', disabled=st.session_state.input_disabled)
    if user_question is not None:
        handle_user_input(user_question)

    with st.sidebar:
        st.title('Setup')

        st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')

        if not are_credentials_set(ADDITIONAL_ENV_VARS):
            api_key, aditional_variables = env_input_fields(ADDITIONAL_ENV_VARS)
            if st.button('Save Credentials', key='save_credentials_sidebar'):
                message = save_credentials(api_key, aditional_variables, prod_mode)  # type: ignore
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
            table_summaries = st.toggle('summarize Tables', value=True)
            text_summaries = st.toggle('summarize Text', value=False)
            st.markdown('**3. Set retrieval steps**')
            raw_image_retrieval = st.toggle('Answer over raw images', value=True)
            st.caption(
                '**Note** If selected the kit will use raw images to generate the answers, \
                if not, image summaries will be used instead'
            )
            # hard setting of llm and lvlm (overwrites models from config.yaml)
            if prod_mode:
                st.markdown('**Optional Set a specific multimodal model and LLM**')
                lvlm_model = st.selectbox('Select the multimodal model to use', LVLM_MODELS, 0)
                llm_model = st.selectbox('Select the LLM to use', LLM_MODELS, 0)
                if st.button('set_model'):
                    st.session_state.multimodal_retriever.set_lvlm(lvlm_model)
                    st.session_state.multimodal_retriever.set_llm(llm_model)
                    # set again qa chain with out ingestion in case step 4 was done previously to avoid re ingestion
                    if st.session_state.qa_chain is not None:
                        if raw_image_retrieval:
                            qa_chain = st.session_state.multimodal_retriever.get_retrieval_chain(
                                image_retrieval_type='raw'
                            )
                        else:
                            qa_chain = st.session_state.multimodal_retriever.get_retrieval_chain(
                                image_retrieval_type='summary'
                            )
                        st.session_state.qa_chain = qa_chain
                    st.toast('Models updated')
            st.markdown('**4. Process your documents and create an in memory vector store**')
            st.caption('**Note:** Depending on the size and number of your documents, this could take several minutes')
            if st.button('Process'):
                if docs:
                    with st.spinner('Processing this could take a while...'):
                        if prod_mode:
                            schedule_temp_dir_deletion(
                                os.path.join(kit_dir, 'data', st.session_state.session_temp_subfolder), EXIT_TIME_DELTA
                            )
                            st.toast(
                                """your session will be active for the next 30 minutes, after this time files and
                                 vectorstores will be deleted"""
                            )
                        st.session_state.qa_chain = st.session_state.multimodal_retriever.st_ingest(
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
                    st.toast('Conversation reset. The next response will clear the history on the screen')


if __name__ == '__main__':
    main()
