import logging
import os
import shutil
import sys
import uuid
from typing import Any, List, Optional, Tuple

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)


import asyncio
import io
import re
import sys

import weave
import yaml
from dotenv import load_dotenv

from utils.eval.evaluator import BaseWeaveEvaluator, BaseWeaveRAGEvaluator, WeaveEvaluator
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

load_dotenv('../.env', override=True)

logging.basicConfig(level=logging.INFO)
logging.info('URL: http://localhost:8501')

APP_DESCRIPTION_PATH = os.path.join(kit_dir, 'streamlit', 'app_description.yaml')


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

    # Create the data/tmp folder if it doesn't exist
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

    return temp_file


def upload_file_options(
    option: str,
) -> Tuple[Optional[List[UploadedFile]], Optional[List[UploadedFile]]] | Optional[List[UploadedFile]]:
    docs: List[UploadedFile] | None
    text_docs: List[UploadedFile] | None
    st.markdown(
        'Example of a Q&A dataset [here](https://github.com/sambanova/ai-starter-kit/blob/main/eval_jumpstart/data/rag_data.csv)'
    )
    docs = st.file_uploader(
        'Add your Q&A dataset',
        accept_multiple_files=True,
        type=['.csv', '.json'],
    )
    if option == 'Evaluate multiple LLMs':
        return docs
    elif option == 'Evaluate Rag Chain':
        text_docs = st.file_uploader('Add PDF file', accept_multiple_files=True, type=['pdf'])
        return docs, text_docs
    else:
        raise ValueError(f'Invalid Evaluation Option: {option}.')


def initialize_base_evaluator(option: str) -> WeaveEvaluator:
    if option == 'Evaluate multiple LLMs':
        return BaseWeaveEvaluator()
    elif option == 'Evaluate Rag Chain':
        return BaseWeaveRAGEvaluator()
    else:
        raise ValueError(f'Invalid Evaluation Option: {option}.')


st_description = load_app_description()


def main() -> None:
    prod_mode = False
    llm_type = 'SambaNova Cloud'

    initialize_env_variables(prod_mode)

    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
    )

    if 'project_name' not in st.session_state:
        st.session_state.project_name = None
    if 'qna_file_path' not in st.session_state:
        st.session_state.qna_file_path = None
    if 'enable_evaluation' not in st.session_state:
        st.session_state.enable_evaluation = False
    if 'url' not in st.session_state:
        st.session_state.url = None
    if 'evaluation_option' not in st.session_state:
        st.session_state.evaluation_option = None
    if 'evaluation_option_submitted' not in st.session_state:
        st.session_state.evaluation_option_submitted = False
    if 'st_session_id' not in st.session_state:
        st.session_state.st_session_id = str(uuid.uuid4())

    st.title(':orange[SambaNova] Evaluation Kit')

    with st.sidebar:
        st.title('Setup')

        # Callout to get SambaNova API Key
        st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')
        st.markdown('Get your WANDB API key [here](https://wandb.ai/authorize)')

        if not are_credentials_set():
            api_key, additional_vars = env_input_fields(mode=llm_type)
            if st.button('Save Credentials', key='save_credentials_sidebar'):
                message = save_credentials(api_key, additional_vars, prod_mode)
                st.success(message)
                st.rerun()
        else:
            st.success('Credentials are set')
            if st.button('Clear Credentials', key='clear_credentials'):
                save_credentials('', '', prod_mode)  # type: ignore
                st.rerun()

        if st.session_state.project_name == '' or st.session_state.project_name is None:
            project_name = st.text_input('wandb project name', value='', type='default')

            if (
                st.button('Enter your wandb project name', key='project_name_sidebar')
                and project_name is not None
                and project_name != ''
            ):
                with st.spinner('Processing'):
                    st.session_state.project_name = project_name
                    try:
                        url_pattern = r'https?://\S+'
                        captured_output = io.StringIO()
                        sys.stdout = captured_output
                        weave.init(st.session_state.project_name)
                        sys.stdout = sys.__stdout__
                        captured_logs = captured_output.getvalue()
                        url = re.search(url_pattern, captured_logs)
                        if url is not None:
                            st.session_state.url = url.group(0)
                    except Exception as e:
                        st.error(f'Error: {e}.')
                    st.success('Wandb project name saved successfully!')
                    st.rerun()
        else:
            st.success('Wandb project name is set already set')

            st.markdown('**1. Evaluation options**')

            options = ['Evaluate multiple LLMs', 'Evaluate Rag Chain']

            if not st.session_state.evaluation_option_submitted:
                evaluation_option = st.selectbox('', options)

                submit_button = st.button('Enter Evaluation Option')

                if submit_button:
                    with st.spinner('Processing'):
                        st.session_state.evaluation_option_submitted = True
                        st.session_state.evaluation_option = evaluation_option
                        st.rerun()

            else:
                st.success('Evaluation Option selected!')

            docs: List[UploadedFile] | None
            text_docs: List[UploadedFile] | None

            if (
                isinstance(st.session_state.evaluation_option, str)
                and st.session_state.evaluation_option == 'Evaluate multiple LLMs'
            ):
                docs = upload_file_options(st.session_state.evaluation_option)  # type: ignore
                evaluator = initialize_base_evaluator(st.session_state.evaluation_option)
                if (
                    not (st.session_state.project_name == '' or st.session_state.project_name is None)
                    and docs is not None
                    and len(docs) != 0
                ):
                    if st.sidebar.button('Evaluate!'):
                        with st.spinner('Processing'):
                            try:
                                assert isinstance(docs, list)
                                temp_file = save_files_user(docs)
                                st.session_state.qna_file_path = temp_file
                                st.session_state.enable_evaluation = True

                            except Exception as e:
                                st.error(f'Error: {e}.')

            elif (
                isinstance(st.session_state.evaluation_option, str)
                and st.session_state.evaluation_option == 'Evaluate Rag Chain'
            ):
                docs, text_docs = upload_file_options(st.session_state.evaluation_option)  # type: ignore
                evaluator = initialize_base_evaluator(st.session_state.evaluation_option)
                if (
                    not (st.session_state.project_name == '' or st.session_state.project_name is None)
                    and docs is not None
                    and len(docs) != 0
                    and text_docs is not None
                    and len(text_docs) != 0
                ):
                    if st.sidebar.button('Evaluate!'):
                        with st.spinner('Processing'):
                            try:
                                temp_pdf_file = save_files_user(text_docs)
                                evaluator.populate_vectordb(temp_pdf_file)  # type: ignore
                                temp_file = save_files_user(docs)
                                st.session_state.qna_file_path = temp_file
                                st.session_state.enable_evaluation = True
                            except Exception as e:
                                st.error(f'Error: {e}.')

            with st.expander('Additional settings', expanded=True):
                st.markdown('**Reset app**')
                st.markdown('**Note:** Resetting the app will clear all settings')
                if st.button('Reset app'):
                    st.session_state.project_name = None
                    st.session_state.qna_file_path = None
                    st.session_state.enable_evaluation = False
                    st.session_state.url = None
                    st.session_state.evaluation_option = None
                    st.session_state.evaluation_option_submitted = False
                    st.toast('App reset.')
                    logging.info('App reset.')
                    st.rerun()

    st.write(st_description.get('app_overview'))

    if st.session_state.enable_evaluation:
        st.toast("""Evaluation in progress. This could take a while depending on the dataset size""")

        with st.spinner('Processing'):
            asyncio.run(
                evaluator.evaluate(  # type: ignore
                    filepath=st.session_state.qna_file_path, use_concurrency=True
                )
            )
            if st.session_state.url is not None:
                st.write(f"""Successfully submitted the evaluation. You can check the complete 
                    summary here: {st.session_state.url}""")
            else:
                st.write('Succesfully submitted the evaluation. Check the complete summary!')


if __name__ == '__main__':
    main()
