import logging
import os
import sys
import uuid
from io import BytesIO
from typing import Any, Optional

import streamlit as st
import yaml
from dotenv import load_dotenv

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)
load_dotenv(os.path.join(repo_dir, '.env'))

from sambanova_scribe.src.scribe import MAX_FILE_SIZE, FileSizeExceededError, Scribe
from utils.events.mixpanel import MixpanelEvents
from utils.visual.env_utils import (
    are_credentials_set,
    env_input_fields,
    initialize_env_variables,
    save_credentials,
)

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
ADDITIONAL_ENV_VARS = ['TRANSCRIPTION_BASE_URL', 'TRANSCRIPTION_API_KEY']
logging.basicConfig(level=logging.INFO)
logging.info('URL: http://localhost:8501')


def load_config() -> Any:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


config = load_config()
prod_mode = config.get('prod_mode', False)


def setup_sidebar() -> None:
    """Setup sidebar for Scribe application."""
    with st.sidebar:
        st.title('Setup')
        st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')

        if not are_credentials_set(ADDITIONAL_ENV_VARS):
            api_key, additional_vars = env_input_fields(ADDITIONAL_ENV_VARS)
            if st.button('Save Credentials', key='save_credentials_sidebar'):
                message = save_credentials(api_key, additional_vars, prod_mode)
                st.session_state.sambanova_scribe = Scribe()
                st.success(message)
                st.session_state.mp_events.api_key_saved()
                st.rerun()
        else:
            st.success('Credentials are set')
            if st.button('Clear Credentials'):
                save_credentials('', {}, prod_mode)
                st.rerun()

        if are_credentials_set(ADDITIONAL_ENV_VARS):
            if st.session_state.sambanova_scribe is None:
                st.session_state.sambanova_scribe = Scribe()


def process_audio(input_method: str, audio_file: BytesIO, youtube_link: str) -> Optional[BytesIO]:
    """
    Process audio using Scribe.

    Args:
        input_method (str): The method used to input audio ('YouTube link' or 'Upload audio file').
        audio_file (BytesIO): The audio file to process.
        youtube_link (str): The YouTube link to download audio from.

    Returns:
        BytesIO: The processed audio file.
    """
    if input_method == 'YouTube link':
        st.session_state.mp_events.input_submitted('youtube_video_input')
        st.write('Downloading audio from YouTube link ....')
        audio_file_path = st.session_state.sambanova_scribe.download_youtube_audio(youtube_link)
        if not audio_file_path:
            st.error('Failed to download audio. Please try again.')
            return None
        st.write('Processing YouTube audio ...')
        with open(audio_file_path, 'rb') as f:
            audio_file = BytesIO(f.read())
        audio_file.name = os.path.basename(audio_file_path)
        st.session_state.sambanova_scribe.delete_downloaded_file(audio_file_path)
    elif input_method == 'Upload audio file':
        st.session_state.mp_events.input_submitted('audio_file_input')
        assert hasattr(audio_file, 'size')
        if audio_file.size > MAX_FILE_SIZE:
            raise FileSizeExceededError(f'File size exceeds {MAX_FILE_SIZE/1024/1024:.2f} MB limit')
    return audio_file


def main() -> None:
    """Main function for Scribe application."""
    initialize_env_variables(prod_mode, ADDITIONAL_ENV_VARS)

    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )

    if 'sambanova_scribe' not in st.session_state:
        st.session_state.sambanova_scribe = None
    if 'transcription_text' not in st.session_state:
        st.session_state.transcription_text = None
    if 'st_session_id' not in st.session_state:
        st.session_state.st_session_id = str(uuid.uuid4())
    if 'mp_events' not in st.session_state:
        st.session_state.mp_events = MixpanelEvents(
            os.getenv('MIXPANEL_TOKEN'),
            st_session_id=st.session_state.st_session_id,
            kit_name='sambanova_scribe',
            track=prod_mode,
        )
        st.session_state.mp_events.demo_launch()

    st.title(':orange[SambaNova] Scribe')
    setup_sidebar()
    if st.session_state.sambanova_scribe is not None:
        try:
            input_method = st.radio('Choose input method:', ['Upload audio file', 'YouTube link'])

            with st.form('sambanovaform'):
                audio_file = (
                    st.file_uploader('Upload an audio file', type=['mp3', 'm4a', 'wav'])
                    if input_method == 'Upload audio file'
                    else None
                )
                youtube_link = st.text_input('Enter YouTube link:', '') if input_method == 'YouTube link' else None

                submitted = st.form_submit_button('Transcribe')

                if submitted:
                    st.session_state.transcription_text = None
                    if (input_method == 'Upload audio file' and not audio_file) or (
                        input_method == 'YouTube link' and not youtube_link
                    ):
                        st.error(f'Please {input_method.lower()}')
                    else:
                        audio_file = process_audio(input_method, audio_file, youtube_link)  # type: ignore
                        if audio_file:
                            st.write('Transcribing audio in background...')
                            st.session_state.transcription_text = st.session_state.sambanova_scribe.transcribe_audio(
                                audio_file
                            )
                            st.toast('Transcription complete!')
            if st.session_state.transcription_text is not None:
                st.markdown(r'$\Large{\textsf{Transcription}}$')
                st.text_area('', st.session_state.transcription_text, height=150)
                if st.button('Summarize'):
                    st.session_state.mp_events.input_submitted('summarize_transcription')
                    summary = st.session_state.sambanova_scribe.summarize(st.session_state.transcription_text)
                    st.toast('Summarization complete!')
                    st.markdown(r'$\Large{\textsf{Bullet Point Summary:}}$')
                    st.text_area('', f'{summary}', height=150)

        except Exception as e:
            st.error(str(e))
            if st.button('Clear'):
                st.rerun()


if __name__ == '__main__':
    main()
