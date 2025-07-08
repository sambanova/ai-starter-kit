import base64
import logging
import os
import sys
import time
import uuid
from io import BytesIO
from typing import Any, List, Optional, Tuple, Union

import streamlit as st
import yaml
from dotenv import load_dotenv
from st_utils import tabs

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from sambanova_scribe.src.scribe import MAX_FILE_SIZE, FileSizeExceededError, Scribe
from utils.events.mixpanel import MixpanelEvents
from utils.visual.env_utils import (
    are_credentials_set,
    env_input_fields,
    initialize_env_variables,
    save_credentials,
)

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
ADDITIONAL_ENV_VARS = []
logging.basicConfig(level=logging.INFO)


@st.cache_data
def init() -> Tuple[str, bool]:
    load_dotenv(os.path.join(repo_dir, '.env'), override=True)
    config = load_config()
    llm_type = 'SambaStudio' if config.get('llm', {}).get('type') == 'sambastudio' else 'SambaNova Cloud'
    prod_mode = config.get('prod_mode', False)
    return llm_type, prod_mode


@st.cache_data
def load_config() -> Any:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def process_audio(
    audio_file: BytesIO, input_method: str = 'Upload audio file', youtube_link: Optional[str] = None
) -> Optional[BytesIO]:
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


def set_session_state_variables() -> None:
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
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = None
    if 'recording' not in st.session_state:
        st.session_state.recording = None
    initialize_env_variables(prod_mode, ADDITIONAL_ENV_VARS)


def handle_audio_qa(query: str, audio: BytesIO) -> None:
    with st.session_state.spinner_placeholder, st.spinner('processing'):
        st.session_state.qa_history.append(query)
        st.session_state.qa_history.append(audio)
        response = st.session_state.sambanova_scribe.query_audio_pipeline(audio, query)
        st.session_state.qa_history.append(response)
        st.rerun()


def handle_chat(query: Optional[str], audio: Optional[BytesIO]) -> None:
    with st.session_state.spinner_placeholder, st.spinner('processing'):
        if audio is not None:
            transcript = st.session_state.sambanova_scribe.transcribe_audio(audio)
            st.session_state.chat_history.append(transcript)
            st.session_state.chat_history.append(audio)
        else:
            text = query
            st.session_state.chat_history.append(text)
            st.session_state.chat_history.append(None)

        response = st.session_state.sambanova_scribe.query_audio(audio, query)
        st.session_state.chat_history.append(response)

        st.rerun()


def render_chat(chat_history: List[Union[str, BytesIO]]) -> None:
    with st.container(height=500, border=False):
        for ques, audio, ans in zip(
            chat_history[::3],
            chat_history[1::3],
            chat_history[2::3],
        ):
            with st.chat_message('user'):
                st.write(f'{ques}')
                if audio is not None:
                    st.audio(audio)

            with st.chat_message(
                'ai',
                avatar=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
            ):
                st.write(f'{ans}')
        st.session_state.spinner_placeholder = st.empty()


def reset_conversation() -> None:
    st.session_state.chat_history = []
    st.session_state.qa_history = []
    st.session_state.sambanova_scribe.reset_query_audio_conversation()
    st.toast('QA and chat history reset')
    time.sleep(0.5)
    st.rerun()


def setup_sidebar() -> None:
    """Setup sidebar for Scribe application."""
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
            api_key, additional_vars = env_input_fields(ADDITIONAL_ENV_VARS, mode=llm_type)
            if st.button('Save Credentials', key='save_credentials_sidebar'):
                message = save_credentials(api_key, additional_vars, prod_mode)
                st.session_state.mp_events.api_key_saved()
                st.success(message)
                st.rerun()
        else:
            st.success('Credentials are set')
            if st.button('Clear Credentials'):
                save_credentials('', {}, prod_mode)
                st.rerun()

        if are_credentials_set(ADDITIONAL_ENV_VARS):
            if prod_mode:
                sambanova_api_key = st.session_state.SAMBANOVA_API_KEY
            else:
                if 'SAMBANOVA_API_KEY' in st.session_state:
                    sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY') or st.session_state.SAMBANOVA_API_KEY
                else:
                    sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')

            if st.session_state.sambanova_scribe is None:
                st.session_state.sambanova_scribe = Scribe(sambanova_api_key)
            with st.expander('Additional settings', expanded=True):
                st.markdown('**Reset chat**')
                st.markdown('**Note:** Resetting the Conversation will clear all tabs history')
                if st.button('Reset'):
                    reset_conversation()


def main() -> None:
    """Main function for Scribe application."""
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
    col1, col2, col3 = st.columns([8, 1, 8])
    with col2:
        st.image(os.path.join(repo_dir, 'images', 'scribe_icon.png'))
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
        <div class="kit-title">SambaNova Scribe</div>
    """, unsafe_allow_html=True)
    
    if are_credentials_set(ADDITIONAL_ENV_VARS):
        active_tab = tabs(['Transcribe', 'Chat', 'Audio QA'], 1)
        if active_tab != st.session_state.active_tab:
            st.session_state.active_tab = active_tab
            reset_conversation()
        if active_tab == 'Transcribe':  # with tab1:
            if st.session_state.sambanova_scribe is not None:
                try:
                    input_method = st.radio('Choose input method:', ['Upload audio file', 'YouTube link'])

                    with st.form('sambanovaform'):
                        audio_file = (
                            st.file_uploader('Upload an audio file', type=['mp3', 'm4a', 'wav'])
                            if input_method == 'Upload audio file'
                            else None
                        )
                        youtube_link = (
                            st.text_input('Enter YouTube link:', '') if input_method == 'YouTube link' else None
                        )

                        submitted = st.form_submit_button('Transcribe')

                        if submitted:
                            st.session_state.transcription_text = None
                            if (input_method == 'Upload audio file' and not audio_file) or (
                                input_method == 'YouTube link' and not youtube_link
                            ):
                                st.error(f'Please {input_method.lower()}')
                            else:
                                audio_file = process_audio(audio_file, input_method, youtube_link)  # type: ignore
                                if audio_file:
                                    st.write('Transcribing audio in background...')
                                    st.session_state.transcription_text = (
                                        st.session_state.sambanova_scribe.transcribe_audio(audio_file)
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

        elif active_tab == 'Chat':
            render_chat(st.session_state.chat_history)
            c1, c2 = st.columns([5, 1])
            with c1:
                user_query = st.chat_input(key='chat')
            with c2:
                recording = st.audio_input('', label_visibility='collapsed')
            if recording != st.session_state.recording:
                st.session_state.recording = recording
            else:
                recording = None
            if user_query is not None or recording is not None:
                handle_chat(user_query, recording)

        elif active_tab == 'Audio QA':
            render_chat(st.session_state.qa_history)
            c1, c2 = st.columns([3, 5])
            with c1:
                audio = st.file_uploader('audio', ('mp3', 'wav'), False, label_visibility='collapsed')
            with c2:
                user_query = st.chat_input(key='qa')

            if user_query is not None:
                if audio is None:
                    st.toast('First upload an audio')
                else:
                    handle_audio_qa(user_query, audio)


if __name__ == '__main__':
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'), 
        layout='wide'
    )

    llm_type, prod_mode = init()
    set_session_state_variables()
    setup_sidebar()
    main()
