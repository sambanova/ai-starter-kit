import streamlit as st
from openai import OpenAI
import os
import sys
import logging
import yaml
from io import BytesIO
from download_audio import download_youtube_audio, delete_downloaded_file
from dotenv import load_dotenv


# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)
load_dotenv(os.path.join(repo_dir, ".env"))

from utils.visual.env_utils import (
    env_input_fields,
    initialize_env_variables,
    are_credentials_set,
    save_credentials,
)

CONFIG_PATH = os.path.join(current_dir, "config.yaml")

logging.basicConfig(level=logging.INFO)
logging.info("URL: http://localhost:8501")


def load_config():
    with open(CONFIG_PATH, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


config = load_config()
prod_mode = config.get("prod_mode", False)
additional_env_vars = config.get("additional_env_vars", None)

API_KEY = os.getenv("SAMBANOVA_API_KEY")
BASE_URL = os.getenv("SAMBANOVA_BASE_URL")
AUDIO_MODEL = "whisper-large-v3"


def transcribe_audio(session_state, audio_file, config):
    transcript = session_state.client.audio.transcriptions.create(
        model=config.get("audio_model").get("select_expert"),
        file=audio_file,
        language=config.get("audio_model").get("language"),
        temperature=config.get("audio_model").get("temperature"),
    )
    return transcript.text


def set_client(session_state, base_url=None, api_key=None):
    print(f"base_url: {base_url}, api_key: {api_key}")
    if "client" not in session_state and api_key and base_url:
        session_state.client = OpenAI(base_url=base_url, api_key=api_key)


def setup_sidebar():
    with st.sidebar:
        st.title("Setup")
        st.markdown(
            "Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)"
        )

        if not are_credentials_set():
            API_KEY, additional_vars = env_input_fields()
            if st.button("Save Credentials", key="save_credentials_sidebar"):
                message = save_credentials(API_KEY, additional_vars, prod_mode)
                st.success(message)
                st.rerun()
        else:
            st.success("Credentials are set")
            if st.button("Clear Credentials"):
                save_credentials(
                    "", {var: "" for var in (additional_env_vars or [])}, prod_mode
                )
                st.rerun()


def process_audio(input_method, audio_file, youtube_link):
    if input_method == "YouTube link":
        st.write("Downloading audio from YouTube link ....")
        audio_file_path = download_youtube_audio(youtube_link)
        if not audio_file_path:
            st.error("Failed to download audio. Please try again.")
            return None
        st.write("Processing YouTube audio ...")
        with open(audio_file_path, "rb") as f:
            audio_file = BytesIO(f.read())
        audio_file.name = os.path.basename(audio_file_path)
        delete_downloaded_file(audio_file_path)
    return audio_file


def main():
    initialize_env_variables(prod_mode)

    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )
    st.title(":orange[SambaNova] Scribe")

    setup_sidebar()
    set_client(st.session_state, base_url=BASE_URL, api_key=API_KEY)

    try:
        input_method = st.radio(
            "Choose input method:", ["Upload audio file", "YouTube link"]
        )

        with st.form("sambanovaform"):
            audio_file = (
                st.file_uploader("Upload an audio file", type=["mp3", "m4a", "wav"])
                if input_method == "Upload audio file"
                else None
            )
            youtube_link = (
                st.text_input("Enter YouTube link:", "")
                if input_method == "YouTube link"
                else None
            )

            submitted = st.form_submit_button("Transcribe")

            if submitted:
                if (input_method == "Upload audio file" and not audio_file) or (
                    input_method == "YouTube link" and not youtube_link
                ):
                    st.error(f"Please {input_method.lower()}")
                else:
                    audio_file = process_audio(input_method, audio_file, youtube_link)
                    if audio_file:
                        if not API_KEY:
                            st.session_state.client = OpenAI(
                                base_url=BASE_URL, api_key=API_KEY
                            )

                        st.write("Transcribing audio in background...")
                        transcription_text = transcribe_audio(
                            st.session_state, audio_file, config
                        )
                        st.write("Transcription complete!")
                        st.markdown(r"$\Large{\textsf{Transcription}}$")
                        st.text_area("", transcription_text, height=300)

    except Exception as e:
        st.error(str(e))
        if st.button("Clear"):
            st.rerun()


if __name__ == "__main__":
    main()
