import os
import sys
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, Union

import streamlit as st
import yaml
import yt_dlp
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

from utils.model_wrappers.api_gateway import APIGateway

sys.path.append(kit_dir)
sys.path.append(repo_dir)
load_dotenv(os.path.join(repo_dir, '.env'))


CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB in bytes


class FileSizeExceededError(Exception):
    pass


class Scribe:
    """Downloading, transcription and summarization class"""

    def __init__(self) -> None:
        """
        Create a new Scribe class
        """
        config = self.get_config_info()
        self.llm_info = config[0]
        self.audio_model_info = config[1]
        self.prod_mode = config[2]
        self.client = self.set_client()
        self.llm = self.set_llm()

    def get_config_info(self) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """
        Loads json config file
        """
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        llm_info = config['llm']
        audio_model_info = config['audio_model']
        prod_mode = config['prod_mode']

        return llm_info, audio_model_info, prod_mode

    def set_client(self) -> OpenAI:
        """
        Sets the transcription client.

        Returns:
        client: The transcription client.
        """
        if self.prod_mode:
            transcription_base_url = st.session_state.TRANSCRIPTION_BASE_URL
            transcription_api_key = st.session_state.TRANSCRIPTION_API_KEY

        else:
            if 'TRANSCRIPTION_API_KEY' in st.session_state:
                transcription_api_key = (
                    os.environ.get('TRANSCRIPTION_API_KEY') or st.session_state.TRANSCRIPTION_API_KEY
                )
            else:
                transcription_api_key = os.environ.get('TRANSCRIPTION_API_KEY')

            if 'TRANSCRIPTION_BASE_URL' in st.session_state:
                transcription_base_url = (
                    os.environ.get('TRANSCRIPTION_BASE_URL') or st.session_state.TRANSCRIPTION_BASE_URL
                )
            else:
                transcription_base_url = os.environ.get('TRANSCRIPTION_BASE_URL')

        return OpenAI(base_url=transcription_base_url, api_key=transcription_api_key)

    def set_llm(self) -> Union[LLM, BaseChatModel]:
        """
        Sets the sncloud, or sambastudio LLM based on the llm type attribute.

        Returns:
        LLM: The SambaStudio Cloud or Sambastudio Langchain LLM.
        """

        if self.prod_mode:
            sambanova_api_key = st.session_state.SAMBANOVA_API_KEY
        else:
            if 'SAMBANOVA_API_KEY' in st.session_state:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY') or st.session_state.SAMBANOVA_API_KEY
            else:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')

        llm = APIGateway.load_llm(
            type=self.llm_info['type'],
            streaming=False,
            coe=self.llm_info['coe'],
            do_sample=self.llm_info['do_sample'],
            max_tokens_to_generate=self.llm_info['max_tokens_to_generate'],
            temperature=self.llm_info['temperature'],
            select_expert=self.llm_info['select_expert'],
            process_prompt=False,
            sambanova_api_key=sambanova_api_key,
        )
        return llm

    def summarize(self, text: str, num: int = 5) -> str:
        """
        /Crete a bullet points summarY of the text input.

        Args:
            text (str): The text to summarize.
            num (int, optional): The number of bullet points to generate. Defaults to 5.

        Returns:
            str: The bullet points summary of the text.
        """
        prompt_template = load_prompt(os.path.join(kit_dir, 'prompts', 'summary.yaml'))
        chain = prompt_template | self.llm | StrOutputParser()
        summary = chain.invoke({'text': text, 'num': num})
        return summary

    def transcribe_audio(self, audio_file: BytesIO) -> str:
        transcript = self.client.audio.transcriptions.create(
            model=self.audio_model_info['model'],
            file=audio_file,
            language=self.audio_model_info['language'],
            temperature=self.audio_model_info['temperature'],
        )
        return transcript.text

    def download_youtube_audio(
        self, url: str, output_path: Optional[str] = None, max_filesize: int = MAX_FILE_SIZE
    ) -> Optional[str]:
        """
        Downloads the audio from a YouTube URL and saves it to the specified output path.

        Args:
            url (str): The YouTube URL to download.
            output_path (str, optional): The path to save the downloaded audio. Defaults to ./data.
            max_filesize (int, optional): The maximum file size in bytes. Defaults to MAX_FILE_SIZE.

        Returns:
            str: The path of the downloaded audio if successful, otherwise None.
        """
        if output_path is None:
            output_path = os.path.join(kit_dir, 'data')
        downloaded_filename = None

        def progress_hook(d: Dict[str, Any]) -> None:
            nonlocal downloaded_filename
            if d['status'] == 'finished':
                downloaded_filename = d['filename']
            elif d['status'] == 'downloading':
                if 'total_bytes' in d and d['total_bytes'] > max_filesize:
                    if 'tmpfilename' in d:
                        try:
                            os.remove(d['tmpfilename'])
                            print(f"Deleted temporary file: {d['tmpfilename']}")
                        except OSError as e:
                            print(f'Error deleting temporary file: {e}')
                    raise FileSizeExceededError(f'File size exceeds {max_filesize/1024/1024:.2f} MB limit')

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [
                {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }
            ],
            'outtmpl': output_path + '/%(title)s.%(ext)s',
            'progress_hooks': [progress_hook],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f'Successfully downloaded audio from: {url}')

            # Ensure the filename has .mp3 extension
            if downloaded_filename and not downloaded_filename.endswith('.mp3'):
                new_filename = os.path.splitext(downloaded_filename)[0] + '.mp3'
                if os.path.exists(new_filename):
                    downloaded_filename = new_filename

            return downloaded_filename

        except FileSizeExceededError as e:
            print(f'Skipped downloading {url}: {str(e)}')
        except yt_dlp.utils.DownloadError as e:
            print(f'An error occurred while downloading {url}: {str(e)}')
        except Exception as e:
            print(f'An unexpected error occurred while downloading {url}: {str(e)}')

        return None

    def delete_downloaded_file(self, file_path: str) -> None:
        """
        Deletes the specified file.

        Args:
            file_path (str): The path of the file to delete.
        """
        try:
            # Check if the file exists
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File '{file_path}' successfully deleted.")
            else:
                print(f"Error: File '{file_path}' does not exist.")
        except PermissionError:
            print(f"PermissionError: You do not have permission to delete the file '{file_path}'.")
        except OSError as e:
            print(f"OSError: Failed to delete the file '{file_path}' due to: {e.strerror}.")
        except Exception as e:
            print(f'An unexpected error occurred: {str(e)}')
