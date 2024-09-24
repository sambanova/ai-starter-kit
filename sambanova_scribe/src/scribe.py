import streamlit as st
from typing import Callable, Dict, List, Tuple, Union
from openai import OpenAI
import os
import sys
import logging
import yaml
from dotenv import load_dotenv
import yt_dlp
import streamlit as st

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)
load_dotenv(os.path.join(repo_dir, ".env"))


CONFIG_PATH = os.path.join(kit_dir, "config.yaml")

class FileSizeExceededError(Exception):
    pass

class Scribe():

    def __init__(self):
        config = self.get_config_info()
        self.llm_info=config[0]
        self.audio_model_info=config[1]
        self.prod_mode=config[2]
        self.client=self.set_client()
    
    def get_config_info(self) -> Tuple[str, str, str]:
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        llm_info = config['llm']
        audio_model_info = config['audio_model']
        prod_mode = config['prod_mode']

        return llm_info, audio_model_info, prod_mode
    
    def set_client(self):
        
        if self.prod_mode:
            transcription_base_url = st.session_state.TRANSCRIPTION_BASE_URL
            transcription_api_key = st.session_state.TRANSCRIPTION_API_KEY
            
        else:
            if 'TRANSCRIPTION_API_KEY' in st.session_state:
                transcription_api_key = os.environ.get('TRANSCRIPTION_API_KEY') or st.session_state.TRANSCRIPTION_API_KEY
            else:
                transcription_api_key = os.environ.get('TRANSCRIPTION_API_KEY')

            if 'TRANSCRIPTION_BASE_URL' in st.session_state:
                transcription_base_url  = os.environ.get('TRANSCRIPTION_BASE_URL') or st.session_state.TRANSCRIPTION_BASE_URL
            else:
                transcription_base_url  = os.environ.get('TRANSCRIPTION_BASE_URL')

        return OpenAI(base_url=transcription_base_url, api_key=transcription_api_key)
    
    def set_llm(self):
        if self.prod_mode:
            sambanova_api_key = st.session_state.SAMBANOVA_API_KEY
        else:
            if 'SAMBANOVA_API_KEY' in st.session_state:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY') or st.session_state.SAMBANOVA_API_KEY
            else:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')
        # TODO add model
        pass
    
    def summarize(self, text):
        # TODO add summarization method
        pass
    
    def transcribe_audio(self, audio_file):
        transcript = self.client.audio.transcriptions.create(
            model=self.audio_model_info["model"],
            file=audio_file,
            language=self.audio_model_info["language"],
            temperature=self.audio_model_info["temperature"],
        )
        return transcript.text
    
    def download_youtube_audio(self, url, output_path=None, max_filesize=25*1024*1024):  # 25 MB in bytes
        if output_path is None:
            output_path = os.path.join(kit_dir, "data")
        downloaded_filename = None

        def progress_hook(d):
            nonlocal downloaded_filename
            if d['status'] == 'finished':
                downloaded_filename =  d['filename']
            elif d['status'] == 'downloading':
                if 'total_bytes' in d and d['total_bytes'] > max_filesize:
                    if 'tmpfilename' in d:
                        try:
                            os.remove(d['tmpfilename'])
                            print(f"Deleted temporary file: {d['tmpfilename']}")
                        except OSError as e:
                            print(f"Error deleting temporary file: {e}")
                    raise FileSizeExceededError(f"File size exceeds {max_filesize/1024/1024:.2f} MB limit")

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_path + '/%(title)s.%(ext)s',
            'progress_hooks': [progress_hook],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Successfully downloaded audio from: {url}")
            
            # Ensure the filename has .mp3 extension
            if downloaded_filename and not downloaded_filename.endswith('.mp3'):
                new_filename = os.path.splitext(downloaded_filename)[0] + '.mp3'
                if os.path.exists(new_filename):
                    downloaded_filename = new_filename

            return downloaded_filename

        except FileSizeExceededError as e:
            print(f"Skipped downloading {url}: {str(e)}")
        except yt_dlp.utils.DownloadError as e:
            print(f"An error occurred while downloading {url}: {str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred while downloading {url}: {str(e)}")
        
        return None


    def delete_downloaded_file(self, file_path):
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
            print(f"An unexpected error occurred: {str(e)}")

