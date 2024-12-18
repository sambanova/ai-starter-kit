import argparse
import base64
import concurrent.futures
import copy
import os
import sys
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
import yt_dlp
from dotenv import load_dotenv
from langchain.output_parsers import OutputFixingParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import load_prompt
from pydantic import BaseModel, Field

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.model_wrappers.api_gateway import APIGateway

load_dotenv(os.path.join(repo_dir, '.env'))

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB in bytes


class FileSizeExceededError(Exception):
    pass


class Transcript(BaseModel):
    transcript: str = Field(description='audio transcription')


class Scribe:
    """Downloading, transcription, question answering and summarization class"""

    def __init__(
        self,
        sambanova_api_key: Optional[str] = None,
    ) -> None:
        """
        Create a new Scribe class

        Args:
        sambanova_api_key (str): sambanova Cloud env api key
        """

        config = self.get_config_info()
        self.llm_info = config[0]
        self.audio_model_info = config[1]
        self.prod_mode = config[2]
        self.sambanova_api_key: Optional[str] = sambanova_api_key
        self.audio_model = self.set_audio_model()
        self.llm = self.set_llm()
        self.reset_query_audio_conversation()

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

    def set_audio_model(self) -> BaseChatModel:
        """
        Sets the audio model.

        Returns:
        audio_model: The audio model.
        """
        audio_model = APIGateway.load_chat(
            type=self.audio_model_info['type'],
            streaming=False,
            max_tokens=self.audio_model_info['max_tokens'],
            temperature=self.audio_model_info['temperature'],
            model=self.audio_model_info['model'],
            sambanova_api_key=self.sambanova_api_key,
        )

        return audio_model

    def set_llm(self) -> Union[LLM, BaseChatModel]:
        """
        Sets the sncloud, or sambastudio LLM based on the llm type attribute.

        Returns:
        LLM: The SambaStudio Cloud or Sambastudio Langchain ChatModel.
        """
        llm = APIGateway.load_chat(
            type=self.llm_info['type'],
            streaming=False,
            do_sample=self.llm_info['do_sample'],
            max_tokens=self.llm_info['max_tokens'],
            temperature=self.llm_info['temperature'],
            model=self.llm_info['model'],
            process_prompt=False,
            sambanova_api_key=self.sambanova_api_key,
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

    def reset_query_audio_conversation(self) -> None:
        self.query_audio_conversation: List[Union[HumanMessage, AIMessage, SystemMessage]] = [
            AIMessage(
                'You are helpful assistant called Scribe developed by SambaNova Systems, '
                'you are helping users in general purpose tasks'
            )
        ]

    def query_audio(self, audio: Optional[Union[BytesIO, str]] = None, query: Optional[str] = None) -> str:
        if audio is not None:
            b64_audio = self.load_encode_audio(audio)
            self.query_audio_conversation.append(
                HumanMessage(
                    content=[
                        {'type': 'audio_content', 'audio_content': {'content': f'data:audio/mp3;base64,{b64_audio}'}}
                    ]
                )
            )
        if query is not None:
            self.query_audio_conversation.append(HumanMessage(f'{query}, explain your response'))

        chain = self.audio_model | StrOutputParser()
        response = chain.invoke(self.query_audio_conversation)
        self.query_audio_conversation.append(AIMessage(response))

        return response

    def query_audio_pipeline(self, audio: Union[BytesIO, str], query: str) -> str:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            transcription_future = executor.submit(self.transcribe_audio, audio)
            audio_result_future = executor.submit(self.query_audio, audio, query)
        transcription = transcription_future.result()
        audio_result = audio_result_future.result()

        conversation = [
            SystemMessage("""
                        You are helpful assistant called Scribe developed by SambaNova Systems.
                        the user will ask information about an audio.
                        You will get the user query, the audio transcription and an intermediate response generated by a model capable of listening the audio
                        Whit those give a final response to the user query
                        """),  # noqa: E501
            HumanMessage(
                f"""
                Transcript: {transcription}
                Intermediate Audio Response: {audio_result}
                Query: {query}
                """
            ),
        ]
        chain = self.llm | StrOutputParser()
        response = chain.invoke(conversation)
        return response

    def encode_to_base64(self, content: bytes) -> str:
        """Encode audio file to base64"""
        return base64.b64encode(content).decode('utf-8')

    def load_encode_audio(self, audio: Union[BytesIO, str]) -> str:
        if isinstance(audio, str):
            with open(audio, 'rb') as file:
                audio_bytes = file.read()
        else:
            # make a copy given BytesIO object is not thread safe
            audio_copy = copy.deepcopy(audio)
            audio_bytes = audio_copy.read()
        b64_audio = self.encode_to_base64(content=audio_bytes)
        return b64_audio

    def transcribe_audio(self, audio_file: Union[BytesIO, str]) -> str:
        b64_audio = self.load_encode_audio(audio_file)
        conversation = [
            AIMessage('You are Automatic Speech Recognition tool'),
            HumanMessage(
                content=[{'type': 'audio_content', 'audio_content': {'content': f'data:audio/mp3;base64,{b64_audio}'}}]
            ),
            HumanMessage(
                f"""Please transcribe the previous audio in the following format
                
                ```
                    {{
                        "transcript":"<audio transcription>"
                    }}
                ```
                
                Always return your response enclosed by ``` and using double quotes
                """
            ),
        ]
        parser = PydanticOutputParser(pydantic_object=Transcript)
        autofix_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm)
        chain = self.audio_model | autofix_parser

        return chain.invoke(conversation).transcript  # type: ignore

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
            'cookiefile': os.path.join(kit_dir, 'data', 'sample_yt_cookies.txt'),
            'username': os.environ.get('YOUTUBE_USERNAME'),
            'password': os.environ.get('YOUTUBE_PASSWORD'),
            'verbose': True,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe, summarize, or query audio')
    parser.add_argument(
        '--task', type=str, choices=['transcribe', 'summarize', 'query'], help='Task to perform', default='transcribe'
    )
    parser.add_argument('--source', type=str, choices=['youtube', 'local'], help='Source of the audio', default='local')
    parser.add_argument('--audio', type=str, help='Path or YouTube link to the audio', required=True)
    parser.add_argument('--query', type=str, help='query when using query audio task', default=None)

    args = parser.parse_args()

    scribe = Scribe()

    if args.source == 'youtube':
        audio_path = scribe.download_youtube_audio(args.audio)
    else:
        audio_path = args.audio

    assert audio_path is not None

    if args.task == 'transcribe':
        transcript = scribe.transcribe_audio(audio_path)
        print(f'Transcript: {transcript}')
    elif args.task == 'summarize':
        transcript = scribe.transcribe_audio(audio_path)
        summary = scribe.summarize(transcript)
        print(f'Summary: {summary}')
    elif args.task == 'query':
        if args.query is None:
            response = scribe.query_audio(audio=audio_path)
            print(f'Response: {response}')
        else:
            response = scribe.query_audio_pipeline(audio=audio_path, query=args.query)
            print(f'Response: {response}')
    if args.source == 'youtube':
        scribe.delete_downloaded_file(audio_path)
