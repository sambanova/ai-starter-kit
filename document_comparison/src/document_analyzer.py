import json
import os
import shutil
import time
from io import BytesIO
from typing import List, Tuple

import tiktoken
import yaml
from langchain_sambanova import ChatSambaNova
from pydantic import SecretStr

from utils.parsing.sambaparse import parse_doc_universal

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
tokenizer = tiktoken.get_encoding('cl100k_base')


class DocumentAnalyzer:
    def __init__(self, sambanova_api_key: str) -> None:
        self.get_config_info()
        self.sambanova_api_key = SecretStr(sambanova_api_key)
        self.set_llm()

    def get_config_info(self) -> None:
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
        with open(os.path.join(repo_dir, config['templates']), 'r') as ifile:
            self.templates = json.load(ifile)

    def set_llm(self) -> None:
        self.llm = ChatSambaNova(api_key=self.sambanova_api_key, **self.llm_info)

    def delete_temp_dir(self, temp_dir: str) -> None:
        """Delete the temporary directory and its contents."""

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)  # throw an error if this fails

    def save_temp_dir(self, doc: BytesIO, temp_subfolder: str) -> str:
        """
        Save user uploaded file to the tmp dir with their file names

        Args:
            docs UploadFile: A list of uploaded files in Streamlit.
            schedule_deletion (bool): wether or not to schedule the deletion of the uploaded files
                temporal folder. default to True.

        Returns:
            str: path where the files are saved.
        """

        # Create the temporal folder to this session if it doesn't exist
        temp_dir = os.path.join(kit_dir, 'data', 'tmp', temp_subfolder)
        if os.path.exists(temp_dir):
            self.delete_temp_dir(temp_dir)
        os.makedirs(temp_dir)

        assert hasattr(doc, 'name'), 'doc has no attribute name.'
        assert callable(doc.getvalue), 'doc has no method getvalue.'
        temp_file = os.path.join(temp_dir, doc.name)
        with open(temp_file, 'wb') as f:
            f.write(doc.getvalue())

        return temp_dir

    def parse_document(self, document: BytesIO, temp_subfolder: str, pdf_only_mode: bool = True) -> str:
        temp_dir = self.save_temp_dir(document, temp_subfolder)
        document_text_lst, _, _ = parse_doc_universal(doc=temp_dir, lite_mode=pdf_only_mode)
        document_text = '\n'.join(document_text_lst)
        self.delete_temp_dir(temp_dir=temp_dir)
        return document_text

    def get_token_count(self, input_text: str) -> int:
        return len(tokenizer.encode(input_text))

    def generate_prompt_messages(
        self, instruction: str, doc1_title: str, doc1_text: str, doc2_title: str, doc2_text: str
    ) -> List[List[str]]:
        messages = [
            ['system', self.system_message],
            ['user', f'Consider the following {doc1_title}:\n{doc1_text}'],
            ['user', f'Consider the following {doc2_title}:\n{doc2_text}'],
            [
                'user',
                f'The {doc1_title} and {doc2_title} above are separate documents.\
                     Do not confuse them.\n{instruction}',
            ],
        ]
        return messages

    def get_analysis(self, messages: List[List[str]]) -> Tuple[str, str]:
        retries = 0
        error_message = ''
        while retries < self.max_retries:
            try:
                response = self.llm.invoke(messages)
                completion = str(response.content).strip()
                usage = response.response_metadata['token_usage']
                break
            except Exception as e:
                retries += 1
                time.sleep(10)
                error_message = str(e)
                pass

        if retries == self.max_retries:
            completion = f'The model endpoint returned the following error: {error_message}'
            usage = None
        return completion, usage
