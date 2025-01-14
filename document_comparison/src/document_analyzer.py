import json
import os
import time
from typing import Tuple

import yaml
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')


class DocumentAnalyzer:
    def __init__(self, sambanova_api_key: str) -> None:
        self.get_config_info()
        self.sambanova_api_key = sambanova_api_key
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
        self.llm = ChatSambaNovaCloud(
            sambanova_api_key=self.sambanova_api_key,
            model=self.llm_info['model'],
            max_tokens=self.llm_info['max_tokens'],
            temperature=self.llm_info['temperature'],
            top_k=self.llm_info['top_k'],
            top_p=self.llm_info['top_p'],
            streaming=self.llm_info['streaming'],
            stream_options={'include_usage': True},
        )

    def get_analysis(self, prompt: str) -> Tuple[str, str]:
        messages = [['system', self.system_message], ['user', prompt]]

        retries = 0
        error_message = ''
        while retries < self.max_retries:
            try:
                response = self.llm.invoke(messages)
                completion = str(response.content).strip()
                usage = response.response_metadata['usage']
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
