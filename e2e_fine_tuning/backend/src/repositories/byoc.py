import os
import re
import json
import logging
from typing import Any, Dict, List

from utils.byoc.src.snsdk_byoc_wrapper import BYOC

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class BYOCRepository:
    def __init__(self, byoc: BYOC, target_dir) -> None:
        self.__byoc = byoc
        self.target_dir = target_dir
        self.checkpoint = None

    def create_checkpoint_config(self, checkpoint_info: str) -> None:
        checkpoint_folder = None
        for root, dirs, files in os.walk(self.target_dir):
            if 'snapshots' in root and checkpoint_info['hf_model_name'].replace('/', '--') in root:
                checkpoint_folder = os.path.join(root, dirs[0])
                break

        # todo: custom exception
        if checkpoint_folder is None:
            return None

        checkpoint = {
            'model_name': checkpoint_info['model_name_sambastudio'],
            'publisher': checkpoint_info['publisher'],
            'description': checkpoint_info['description'],
            'param_count': checkpoint_info['param_count'],
            'checkpoint_path': checkpoint_folder,
        }

        self.checkpoint = checkpoint

        return checkpoint

    def get_suitable_apps(self) -> List[List[Dict[str, Any]]]:
        suitable_apps = self.__byoc.get_suitable_apps(self.checkpoint)
        return suitable_apps

    def update_app(self) -> None:
        suitable_apps = self.get_suitable_apps()
        self.checkpoint['app_id'] = suitable_apps[0][0]['id']

    def set_chat_template(self) -> None:
        jinja_chat_template = """ 
        {% for message in messages %}
            {% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n' + message['content'] | trim + '<|eot_id|>'+'\n' %}
            {% if loop.index0 == 0 %}{% set content = bos_token + content %}
            {% endif %}
            {{content}}
        {% endfor %}
        {{'<|start_header_id|>assistant<|end_header_id|>'+'\n'}}
        """
        # delete scape characters
        jinja_chat_template = re.sub(r"(?<!')\n(?!')", '', jinja_chat_template).strip().replace('  ', '')

        with open(os.path.join(self.checkpoint['checkpoint_path'], 'tokenizer_config.json'), 'r+') as file:
            data = json.load(file)
            data['chat_template'] = jinja_chat_template
            file.seek(0)
            file.truncate()
            json.dump(data, file, indent=4)

    def check_chat_template(self, messages: List[Dict[str, str]]) -> str:
        return self.__byoc.check_chat_templates(messages, checkpoint_paths=self.checkpoint['checkpoint_path'])

    def set_padding_token(self) -> None:
        with open(os.path.join(self.checkpoint['checkpoint_path'], 'config.json'), 'r+') as file:
            data = json.load(file)
            data['pad_token_id'] = None
            file.seek(0)
            file.truncate()
            json.dump(data, file, indent=4)

    def get_model_params(self) -> None:
        checkpoint_config_params = self.__byoc.find_config_params(checkpoint_paths=self.checkpoint['checkpoint_path'])[
            0
        ]
        self.checkpoint.update(checkpoint_config_params)

    def upload_checkpoint(self):
        model_id = self.__byoc.upload_checkpoint(
            model_name=self.checkpoint['model_name'],
            checkpoint_path=self.checkpoint['checkpoint_path'],
            description=self.checkpoint['description'],
            publisher=self.checkpoint['publisher'],
            param_count=self.checkpoint['param_count'],
            model_arch=self.checkpoint['model_arch'],
            seq_length=self.checkpoint['seq_length'],
            vocab_size=self.checkpoint['vocab_size'],
            app_id=self.checkpoint['app_id'],
            retries=3
        )
        return model_id