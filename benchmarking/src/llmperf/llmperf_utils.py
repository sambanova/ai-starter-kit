import json
import time
from collections.abc import Iterable
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from transformers import AutoTokenizer

SAMBANOVA_URL = 'https://api.sambanova.ai/v1/chat/completions'
NUM_RNG_ATTEMPTS = 10  # Unlikely to be used in practice: prevents eternal WHILE-loops
MODEL_TYPE_IDENTIFIER = {
    'mistral': 'mistral',
    'llama3': 'llama3',
    'deepseek': 'deepseek',
    'solar': 'solar',
    'eeve': 'eeve',
    'llama2': 'llama2',
}


class LLMPerfResults:
    """Class with LLM Performance results"""

    def __init__(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata['timestamp'] = self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Updates and flattens dictionary

        Returns:
            dict: transformed dictionary
        """
        data = {
            'name': self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self) -> str:
        """Transforms dictionary to json string

        Returns:
            str: json string
        """
        data = self.to_dict()
        return json.dumps(data)


def get_tokenizer(model_name: str) -> AutoTokenizer:
    """Gets generic tokenizer according to model type

    Args:
        model_name (str): model name

    Returns:
        AutoTokenizer: generic HuggingFace tokenizer
    """
    # Using NousrResearch for calling out model tokenizers without requesting access.
    # Ref: https://huggingface.co/NousResearch
    # Ref: https://huggingface.co/TheBloke
    # Ref: https://huggingface.co/unsloth
    # Ref: https://huggingface.co/deepseek-ai
    # Ref: https://huggingface.co/upstage
    # Ref: https://huggingface.co/yanolja

    if MODEL_TYPE_IDENTIFIER['mistral'] in model_name.lower().replace('-', ''):
        tokenizer = AutoTokenizer.from_pretrained('TheBloke/Mistral-7B-Instruct-v0.2-AWQ')
    elif MODEL_TYPE_IDENTIFIER['llama3'] in model_name.lower().replace('-', ''):
        tokenizer = AutoTokenizer.from_pretrained('unsloth/llama-3-8b-Instruct')
    elif MODEL_TYPE_IDENTIFIER['deepseek'] in model_name.lower().replace('-', ''):
        if 'coder' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base')
        else:
            tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-llm-7b-base')
    elif MODEL_TYPE_IDENTIFIER['solar'] in model_name.lower().replace('-', ''):
        tokenizer = AutoTokenizer.from_pretrained('upstage/SOLAR-10.7B-Instruct-v1.0')
    elif MODEL_TYPE_IDENTIFIER['eeve'] in model_name.lower().replace('-', ''):
        tokenizer = AutoTokenizer.from_pretrained('yanolja/EEVE-Korean-10.8B-v1.0')
    else:
        tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf')
    return tokenizer


def flatten(item: Union[Iterable[Union[str, Iterable[str]]], str]) -> Generator[str, None, None]:
    """Flattens an iterable"""
    for sub_item in item:
        if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
            yield from flatten(sub_item)
        else:
            yield sub_item


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flattens dictionary

    Args:
        d (dict): input dictionary
        parent_key (str, optional): parent key. Defaults to "".
        sep (str, optional): separator. Defaults to "_".

    Returns:
        dict: output flat dictionary
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
