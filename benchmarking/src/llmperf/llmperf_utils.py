import json
import time
from collections.abc import Iterable
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from transformers import AutoTokenizer

NUM_RNG_ATTEMPTS = 10  # Unlikely to be used in practice: prevents eternal WHILE-loops
FAMILY_MODEL_TYPE_IDENTIFIER = {
    'mistral': ['mistral'],
    'llama2': ['llama2'],
    'llama3': ['llama3'],
    'llama4': ['llama4'],
    'deepseek': ['deepseek'],
    'qwen': ['qwen', 'qwq'],
    'solar': ['solar'],
    'eeve': ['eeve'],
    'gpt-oss': ['gptoss'],
    'allam': ['allam'],
}
LVLM_IMAGE_PATHS = {
    'small': './imgs/vision_perf_eval-small.jpg',
    'medium': './imgs/vision_perf_eval-medium.jpg',
    'large': './imgs/vision_perf_eval-large.jpg',
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


def find_family_model_type(model_name: str) -> str:
    """Finds family model type

    Args:
        model_name (str): model name

    Returns:
        str: family model type
    """
    for family, models in FAMILY_MODEL_TYPE_IDENTIFIER.items():
        for model in models:
            if model in model_name.lower().replace('-', '').replace('v', ''):
                return family
    return 'llama2'


def get_tokenizer(model_name: str) -> Any:
    """Gets generic tokenizer according to model type

    Args:
        model_name (str): model name

    Returns:
        AutoTokenizer: generic HuggingFace tokenizer
    """
    # Using multiple sources for calling out model tokenizers without requesting access.
    # Ref: https://huggingface.co/NousResearch
    # Ref: https://huggingface.co/TheBloke
    # Ref: https://huggingface.co/unsloth
    # Ref: https://huggingface.co/deepseek-ai
    # Ref: https://huggingface.co/upstage
    # Ref: https://huggingface.co/yanolja
    # Ref: https://huggingface.co/QuantFactory

    family_model_type = find_family_model_type(model_name)

    if family_model_type == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained('TheBloke/Mistral-7B-Instruct-v0.2-AWQ') # type: ignore[no-untyped-call]
    elif family_model_type == 'llama3':
        if ('3.1' in model_name) or ('3p1' in model_name):
            if 'swallow' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained('tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3') # type: ignore[no-untyped-call]
            else:
                tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.1-8B-Instruct') # type: ignore[no-untyped-call]
        elif ('3.2' in model_name) or ('3p2' in model_name):
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct') # type: ignore[no-untyped-call]
        elif ('3.3' in model_name) or ('3p3' in model_name):
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.3-70B-Instruct') # type: ignore[no-untyped-call]
        else:
            tokenizer = AutoTokenizer.from_pretrained('unsloth/llama-3-8b-Instruct') # type: ignore[no-untyped-call]
    elif family_model_type == 'llama4':
        if 'maverick' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-4-Maverick-17B-128E-Instruct') # type: ignore[no-untyped-call]
        elif 'scout' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-4-Scout-17B-16E-Instruct') # type: ignore[no-untyped-call]
        else:
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-4-Scout-17B-16E-Instruct') # type: ignore[no-untyped-call]
    elif family_model_type == 'deepseek':
        if 'coder' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base') # type: ignore[no-untyped-call]
        elif 'r1':
            tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1') # type: ignore[no-untyped-call]
        else:
            tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-llm-7b-base') # type: ignore[no-untyped-call]
    elif family_model_type == 'qwen':
        if 'qwq' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('unsloth/QwQ-32B') # type: ignore[no-untyped-call]
        elif 'coder' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Qwen2.5-Coder-32B-Instruct') # type: ignore[no-untyped-call]
        else:
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Qwen2.5-72B-Instruct') # type: ignore[no-untyped-call]
    elif family_model_type == 'solar':
        tokenizer = AutoTokenizer.from_pretrained('upstage/SOLAR-10.7B-Instruct-v1.0') # type: ignore[no-untyped-call]
    elif family_model_type == 'eeve':
        tokenizer = AutoTokenizer.from_pretrained('yanolja/EEVE-Korean-10.8B-v1.0') # type: ignore[no-untyped-call]
    elif family_model_type == 'gpt-oss':
        tokenizer = AutoTokenizer.from_pretrained('openai/gpt-oss-120b') # type: ignore[no-untyped-call]
    elif family_model_type == 'allam':
        tokenizer = AutoTokenizer.from_pretrained('humain-ai/ALLaM-7B-Instruct-preview') # type: ignore[no-untyped-call]
    else:
        tokenizer = AutoTokenizer.from_pretrained('openai/gpt-oss-120b') # type: ignore[no-untyped-call]
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
