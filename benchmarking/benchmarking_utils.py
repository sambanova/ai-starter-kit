"""
Benchmarking utilities shared across the benchmarking kit.
"""

from typing import Any

from transformers import AutoTokenizer

# Model family identifiers for tokenizer selection
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


def find_family_model_type(model_name: str) -> str:
    """Finds family model type from model name.

    Args:
        model_name: Model name to identify

    Returns:
        Family model type identifier
    """
    for family, models in FAMILY_MODEL_TYPE_IDENTIFIER.items():
        for model in models:
            if model in model_name.lower().replace('-', '').replace('v', ''):
                return family
    return 'llama2'


def get_tokenizer(model_name: str) -> Any:
    """Gets generic tokenizer according to model type.

    Args:
        model_name: Model name

    Returns:
        AutoTokenizer: Generic HuggingFace tokenizer
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
        tokenizer = AutoTokenizer.from_pretrained('TheBloke/Mistral-7B-Instruct-v0.2-AWQ')  # type: ignore[no-untyped-call]
    elif family_model_type == 'llama3':
        if ('3.1' in model_name) or ('3p1' in model_name):
            if 'swallow' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained('tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3')  # type: ignore[no-untyped-call]
            else:
                tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.1-8B-Instruct')  # type: ignore[no-untyped-call]
        elif ('3.2' in model_name) or ('3p2' in model_name):
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')  # type: ignore[no-untyped-call]
        elif ('3.3' in model_name) or ('3p3' in model_name):
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.3-70B-Instruct')  # type: ignore[no-untyped-call]
        else:
            tokenizer = AutoTokenizer.from_pretrained('unsloth/llama-3-8b-Instruct')  # type: ignore[no-untyped-call]
    elif family_model_type == 'llama4':
        if 'maverick' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-4-Maverick-17B-128E-Instruct')  # type: ignore[no-untyped-call]
        elif 'scout' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-4-Scout-17B-16E-Instruct')  # type: ignore[no-untyped-call]
        else:
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-4-Scout-17B-16E-Instruct')  # type: ignore[no-untyped-call]
    elif family_model_type == 'deepseek':
        if 'coder' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base')  # type: ignore[no-untyped-call]
        elif 'r1':
            tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1')  # type: ignore[no-untyped-call]
        else:
            tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-llm-7b-base')  # type: ignore[no-untyped-call]
    elif family_model_type == 'qwen':
        if 'qwq' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('unsloth/QwQ-32B')  # type: ignore[no-untyped-call]
        elif 'coder' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Qwen2.5-Coder-32B-Instruct')  # type: ignore[no-untyped-call]
        else:
            tokenizer = AutoTokenizer.from_pretrained('unsloth/Qwen2.5-72B-Instruct')  # type: ignore[no-untyped-call]
    elif family_model_type == 'solar':
        tokenizer = AutoTokenizer.from_pretrained('upstage/SOLAR-10.7B-Instruct-v1.0')  # type: ignore[no-untyped-call]
    elif family_model_type == 'eeve':
        tokenizer = AutoTokenizer.from_pretrained('yanolja/EEVE-Korean-10.8B-v1.0')  # type: ignore[no-untyped-call]
    elif family_model_type == 'gpt-oss':
        tokenizer = AutoTokenizer.from_pretrained('openai/gpt-oss-120b')  # type: ignore[no-untyped-call]
    elif family_model_type == 'allam':
        tokenizer = AutoTokenizer.from_pretrained('humain-ai/ALLaM-7B-Instruct-preview')  # type: ignore[no-untyped-call]
    else:
        tokenizer = AutoTokenizer.from_pretrained('openai/gpt-oss-120b')  # type: ignore[no-untyped-call]
    return tokenizer


def get_tokenizer_model_name(model_name: str) -> str:
    """Gets the HuggingFace model name to use as tokenizer for a given model.

    Args:
        model_name: Model name to get tokenizer for

    Returns:
        HuggingFace model name to use as tokenizer
    """
    family_model_type = find_family_model_type(model_name)

    if family_model_type == 'mistral':
        return 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ'
    elif family_model_type == 'llama3':
        if ('3.1' in model_name) or ('3p1' in model_name):
            if 'swallow' in model_name.lower():
                return 'tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3'
            else:
                return 'unsloth/Llama-3.1-8B-Instruct'
        elif ('3.2' in model_name) or ('3p2' in model_name):
            return 'unsloth/Llama-3.2-1B-Instruct'
        elif ('3.3' in model_name) or ('3p3' in model_name):
            return 'unsloth/Llama-3.3-70B-Instruct'
        else:
            return 'unsloth/llama-3-8b-Instruct'
    elif family_model_type == 'llama4':
        if 'maverick' in model_name.lower():
            return 'unsloth/Llama-4-Maverick-17B-128E-Instruct'
        elif 'scout' in model_name.lower():
            return 'unsloth/Llama-4-Scout-17B-16E-Instruct'
        else:
            return 'unsloth/Llama-4-Scout-17B-16E-Instruct'
    elif family_model_type == 'deepseek':
        if 'coder' in model_name.lower():
            return 'deepseek-ai/deepseek-coder-1.3b-base'
        elif 'r1':
            return 'deepseek-ai/DeepSeek-R1'
        else:
            return 'deepseek-ai/deepseek-llm-7b-base'
    elif family_model_type == 'qwen':
        if 'qwq' in model_name.lower():
            return 'unsloth/QwQ-32B'
        elif 'coder' in model_name.lower():
            return 'unsloth/Qwen2.5-Coder-32B-Instruct'
        else:
            return 'unsloth/Qwen2.5-72B-Instruct'
    elif family_model_type == 'solar':
        return 'upstage/SOLAR-10.7B-Instruct-v1.0'
    elif family_model_type == 'eeve':
        return 'yanolja/EEVE-Korean-10.8B-v1.0'
    elif family_model_type == 'gpt-oss':
        return 'openai/gpt-oss-120b'
    elif family_model_type == 'allam':
        return 'humain-ai/ALLaM-7B-Instruct-preview'
    else:
        return 'openai/gpt-oss-120b'
