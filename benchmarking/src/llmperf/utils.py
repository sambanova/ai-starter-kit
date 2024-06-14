import json
import math
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, Tuple
import warnings

from transformers import LlamaTokenizerFast

RESULTS_VERSION = "2023-08-31"
NUM_RNG_ATTEMPTS = 10  # Unlikely to be used in practice: prevents eternal WHILE-loops
MODEL_TYPE_IDENTIFIER = {"mistral": "mistral", "llama3": "llama-3"}


def get_tokenizer(model_name: str) -> LlamaTokenizerFast:
    """Gets generic tokenizer according to model type

    Args:
        model_name (str): model name

    Returns:
        LlamaTokenizerFast: generic HuggingFace tokenizer
    """

    # Using NousrResearch for calling out model tokenizers without requesting access. Ref: https://huggingface.co/NousResearch
    if MODEL_TYPE_IDENTIFIER["mistral"] in model_name.lower():
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "NousResearch/Hermes-2-Pro-Mistral-7B"
        )
    elif MODEL_TYPE_IDENTIFIER["llama3"] in model_name.lower():
        tokenizer = LlamaTokenizerFast.from_pretrained("NousResearch/Meta-Llama-3-8B")
    else:
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "NousResearch/Llama-2-7b-chat-hf"
        )
    return tokenizer


class LLMPerfResults:
    """Class with LLM Performance results"""

    def __init__(
        self,
        name: str,
        metadata: Dict[str, Any] = None,
    ):
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata["timestamp"] = self.timestamp
        self.version = RESULTS_VERSION

    def to_dict(self) -> dict:
        """Updates and flattens dictionary

        Returns:
            dict: transformed dictionary
        """
        data = {
            "version": self.version,
            "name": self.name,
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


def upload_to_s3(results_path: str, s3_path: str) -> None:
    """Upload the results to s3.

    Args:
        results_path: The path to the results file.
        s3_path: The s3 path to upload the results to.
    """

    command = ["aws", "s3", "sync", results_path, f"{s3_path}/"]
    result = subprocess.run(command)
    if result.returncode == 0:
        print("Files uploaded successfully!")
    else:
        print("An error occurred:")
        print(result.stderr)


def build_prompt_user(
    prompt_user_template: str,
    prompt_user_template_tokens: int,
    tokens_to_complement: int,
) -> str:
    """Builds a prompt based on template and tokens needed

    Args:
        prompt_user_template (str): prompt template
        prompt_user_template_tokens (int): prompt template's number of tokens
        tokens_to_complement (int): number of tokens needed

    Returns:
        str: prompt of user
    """

    prompt_user = "".join(
        [prompt_user_template * (tokens_to_complement // prompt_user_template_tokens)]
    )
    prompt_user += prompt_user_template[
        : (tokens_to_complement % prompt_user_template_tokens)
    ]

    return prompt_user


def build_prompt(
    model_name: str,
    prompt_tokens_mean: int,
    prompt_tokens_stddev: int,
) -> Tuple[str, int]:
    """Generate a prompt that randomly samples lines from a the shakespeare sonnet at sonnet.txt.

    Args:
        model_name (str): name of the model
        prompt_tokens_mean (int): The mean tokens of the prompt to generate.
        prompt_tokens_stddev (int): The standard deviation of the tokens of the prompt to generate.
        tokenizer (LlamaTokenizerFast): tokenizer instantiated for specific model

    Returns:
        Tuple[str, int]: A tuple of the prompt and the length of the prompt.
    """

    # Get tokenizer
    tokenizer = get_tokenizer(model_name)
    get_token_length = lambda text: len(tokenizer.encode(text))

    # Define prompt template
    prompt_user_template = "Create a movie script of the whole Star Wars movie with details. Describe how every character felt, include environment details and onomatopoeias."
    prompt_user_template_tokens = get_token_length(prompt_user_template)

    # Generate random number of input tokens
    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )

    # Prompt for Mistral models
    if MODEL_TYPE_IDENTIFIER["mistral"] in model_name.lower():

        tokens_to_complement = num_prompt_tokens

        prompt_user = build_prompt_user(
            prompt_user_template, prompt_user_template_tokens, tokens_to_complement
        )

        prompt = "[INST]" + prompt_user + "[/INST]"

    # Prompt for Llama3 models
    elif MODEL_TYPE_IDENTIFIER["llama3"] in model_name.lower():

        prompt_system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant that generates movie scripts<|eot_id|>"
        prompt_user = "<|start_header_id|>user<|end_header_id|>Create a movie script of the whole Star Wars movie with details. Describe how every character felt, include environment details and onomatopoeias.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        tokens_to_complement = num_prompt_tokens - get_token_length(prompt_system)
        prompt_user = build_prompt_user(
            prompt_user_template, prompt_user_template_tokens, tokens_to_complement
        )
        prompt = (
            prompt_system
            + "<|start_header_id|>user<|end_header_id|>"
            + prompt_user
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

    # Prompt for Llama2 and other models
    else:

        prompt_system = "[INST]<<SYS>>You are a helpful assistant that generates movie scripts<</SYS>>"
        tokens_to_complement = num_prompt_tokens - get_token_length(prompt_system)
        prompt_user = build_prompt_user(
            prompt_user_template, prompt_user_template_tokens, tokens_to_complement
        )
        prompt = prompt_system + prompt_user + "[/INST]"

    return (prompt, num_prompt_tokens)


def sample_random_positive_int(mean: int, stddev: int) -> int:
    """Sample random numbers from a gaussian distribution until a positive number is sampled.

    Args:
        mean: The mean of the gaussian distribution to sample from.
        stddev: The standard deviation of the gaussian distribution to sample from.

    Returns:
        A random positive integer sampled from the gaussian distribution.
    """

    ret = -1
    num_attempts = 0
    while ret <= 0:
        ret = int(random.gauss(mean, stddev))
        num_attempts += 1
        if num_attempts > NUM_RNG_ATTEMPTS:
            warnings.warn(
                f"Could not generate a random, positive integer after {NUM_RNG_ATTEMPTS} attempts. \n"
                f"Check your choices for mean (currently: {mean}) and stddev (currently: {stddev}). "
                f"Returning current value ({mean}) instead."
            )
            return mean
    return ret


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flattens dictionary

    Args:
        d (dict): input dictionary
        parent_key (str, optional): parent key. Defaults to "".
        sep (str, optional): separator. Defaults to "_".

    Returns:
        dict: output flat dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
