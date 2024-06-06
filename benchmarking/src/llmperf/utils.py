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


def randomly_sample_sonnet_lines_prompt(
    model_name: str,
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    expect_output_tokens: int = 150,
) -> Tuple[str, int]:
    """Generate a prompt that randomly samples lines from a the shakespeare sonnet at sonnet.txt.

    Args:
        model: name of the model
        prompt_tokens_mean: The mean tokens of the prompt to generate.
        prompt_tokens_stddev: The standard deviation of the tokens of the prompt to generate.
        expect_output_tokens: The number of tokens to expect in the output. This is used to
        determine the length of the prompt. The prompt will be generated such that the output
        will be approximately this many tokens.

    Note:
        tokens will be counted from the sonnet using the Llama tokenizer. Using one tokenizer
        ensures a fairer comparison across different LLMs. For example, if gpt 3.5 tokenizes
        a prompt in less tokens than Llama2, then this will be reflected in the results since
        they will be fed identical prompts.

    Returns:
        A tuple of the prompt and the length of the prompt.
    """

    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )

    get_token_length = lambda text: len(tokenizer.encode(text))

    mistral_model_type = "mistral"
    llama3_type_name = "llama-3"
    num_prompt_tokens = -1

    if mistral_model_type in model_name.lower():
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )
        prompt = "".join(["<s>" * (num_prompt_tokens - 1)])

    elif llama3_type_name in model_name.lower():
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )
        prompt = "".join(["<|eot_id|>" * (num_prompt_tokens - 1)])

    else:

        prompt = "[INST]Your task is to repeat the lines bellow 999999 times.\n'''"

        # Get a prompt length that is at least as long as the base prompt
        # n.b.: this means that we are not actually sampling from the Normal distribution but rather from a _censored_
        #   Normal distribution. This will not necessarily have the requested mean and STD, but the difference is negligible
        #   when the mean (minus an STD or 3) is much larger than the base prompt length (i.e.: about 122 tokens for Llama-3
        #   model and about 57 tokens otherwise)

        num_attempts = 0
        while num_prompt_tokens < get_token_length(prompt):
            num_prompt_tokens = sample_random_positive_int(
                prompt_tokens_mean, prompt_tokens_stddev
            )
            num_attempts += 1
            if num_attempts > NUM_RNG_ATTEMPTS:
                warnings.warn(
                    f"Could not generate a long enough prompt after {NUM_RNG_ATTEMPTS} attempts. \n"
                    f"Consider increasing prompt_tokens_mean (currently: {prompt_tokens_mean}). "
                    f"Returning the default prompt instead."
                )
                num_prompt_tokens = get_token_length(prompt)
                break

        sonnet_path = pathlib.Path(__file__).parent.resolve() / "sonnet.txt"
        with open(sonnet_path, "r") as f:
            sonnet_lines = f.readlines()
        random.shuffle(sonnet_lines)

        # Because there isn't a 1:1 correspondence between text characters and tokens,
        # it could be impossible to get exactly the number of input tokens requested.
        # The following is guaranteed to return, will usually get things exactly correct,
        # and very occasionally might generate a few tokens more than requested.
        while num_prompt_tokens > get_token_length(prompt):
            for line in sonnet_lines:
                new_prompt = prompt + line
                new_tokens = get_token_length(new_prompt)
                if num_prompt_tokens > new_tokens:
                    prompt = new_prompt
                elif num_prompt_tokens == new_tokens:
                    prompt = new_prompt
                    # return (prompt, num_prompt_tokens)
                    break
                else:
                    for character in line:
                        prompt += character
                        if num_prompt_tokens <= get_token_length(prompt):
                            # return (prompt, num_prompt_tokens)
                            break

        prompt += "'''\nAnswer:[/INST]"

        # print(prompt, flush=True)
        # print(f"tokens_prompt: {get_token_length(prompt)}", flush=True)
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
                f"Returning 1 instead."
            )
            return 1
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
