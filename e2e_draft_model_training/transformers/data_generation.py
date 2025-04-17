"""
A module to generate completions using a HuggingFace language model (LLM).

This script loads a YAML configuration file which specifies the paths for
the input dataset, output directory, run name, model name, and device to use.
It reads an input JSON file containing conversation records, constructs prompts
by combining system prompts and user instructions, and performs generation using
a loaded HuggingFace causal language model. The generated completions are then
assembled into conversation objects and saved to a JSON file.
"""

import json
import logging
import os
from typing import Any, Dict, List

import torch
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams  # type: ignore

load_dotenv()

logging.basicConfig(level=logging.INFO)


class HuggingFaceLLMCompletions:
    """
    A class to generate completions using a HuggingFace language model (LLM).

    It loads configuration parameters from a YAML file, instantiates the LLM and tokenizer,
    reads input data from a JSON file, generates completions based on prompts, and saves the
    resulting conversation outputs.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the HuggingFaceLLMCompletions instance by loading the configuration,
        the model, and the tokenizer.

        Parameters:
            config_path (str): Path to the YAML configuration file.
        """
        # Load YAML configuration.
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Read configuration settings.
        self.test_data: str = self.config.get('test_data', 'data/datasets/xxx.json')
        self.output_dir: str = self.config.get('output_dir', 'transformers/results')
        self.run_name: str = self.config.get('run_name', 'example_run')
        # This field might be used if the dataset uses a different field name for user text.
        self.dataset_text_field: str = self.config.get('dataset_text_field', 'training_text')
        model_name_or_path: str = self.config.get('inference_model', 'meta-llama/Llama-3.2-1B-Instruct')
        devices: str = self.config.get('devices', '0')
        # Determine device: use "cuda:<device>" if CUDA is available, otherwise fallback to CPU.
        self.device: str = f'cuda:{devices}' if torch.cuda.is_available() else 'cpu'
        logging.info('Using device: %s', self.device)
        logging.info('Loading model %s ...', model_name_or_path)

        # Load the model and tokenizer from HuggingFace.
        tensor_parallel_size = self.config.get('tensor_parallel_size', torch.cuda.device_count())
        self.model = LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', None)
        self.max_tokens = self.config.get('max_tokens', None)
        self.stop = self.config.get('stop')
        self.sampling_params = SamplingParams(temperature=self.top_p, max_tokens=self.max_tokens, stop=self.stop)

    def generate_completion(
        self, system_prompt: str, user_instruction: str, max_new_tokens: int = 256, temperature: float = 0.7
    ) -> str:
        """
        Generate a completion from the model given a system prompt and a user instruction.

        The prompt is constructed by concatenating the system prompt, a 'User:' marker with
        the user instruction, and an 'Assistant:' marker prompting the model to generate a reply.

        Parameters:
            system_prompt (str): The system prompt text providing context.
            user_instruction (str): The user instruction or question.
            max_new_tokens (int): Maximum number of tokens to generate (default: 256).
            temperature (float): Sampling temperature for diversity (default: 0.7).

        Returns:
            str: The generated assistant reply.
        """
        # Construct the prompt by combining system and user instructions.
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_instruction}]

        # Create the prompts
        prompts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Generate the outputs
        outputs = self.model.generate(prompts=prompts, sampling_params=self.sampling_params, use_tqdm=False)

        # Decode the full generated sequence.
        full_output: str = outputs[0].outputs[0].text

        return full_output

    def load_input_data(self) -> List[Dict[str, Any]]:
        """
        Load input data from a JSON file as specified by the configuration.

        The JSON file is expected to contain a list of records with keys such as "id",
        "system_prompt", and "instruction".

        Returns:
            List[Dict[str, Any]]: A list of records from the input dataset.
        """
        logging.info('Loading input data from %s ...', self.test_data)
        with open(self.test_data, 'r', encoding='utf-8') as f:
            data: List[Dict[str, Any]] = json.load(f)
        return data

    def run_inference(self) -> List[Dict[str, Any]]:
        """
        Run inference on the input data, generating completions for each record.

        For each record, a conversation is constructed that includes:
          - A system message from the "system_prompt" field.
          - A user message from the "instruction" field.
          - An assistant message generated by the model.

        Returns:
            List[Dict[str, Any]]: A list where each item is a dictionary with a
            "conversation" key mapping to the conversation list.
        """
        input_data: List[Dict[str, Any]] = self.load_input_data()
        results: List[Dict[str, Any]] = []

        for record in tqdm(input_data, desc='Processing records'):
            system_prompt: str = record.get('system_prompt', '')
            user_instruction: str = record.get('instruction', '')
            # Generate the assistant's output.
            assistant_output: str = self.generate_completion(system_prompt, user_instruction)
            conversation: List[Dict[str, str]] = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_instruction},
                {'role': 'assistant', 'content': assistant_output},
            ]
            results.append({'conversation': conversation})
        return results

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Save the inference results to a JSON file in the designated output directory.

        Parameters:
            results (List[Dict[str, Any]]): The list of conversation records to save.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        output_file: str = os.path.join(self.output_dir, f'{self.run_name}_completions.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        logging.info('Results saved to %s', output_file)


def main() -> None:
    """
    Main function to run the HuggingFace LLM completion generation workflow.

    It parses command-line arguments to obtain the path to the YAML configuration,
    instantiates the HuggingFaceLLMCompletions object, runs inference on the input data,
    and saves the results.
    """

    # Instantiate the completion generator with the provided configuration.
    llm_completer = HuggingFaceLLMCompletions('01_config_data_generation.yaml')
    # Run inference to generate completions.
    results = llm_completer.run_inference()
    # Save the output conversations to a JSON file.
    llm_completer.save_results(results)


if __name__ == '__main__':
    main()
