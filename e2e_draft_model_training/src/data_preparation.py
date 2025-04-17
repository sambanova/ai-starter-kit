"""
This module prepares one or more datasets for training a draft model.
It should be used in conjunction with the `02_config_data_preparation.yaml` configuration.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import datasets  # type: ignore
import yaml
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logging.basicConfig(level=logging.INFO)


def tokenize_dialog(item: Dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    """
    Tokenizes a dialog entry using the provided tokenizer.

    Constructs the dialog from the 'conversation' key if available.
    Otherwise, it creates a conversation using the 'prompt' and 'completion' keys.
    This dialog is then processed through the tokenizer's chat template API.

    Args:
        item (Dict[str, Any]): A dictionary containing dialog information. The expected keys are:
                               - 'conversation': The conversation list (if available); or,
                               - 'prompt' and 'completion': Used to create a conversation.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to process the dialog.

    Returns:
        Dict[str, Any]: A dictionary with key 'training_text' containing the tokenized dialog.
    """
    if 'conversation' not in item:
        dialog = [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['completion']}]
    else:
        dialog = item['conversation']

    # Apply the chat template without tokenizing individual utterances or adding generation prompts.
    tokenized_input = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=False)
    return {'training_text': tokenized_input}


def get_dataset(file_paths: List[str], tokenizer: PreTrainedTokenizerBase, output_path: Optional[str] = None) -> None:
    """
    Loads, tokenizes, and saves a dataset from provided JSON/JSONL files.

    Processes the dataset differently depending on whether there are multiple files.
      - For multiple files: Loads the dataset with streaming disabled, tokenizes each entry,
        and writes the results as newline-separated JSON objects.
      - For a single file: Loads the dataset (with streaming if '.jsonl'), tokenizes each entry,
        and saves the full list of tokenized entries as a JSON array.

    Args:
        file_paths (List[str]): A list of paths for input data files.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for processing dialogs.
        output_path (Optional[str]): The path to store the tokenized dataset.
            When processing multiple files, this must be provided.
            For a single file, a default name based on the input file is derived if not provided.
    """
    logging.info('File paths:' + (', ').join(file_paths))

    if len(file_paths) > 1:
        dataset = datasets.load_dataset('json', data_files=file_paths, split='train', streaming=False)
        dataset = dataset.map(lambda x: tokenize_dialog(x, tokenizer))
        if output_path is None:
            raise ValueError('Output path must be provided when processing multiple files.')

        # Write each tokenized entry on a separate line.
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    else:
        file_path = file_paths[0]
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        # Load dataset based on file type.
        if file_extension == '.jsonl':
            dataset = datasets.load_dataset('json', data_files=file_path, split='train', streaming=True)
        else:
            dataset = datasets.load_dataset('json', data_files=file_path)['train']

        dataset = dataset.map(lambda x: tokenize_dialog(x, tokenizer))

        # Remove all columns except for 'training_text'.
        columns_to_keep = [col for col in dataset.column_names if col == 'training_text']
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
        data_list = dataset.to_list()

        if output_path is None:
            base_name = file_path.rsplit('.json', 1)[0].rsplit('.jsonl', 1)[0]
            output_path = base_name + '_templated.json'

        # Save the complete tokenized dataset as a JSON array.
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)

    logging.info(f'Processed file saved at: {output_path}')


if __name__ == '__main__':
    """
    Main entry point for processing the dialog dataset.

    The script reads configuration settings from a YAML file named 
    '01_config_data_preparation', which must contain at least the following keys:
      - file_path: Path to the input JSON/JSONL file or directory.
      - model: The pretrained model identifier for tokenizer initialization.
      - output_path (optional): Path where the processed file will be saved.
    
    If 'file_path' refers to a directory, all files ending with '.jsonl' within that directory are processed.
    Otherwise, it assumes that multiple paths are comma-separated.
    """
    # Read the configuration parameters from a YAML configuration file.
    with open('02_config_data_preparation.yaml', 'r', encoding='utf-8') as file:
        config: Dict[str, Any] = yaml.safe_load(file)

    file_path: str = config['file_path']
    model: str = config['model']
    output_path: Optional[str] = config.get('output_path', None)
    if output_path is None:
        # Remove the '.json' or '.jsonl' extension and append '_templated.json'.
        output_path = file_path.rsplit('.json', 1)[0].rsplit('.jsonl', 1)[0] + '_templated.json'

    # Initialize the tokenizer from the pretrained model.
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Determine if the provided file path is a directory or a comma separated list.
    if os.path.isdir(file_path):
        file_paths: List[str] = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.jsonl')]
    else:
        file_paths = file_path.split(',')

    # Get the dataset
    get_dataset(file_paths, tokenizer, output_path)
