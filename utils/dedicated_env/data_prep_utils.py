import importlib.util
import logging
import os
import subprocess
import sys
from typing import Any, List, Optional, Union

from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)
load_dotenv(os.path.join(repo_dir, '.env'), override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)


def setup_generative_data_prep(force_gen_data_prep_install: bool = False) -> None:
    """
    Configures the generative data preparation module.

    This function initializes and updates the generative data preparation git submodule,
    then installs the generative data preparation module using pip.
    If any errors occur during the process, an exception is raised.

    Args:
    - force_gen_data_prep_install (bool): whether or not to force reinstallation

    Returns:
        None
    """

    generative_data_prep_dir = os.path.join(current_dir, 'generative_data_prep')

    # Check if the package is already installed
    package_spec = importlib.util.find_spec('generative_data_prep')
    if package_spec is not None and not force_gen_data_prep_install:
        logging.info('generative_data_prep is already installed. Skipping installation.')

    else:
        # init and update generative data preparation submodule
        command = ['git', 'submodule', 'update', '--recursive', '--init', generative_data_prep_dir]
        response = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        for line in response.stdout.split('\n'):
            if len(line) > 0:
                logging.info(line.strip())
        for line in response.stderr.split('\n'):
            if len(line) > 0:
                logging.error(line.strip())
        if response.returncode != 0:
            raise Exception(f'Error executing command: {"".join(command)}', response.stderr)

        # install generative data preparation module
        command = ['pip', 'install', generative_data_prep_dir]
        response = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        for line in response.stdout.split('\n'):
            if len(line) > 0:
                logging.info(line.strip())
        for line in response.stderr.split('\n'):
            if len(line) > 0:
                logging.error(line.strip())
        if response.returncode != 0:
            raise Exception(f'Error executing command: {"".join(command)}', response.stderr)
        logging.info('Gen data preparation module set up successfully')


def run_generative_data_prep_pipeline(
    input_path: str,
    output_path: str,
    tokenizer: str,
    max_seq_length: int,
    shuffle: str,
    input_packing_config: str,
    prompt_keyword: str,
    completion_keyword: str,
    num_training_splits: int,
    apply_chat_template: bool,
) -> None:
    """
    Runs the generative data preparation pipeline to process a JSONL file.

    This function uses the generative data preparation module to convert a JSONL dataset file into a format suitable for
    fine-tuning in sambastudio. The function executes the pipeline command with the provided parameters.

    Args:
    - input_path (str): The path to the input JSONL file.
    - output_path (str): The path to the output directory where the processed files will be saved.
    - tokenizer (str): The name of the pretrained tokenizer to be used for tokenization.
    - max_seq_length (int): The maximum sequence length for the input data.
    - shuffle (str): The shuffle mode for the data.
    - input_packing_config (str): method of placing text into sequences.
    - prompt_keyword (str):  prompt keyword.
    - completion_keyword (str):  completion keyword.
    - num_training_splits (int): number of training splits to generate
    - apply_chat_template (bool): Whether to tokenize the data using the tokenizer_config.json chat_template.

    see more: https://github.com/sambanova/generative_data_prep?tab=readme-ov-file#flags

    Returns:
    - None
    """
    command = [
        'python',
        '-m',
        'generative_data_prep',
        'pipeline',
        '--input_path',
        input_path,
        '--output_path',
        output_path,
        '--pretrained_tokenizer',
        tokenizer,
        '--max_seq_length',
        str(max_seq_length),
        '--shuffle',
        shuffle,
        '--input_packing_config',
        input_packing_config,
        '--prompt_keyword',
        prompt_keyword,
        '--completion_keyword',
        completion_keyword,
        '--num_training_splits',
        str(num_training_splits),
        '--keep_split_jsonls',
    ]
    if apply_chat_template:
        command.append('--apply_chat_template')

    response = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    for line in response.stdout.split('\n'):
        if len(line) > 0:
            logging.info(line.strip())
    for line in response.stderr.split('\n'):
        if len(line) > 0:
            logging.error(line.strip())
    if response.returncode != 0:
        raise Exception(f'Error executing command: {"".join(command)}', response.stderr)
    logging.info('Gen data preparation pipeline ran successfully')


def merge_jsonl_files(input_paths: Union[List[Any], str], output_path: Optional[str] = None) -> str:
    """
    Merges multiple JSONL files into a single JSONL file.

    This function takes a list of input JSONL file paths or a single JSONL file path,
    merges them into a single JSONL file, and saves it to the specified output path.
    If no output path is provided, the function will generate a default output path based in first input element.

    Args:
    - input_paths (Union[list, str]): A list of input JSONL file paths or a single JSONL file path.
    - output_path (Optional[str], optional): The output path for the merged JSONL file. Defaults to None.

    Returns:
    - str: The path of the merged JSONL file.
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    for path in input_paths:
        if not path.endswith('.jsonl'):
            raise ValueError(f'File {path} is not a JSONL file.')

    if output_path is None:
        output_path = ''.join(input_paths[0].split('.')[:-1]) + '_merged.jsonl'

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w') as outfile:
        for path in input_paths:
            with open(path, 'r') as input_file:
                for line in input_file:
                    outfile.write(line)

    logging.info(f'input jsonl files merged into {output_path}')
    return output_path


def gen_data_prep_pipeline(
    input_files: Union[list[Any], str],
    output_path: str,
    tokenizer: str,
    max_seq_length: int,
    shuffle: str = 'on_RAM',
    input_packing_config: str = 'full',
    prompt_keyword: str = 'prompt',
    completion_keyword: str = 'completion',
    num_training_splits: int = 32,
    apply_chat_template: bool = False,
    force_gen_data_prep_install: bool = False,
) -> str:
    """
    checks ig dataset is not already created, Merges input JSONL files,
    then sets up and runs the generative data preparation pipeline.

    Args:
    - input_files (Union[list, str]): A list of input JSONL file paths or a single JSONL file path.
    - output_path (str): The path to the output directory where the processed files will be saved.
    - tokenizer (str): The name of the pretrained tokenizer to be used for tokenization.
    - max_seq_length (int): The maximum sequence length for the input data.
    - shuffle (str): The shuffle mode for the data. Default is 'on_RAM'.
    - input_packing_config (str): method of placing text into sequences. Default is full
    - prompt_keyword (str):  prompt keyword. Default is 'prompt'
    - completion_keyword (str):  completion keyword. Default is 'completion'
    - num_training_splits (int): number of training splits to generate. Default is 32
    - apply_chat_template (bool): Whether to tokenize the data using the
        tokenizer_config.json chat_template. Default is False
    -force_gen_data_prep_install (bool): force the re-installation of generative data preparation
        repository package. Default is false

    see more: https://github.com/sambanova/generative_data_prep?tab=readme-ov-file#flags

    Returns:
    - str: The path of the output directory where the processed dataset is stored.
    """

    merged_jsonl_file = merge_jsonl_files(input_files)
    setup_generative_data_prep(force_gen_data_prep_install)

    if os.path.exists(output_path) and os.listdir(output_path):
        logging.warning(f'Training dataset already exists at {output_path} and will not be regenerated.')
        return output_path

    run_generative_data_prep_pipeline(
        merged_jsonl_file,
        output_path,
        tokenizer,
        max_seq_length,
        shuffle,
        input_packing_config,
        prompt_keyword,
        completion_keyword,
        num_training_splits,
        apply_chat_template,
    )
    return output_path
