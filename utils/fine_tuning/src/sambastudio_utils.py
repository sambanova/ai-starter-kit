import os
import sys
import logging
import subprocess
from typing import Union, Optional
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
fine_tuning_dir = os.path.abspath(os.path.join(current_dir, '..'))
utils_dir = os.path.abspath(os.path.join(fine_tuning_dir, '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)
load_dotenv(os.path.join(repo_dir, '.env'), override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)

def setup_generative_data_prep() -> None:
    """
    Configures the generative data preparation module.

    This function initializes and updates the generative data preparation git submodule,
    then installs the generative data preparation module using pip.
    If any errors occur during the process, an exception is raised.

    Returns:
        None
    """

    generative_data_prep_dir = os.path.join(fine_tuning_dir, 'generative_data_prep')
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
        raise Exception(f"Error executing command: {''.join(command)}", response.stderr)

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
        raise Exception(f"Error executing command: {''.join(command)}", response.stderr)
    logging.info('Gen data preparation module set up successfully')


def run_generative_data_prep_pipeline(
    input_path: str, output_path: str, tokenizer: str, max_seq_length: int, shuffle='on_RAM'
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
    - shuffle (str, optional): The shuffle mode for the data. Default is 'on_RAM'.

    Returns:
    - None
    """
    command = [
        'python',
        '-m',
        'generative_data_prep',
        'pipeline',
        '--input_file_path',
        input_path,
        '--output_path',
        output_path,
        '--pretrained_tokenizer',
        tokenizer,
        '--max_seq_length',
        str(max_seq_length),
        '--shuffle',
        shuffle,
        '--keep_split_jsonls',
    ]
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
        raise Exception(f"Error executing command: {''.join(command)}", response.stderr)
    logging.info('Gen data preparation pipeline ran successfully')


def merge_jsonl_files(input_paths: Union[list, str], output_path: Optional[str] = None) -> str:
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
    input_files: Union[list, str], 
    output_path: str, 
    tokenizer: str, 
    max_seq_length: int
    ) -> str:
    """
    Merges input JSONL files, sets up and runs the generative data preparation pipeline.
    
    Args:
    - input_files (Union[list, str]): A list of input JSONL file paths or a single JSONL file path.
    - output_path (str): The path to the output directory where the processed files will be saved.
    - tokenizer (str): The name of the pretrained tokenizer to be used for tokenization.
    - max_seq_length (int): The maximum sequence length for the input data.

    Returns:
    - str: The path of the output directory where the processed dataset is stored.
    """
    merged_jsonl_file = merge_jsonl_files(input_files)
    setup_generative_data_prep()
    run_generative_data_prep_pipeline(merged_jsonl_file, output_path, tokenizer, max_seq_length)
    return output_path
