import os
import re
import sys
import time
import glob
import shutil
import argparse
import subprocess
import pandas as pd
from collections import defaultdict
import logging
import yaml
from general_model_testing_object import GeneralModelTestObject

def setup_logging(config):
    """Set up logging configuration."""
    logging.basicConfig(level=getattr(logging, config['log_level']), 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    file_handler = logging.FileHandler(config['log_file'])
    file_handler.setLevel(getattr(logging, config['log_level']))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()

def load_config(config_file='config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)

import os

def ensure_directories_exist(config):
    """
    Ensure that the necessary directories exist, creating them if they don't.
    
    Args:
        config (dict): The configuration dictionary containing path information.
    """
    directories = [
        os.path.dirname(config['master_menu_path']),
        config['artifacts_path']
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
            except OSError as e:
                logger.error(f"Error creating directory {directory}: {e}")
                sys.exit(1)
        else:
            logger.info(f"Directory already exists: {directory}")

def snapi_byoc_import_model(modelname, ckpt_path, appname, modelarch, paracnt, sslen, vocabsize, source="LOCAL"):
    """
    Import a model using SNAPI BYOC.
    
    Args:
        modelname (str): Name of the model
        ckpt_path (str): Path to the checkpoint
        appname (str): Name of the application
        modelarch (str): Model architecture
        paracnt (str): Parameter count
        sslen (str): Sequence length
        vocabsize (str): Vocabulary size
        source (str): Source of the model (default: "LOCAL")
    
    Returns:
        tuple: (model_added_successfully, time_taken, modelname, uuid)
    """
    modelname = f"qa-{modelname}-{source}-byoc-test"
    command = (
        f"snapi import model create -m \"{modelname}\" -a \"{appname}\" -t {source} "
        f"-s {ckpt_path} -p \"sns\" -d \"QA test on byoc\" -ma {modelarch} "
        f"-c {paracnt} -l {sslen} -vs {vocabsize} -ni"
    )
    logger.info(f"SNAPI BYOC command: {command}")

    try:
        res = subprocess.run(command, input='yes', encoding='ascii', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    except subprocess.TimeoutExpired:
        logger.error("SNAPI BYOC import command timed out after 10 minutes")
        return False, None, modelname, None

    if res.returncode != 0:
        logger.error(f"SNAPI BYOC import command failed with error: {res.stderr}")
        return False, None, modelname, None

    content = res.stdout

    model_added_pattern = r"Model added successfully"
    time_pattern = r"Time taken to upload the model:\s([\d\.]+)\sseconds"
    uuid_pattern = r"snapi import model status -m ([a-f0-9\-]+)"

    model_added_successfully = re.search(model_added_pattern, content) is not None
    time_taken_match = re.search(time_pattern, content)
    time_taken = float(time_taken_match.group(1)) if time_taken_match else None
    uuid_match = re.search(uuid_pattern, content)
    uuid = uuid_match.group(1) if uuid_match else None

    return model_added_successfully, time_taken, modelname, uuid

def check_byoc_checkpoint_status(modelid):
    """
    Check the status of a BYOC checkpoint.
    
    Args:
        modelid (str): ID of the model
    
    Returns:
        str: Status of the checkpoint
    """
    command = f"snapi import model status -m {modelid}"

    try:
        res = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, text=True)
    except subprocess.TimeoutExpired:
        logger.error("Checkpoint status check timed out after 5 minutes")
        return None

    if res.returncode != 0:
        logger.error(f"Checkpoint status check failed with error: {res.stderr}")
        return None

    for line in res.stdout.splitlines():
        if 'Status' in line:
            return line.split(': ')[-1].strip()

    logger.warning("Status not found in command output")
    return None

def create_coe_one_expert(modelname):
    """
    Create a Composite of Experts (COE) with one expert.
    
    Args:
        modelname (str): Name of the model
    
    Returns:
        tuple: (coename, status)
    """
    coename = f"{modelname}-coe"
    command = f"snapi model add-composite -n {coename} -e {modelname}"

    try:
        res = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, text=True)
    except subprocess.TimeoutExpired:
        logger.error("COE creation command timed out after 5 minutes")
        return coename, False

    if res.returncode != 0:
        logger.error(f"COE creation failed with error: {res.stderr}")
        return coename, False

    status = 'Successfully' in res.stdout or 'successfully' in res.stdout
    return coename, status

def deploy_one_endpoint(modelname):
    """
    Deploy one endpoint for a given model.
    
    Args:
        modelname (str): Name of the model
    
    Returns:
        tuple: (project_id, endpoint_id, api_key, endpoint_name)
    """
    handler = GeneralModelTestObject("snapi_LLM_1", "SN40L")
    res, api_key, url = handler.model_endpoint(modelname)

    if not res or not url:
        logger.error("Failed to deploy endpoint")
        return None, None, None, None

    try:
        tmp = url.split('/')
        project_id = tmp[-2]
        endpoint_id = tmp[-1]
    except IndexError:
        logger.error(f"Failed to parse URL: {url}")
        return None, None, None, None

    return project_id, endpoint_id, api_key, f"{modelname.lower()}-1ins-endpoint-sn40l"

def delete_one_endpoint(endpoint_name):
    """
    Delete a deployed endpoint.
    
    Args:
        endpoint_name (str): Name of the endpoint to delete
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    command = f"snapi endpoint delete -p snapi_LLM_1 -e {endpoint_name}"

    try:
        res = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, text=True)
    except subprocess.TimeoutExpired:
        logger.error("Endpoint deletion command timed out after 5 minutes")
        return False

    if res.returncode != 0:
        logger.error(f"Endpoint deletion failed with error: {res.stderr}")
        return False

    return 'Successfully' in res.stdout or 'successfully' in res.stdout

def delete_one_coe(coename):
    """
    Delete a Composite of Experts (COE).
    
    Args:
        coename (str): Name of the COE to delete
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    command = f"snapi model remove -m {coename}"

    try:
        res = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, text=True)
    except subprocess.TimeoutExpired:
        logger.error("COE deletion command timed out after 5 minutes")
        return False

    if res.returncode != 0:
        logger.error(f"COE deletion failed with error: {res.stderr}")
        return False

    return 'Successfully' in res.stdout or 'successfully' in res.stdout

def delete_byoc_model(modelname):
    """
    Delete a BYOC model.
    
    Args:
        modelname (str): Name of the model to delete
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    command = f"snapi import model delete -m {modelname}"

    try:
        res = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, text=True)
    except subprocess.TimeoutExpired:
        logger.error("BYOC model deletion command timed out after 5 minutes")
        return False

    if res.returncode != 0:
        logger.error(f"BYOC model deletion failed with error: {res.stderr}")
        return False

    return 'Successfully' in res.stdout or 'successfully' in res.stdout

def run_turbo_coe_benchmark(project_id, endpoint_id, api_key, expert, coename, config):
    script_name = os.path.basename(config['benchmark_script_path'])
    bash_command = (
        f"./{script_name} "
        f"--streamURL {config['benchmark_api_url']}/{project_id}/{endpoint_id} "
        f"--apiKEY {api_key} --experts {expert} --app {coename} "
        f"--promptDir {config['benchmark_prompt_dir']} "
        f"--promptLongSuffix {config['benchmark_prompt_long_suffix']} "
        f"--promptShortSuffix {config['benchmark_prompt_short_suffix']} "
        f"--promptLongSuperSuffix {config['benchmark_prompt_long_super_suffix']}"
    )

    logger.info(f"Running benchmark command: {bash_command}")
    current_dir = os.getcwd()
    os.chdir(script_directory)

    try:
        result = subprocess.run(bash_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=config['job_completion_timeout'], text=True)
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark failed with error: {e.stderr}")
    except subprocess.TimeoutExpired:
        logger.error(f"Benchmark timed out after {config['job_completion_timeout']} seconds")
    finally:
        os.chdir(current_dir)

def read_master_menu(filepath):
    """
    Read the master menu CSV file and return a dictionary of model configurations.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        dict: Dictionary of model configurations
    """
    if not os.path.exists(filepath):
        logger.error(f"Master menu file not found: {filepath}")
        logger.info(f"Please ensure the file exists at {filepath}")
        return {}

    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        logger.error(f"Master menu file is empty: {filepath}")
        return {}


    model_dict = defaultdict(dict)
    for _, row in df.iterrows():
        model_name = row.get('model_name')
        if pd.notna(model_name):
            model_details = {}
            for column, value in row.items():
                if pd.notna(value):
                    value = str(value).strip()
                    if column == 'Param Count':
                        model_details[column] = f"{int(float(value))}b"
                    elif column == '3-graph':
                        model_details[column] = [int(float(x.replace('k', '')) * 1024) if 'k' in x else int(float(x)) for x in value.split(',')]
                    elif column == 'Batch Sizes':
                        model_details[column] = [int(float(x)) for x in value.split(',')]
                    else:
                        model_details[column] = value

            model_dict[model_name] = model_details

    for model in list(model_dict):
        if 'model_name' in model_dict[model]:
            del model_dict[model]['model_name']
        if 'BYOC candidate' not in model_dict[model]:
            model_dict[model]['BYOC candidate'] = None

    return model_dict

def main_byoc(config_file='config.yaml'):
    config = load_config(config_file)

    # Ensure necessary directories exist
    ensure_directories_exist(config)

    model_dict = read_master_menu(config['master_menu_path'])
    model_tester = GeneralModelTestObject(config['project'], config['chiparch'])

    for modelname, model_config in model_dict.items():
        logger.info(f"Processing model: {modelname}")

        ckpt_path = model_config.get('BYOC candidate')
        if not ckpt_path:
            logger.warning(f"No BYOC candidate found for {modelname}. Skipping.")
            continue

        model_specific_config = config['models'].get(modelname, {})
        appname = model_specific_config.get('appname', model_config.get('App name', 'DefaultApp'))
        source = config['byoc_source']
        modelarch = model_specific_config.get('modelarch', model_config.get('model_arch', 'DefaultArch'))
        paracnt = model_specific_config.get('paracnt', model_config.get('Param Count', '0b'))
        sslen = model_specific_config.get('sslen', model_config.get('Seq Length', '0'))
        vocabsize = model_specific_config.get('vocabsize', model_config.get('Vocab Size', '0'))

        model_added_successfully, time_taken, model_name, model_id = snapi_byoc_import_model(
            modelname, ckpt_path, appname, modelarch, paracnt, sslen, vocabsize, source
        )

        if not model_added_successfully:
            logger.error(f"Model import failed for {modelname}. Skipping to next model.")
            continue

        logger.info(f"Model {model_name} (ID: {model_id}) uploaded in {time_taken} seconds.")

        status = None
        start_time = time.time()
        while status != "Available" and time.time() - start_time < config['endpoint_creation_timeout']:
            status = check_byoc_checkpoint_status(model_id)
            logger.info(f"BYOC upload status for {model_name}: {status}")
            if status is None or status == "Failed":
                logger.error(f"BYOC checkpoint status check failed for {model_name}.")
                break
            if status != "Available":
                time.sleep(60)  # Wait for 1 minute before checking again

        if status != "Available":
            logger.error(f"BYOC upload did not complete successfully for {model_name}. Cleaning up and skipping to next model.")
            delete_byoc_model(model_id)
            continue

        coename, coe_status = create_coe_one_expert(model_name)
        logger.info(f"COE creation for {model_name}: Name={coename}, Status={coe_status}")

        if not coe_status:
            logger.error(f"COE creation failed for {model_name}. Cleaning up and skipping to next model.")
            delete_byoc_model(model_id)
            continue

        project_id, endpoint_id, api_key, endpoint_name = deploy_one_endpoint(coename)
        if not all([project_id, endpoint_id, api_key, endpoint_name]):
            logger.error(f"Endpoint deployment failed for {model_name}. Cleaning up and skipping to next model.")
            delete_one_coe(coename)
            delete_byoc_model(model_id)
            continue

        logger.info(f"Endpoint deployed for {model_name}: Project ID={project_id}, Endpoint ID={endpoint_id}, Endpoint Name={endpoint_name}")

        logger.info(f"Starting benchmark for {model_name}...")
        run_turbo_coe_benchmark(project_id, endpoint_id, api_key, model_name, coename, config)
        logger.info(f"Benchmark completed for {model_name}.")

        logger.info(f"Cleaning up resources for {model_name}...")
        
        if not delete_one_endpoint(endpoint_name):
            logger.warning(f"Failed to delete endpoint {endpoint_name} for {model_name}.")
        
        if not delete_one_coe(coename):
            logger.warning(f"Failed to delete COE {coename} for {model_name}.")
        
        if not delete_byoc_model(model_id):
            logger.warning(f"Failed to delete BYOC model {model_id} for {model_name}.")

        logger.info(f"Finished processing {model_name}.")
        

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BYOC Import and Benchmark Tool")
    parser.add_argument('-c', '--config', default='config.yaml', help='Path to the configuration file')
    parser.add_argument('-m', '--model', help='Process a specific model (optional)')
    parser.add_argument('-l', '--list', action='store_true', help='List available models')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    model_dict = read_master_menu(config['master_menu_path'])

    if args.list:
        print("Available models:")
        for model in model_dict.keys():
            print(f"- {model}")
        sys.exit(0)

    if args.model:
        if args.model not in model_dict:
            logger.error(f"Model '{args.model}' not found in the master menu.")
            sys.exit(1)
        model_dict = {args.model: model_dict[args.model]}

    main_byoc(args.config)