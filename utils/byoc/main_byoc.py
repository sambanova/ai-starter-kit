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
from general_model_testing_object import GeneralModelTestObject


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Create a file handler for logging
file_handler = logging.FileHandler('byoc_process.log')
file_handler.setLevel(logging.INFO)

# Create a logging format and add the handler to the logger
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def snapi_byoc_import_model(modelname, ckpt_path, appname, modelarch, paracnt, sslen, vocabsize, source="LOCAL"):
    modelname = f"qa-{modelname}-{source}-byoc-test"
    command = "snapi import model create " + \
              f"""-m "{modelname}" """ + \
              f"""-a "{appname}" """ + \
              f"-t {source} " + \
              f"-s {ckpt_path} " + \
              f"""-p "sns" """ + \
              f"""-d "QA test on byoc" """ + \
              f"-ma {modelarch} " + \
              f"-c {paracnt} " + \
              f"-l {sslen} " + \
              f"-vs {vocabsize} -ni"
    logger.info(f"SNAPI BYOC command: {command}")

    res = subprocess.run(command, input='yes', encoding='ascii', shell=True, stdout=subprocess.PIPE)
    content = res.stdout

    model_added_pattern = r"Model added successfully"
    time_pattern = r"Time taken to upload the model:\s([\d\.]+)\sseconds"
    uuid_pattern = r"snapi import model status -m ([a-f0-9\-]+)"

    # Check for "Model added successfully"
    model_added_successfully = re.search(model_added_pattern, content) is not None

    # Extract time taken
    time_taken_match = re.search(time_pattern, content)
    time_taken = float(time_taken_match.group(1)) if time_taken_match else None

    # Extract UUID
    uuid_match = re.search(uuid_pattern, content)
    uuid = uuid_match.group(1) if uuid_match else None

    return model_added_successfully, time_taken, modelname, uuid
            
def check_byoc_checkpoint_status(modelid):
    command = f"snapi import model status -m {modelid}"

    res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = res.communicate()
    
    if error:
        print("Error:", error.decode())

    status = None
    for line in output.decode().splitlines():
        if 'Status' in line:
            tmp = line.split(': ')
            status = tmp[-1].strip()

    return status

def create_coe_one_expert(modelname):
    coename = f"{modelname}-coe"
    command = f"snapi model add-composite -n {coename} -e {modelname}"

    res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = res.communicate()
    
    if error:
        print("Error:", error.decode())

    status = False
    for line in output.decode().splitlines():
        if 'Successfully' in line or 'successfully' in line:
            status = True

    return coename, status

def deploy_one_endpoint(modelname):
    handler = GeneralModelTestObject("snapi_LLM_1", "SN40L")

    res, api_key, url = handler.model_endpoint(modelname)

    try:
        tmp = url.split('/')
        project_id = tmp[-2]
        endpoint_id = tmp[-1]
    except:
        return None, None, None, None

    if not res:
        return None, None, None, None
    
    return project_id, endpoint_id, api_key, f"{modelname.lower()}-1ins-endpoint-sn40l"

def delete_one_endpoint(endpoint_name):
    command = f"snapi endpoint delete -p snapi_LLM_1 -e {endpoint_name}"

    res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = res.communicate()
    
    if error:
        print("Error:", error.decode())

    status = False
    for line in output.decode().splitlines():
        print(line)
        if 'Successfully' in line or 'successfully' in line:
            status = True

    return status

def delete_one_coe(coename):
    command = f"snapi model remove -m {coename}"

    res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = res.communicate()
    
    if error:
        print("Error:", error.decode())

    status = False
    for line in output.decode().splitlines():
        if 'Successfully' in line or 'successfully' in line:
            status = True

    return status

def delete_byoc_model(modelname):
    command = f"snapi import model delete -m {modelname}"

    res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = res.communicate()
    
    if error:
        print("Error:", error.decode())

    status = False
    for line in output.decode().splitlines():
        if 'Successfully' in line or 'successfully' in line:
            status = True

    return status

def run_turbo_coe_benchmark(project_id, endpoint_id, api_key, expert, coename):
    script_directory = '../sambaX'

    # Define the bash command to run
    bash_command = f"./m2_run_GQ_turbo_coe.sh --streamURL https://sjc3-tstest.sambanova.net/api/v2/predict/generic/stream/{project_id}/{endpoint_id} --apiKEY {api_key} --experts {expert} --app {coename}"

    print("bash command: ", bash_command)
    # Change the current working directory to the script directory
    os.chdir(script_directory)

    # Run the bash command
    try:
        result = subprocess.run(bash_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Print the output of the command
        print(result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        # Print the error if the command fails
        print(e.stderr.decode('utf-8'))

def read_master_menu(filepath):
    # Read the CSV file
    df = pd.read_csv(filepath)

    # Convert DataFrame to dictionary with 'model_name' as the key
    model_dict = defaultdict(dict)
    for index, row in df.iterrows():
        model_name = row['model_name']
        if pd.notna(model_name):  # Check if model_name is not NaN
            # Create a dictionary for the current row with specific transformations
            model_details = {}
            for column, value in row.items():
                if pd.notna(value):
                    value = str(value).strip()
                    if column == 'Param Count':
                        # Convert float string to int correctly
                        model_details[column] = f"{int(float(value))}b"  # Format with 'b' suffix
                    elif column == '3-graph':
                        # Process '3-graph' entries
                        model_details[column] = [int(float(x.replace('k', '')) * 1024) if 'k' in x else int(float(x)) for x in value.split(',')]
                    elif column == 'Batch Sizes':
                        # Convert batch sizes to integer list
                        model_details[column] = [int(float(x)) for x in value.split(',')]
                    else:
                        model_details[column] = value  # Keep other fields as strings

            model_dict[model_name] = model_details

    # Remove the 'model_name' key from each sub-dictionary
    for model in list(model_dict):
        if 'model_name' in model_dict[model]:
            del model_dict[model]['model_name']
        
        if 'BYOC candidate' not in model_dict[model]:
            model_dict[model]['BYOC candidate'] = None 

    return model_dict

def main_byoc():

    model_dict = read_master_menu("../../data/csv/samba_turbo_byoc_menu.csv")

    for modelname in model_dict:

        ckpt_path = model_dict[modelname]['BYOC candidate']
        appname = model_dict[modelname]['App name']
        source = 'LOCAL'
        modelarch = model_dict[modelname]['model_arch']
        paracnt = model_dict[modelname]['Param Count']
        sslen = model_dict[modelname]['Seq Length']
        vocabsize = model_dict[modelname]['Vocab Size']
        
        model_added_successfully, time_taken, model_name, model_id = snapi_byoc_import_model(modelname, 
                                                                            ckpt_path, 
                                                                            appname, 
                                                                            modelarch, 
                                                                            paracnt, 
                                                                            sslen, 
                                                                            vocabsize, 
                                                                            source)
        if not model_added_successfully:
            logger.error("Model import failed. Exiting...")
            continue
        logger.info(f"ckpt upload time consumption: {time_taken}, model name: {model_name}, model id: {model_id}")

        time.sleep(180)

        status = None
        while status != "Available":
            status = check_byoc_checkpoint_status(model_id)
            logger.info(f"BYOC upload from Local status: {status}")
            if status is None or status == "Failed":
                logger.error("BYOC checkpoint status check failed. Exiting...")
                break

        if status == "Failed" or not status: # no checkpoint uploaded successfully
            continue

        coename, status = create_coe_one_expert(model_name)
        logger.info(f"coe name: {coename}; coe status: {status}")

        if not status:
            logger.error("COE creation failed. Exiting...")
            continue

        time.sleep(60)

        project_id, endpoint_id, api_key, endpoint_name = deploy_one_endpoint(coename)
        if not all([project_id, endpoint_id, api_key, endpoint_name]):
            logger.error("Endpoint deployment failed. Exiting...")
            continue
        logger.info(f"endpoint information: project id: {project_id}, endpoint id: {endpoint_id}, api key: {api_key}, endpoint name: {endpoint_name}")

        logger.info(f"Started to run benchmark...")
        run_turbo_coe_benchmark(project_id, endpoint_id, api_key, model_name, coename)
        logger.info(f"Finished to run benchmark...")

        time.sleep(30)

        status = delete_one_endpoint(endpoint_name)
        if not status:
            logger.error(f"Failed to delete endpoint, endpoint name: {endpoint_name}. Exiting...")
            continue
        logger.info(f"status of deleting endpoint: {status}")

        status = delete_one_coe(coename)
        if not status:
            logger.error("Failed to delete COE. Exiting...")
            continue
        logger.info(f"status of deleting coe: {status}")

        status = delete_byoc_model(model_id)
        if not status:
            logger.error("Failed to delete BYOC checkpoint. Exiting...")
            continue
        logger.info(f"status of deleting byoc checkpoint: {status}")

if __name__=="__main__":
    main_byoc()
