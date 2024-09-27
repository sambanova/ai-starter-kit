import os
import sys

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import io
import json
import shutil
import tarfile
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv(os.path.join(repo_dir, '.env'))

PENDING_RDU_JOB_STATUS = 'PENDING_RDU'
SUCCESS_JOB_STATUS = 'EXIT_WITH_0'
FAILED_JOB_STATUS = 'FAILED'


class BatchClipProcessor:
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the BatchClipProcessor class.

        Args:
            config_path (str, optional): Path to the YAML configuration file. Defaults to './config.yaml'.
        """
        if config_path is None:
            config_path = os.path.join(kit_dir, 'config.yaml')

        self.config = self._load_config(config_path)

        self.headers = {
            'content-type': 'application/json',
            'key': os.getenv('SAMBASTUDIO_KEY'),
        }
        self.datasets_path = self.config['clip']['datasets']['datasets_path']
        self.dataset_id = None
        self.dataset_name = self.config['clip']['datasets']['dataset_name']
        self.dataset_description = self.config['clip']['datasets']['dataset_description']
        self.dataset_source_type = self.config['clip']['datasets']['dataset_source_type']
        self.dataset_source_file = self.config['clip']['datasets']['dataset_source_file']

        self.clip_app_id = self.config['clip']['apps']['clip_app_id']

        self.base_url = self.config['clip']['urls']['base_url']
        self.datasets_url = self.config['clip']['urls']['datasets_url']
        self.projects_url = self.config['clip']['urls']['projects_url']
        self.jobs_url = self.config['clip']['urls']['jobs_url']
        self.download_results_url = self.config['clip']['urls']['download_results_url']

        self.project_name = self.config['clip']['projects']['project_name']
        self.project_description = self.config['clip']['projects']['project_description']
        self.project_id = None

        self.job_name = self.config['clip']['jobs']['job_name']
        self.job_task = self.config['clip']['jobs']['job_task']
        self.job_type = self.config['clip']['jobs']['job_type']
        self.job_description = self.config['clip']['jobs']['job_description']
        self.model_checkpoint = self.config['clip']['jobs']['model_checkpoint']

        self.output_path = self.config['clip']['output']['output_path']

    def _load_config(self, file_path: str) -> Any:
        """Loads a YAML configuration file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: The configuration data loaded from the YAML file.
        """
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _get_call(
        self, url: str, params: Optional[Dict[str, Any]] = None, success_message: Optional[str] = None
    ) -> requests.Response:
        """Make a GET request to the specified URL.

        Args:
            url (str): The URL to make the GET request to.
            params (Optional[Dict], optional): A dictionary of parameters to pass to the URL.
                Defaults to None.
            success_message (Optional[str], optional): A message to log upon successful completion of the
                GET request. Defaults to None.

        Returns:
            requests.Response: The response from the GET request.
        """
        response = requests.get(url, params=params, headers=self.headers)

        if response.status_code == 200:
            logging.info('GET request successful!')
            logging.info(success_message)
            logging.debug(f'Response: {response.text}')
        else:
            logging.error(f'GET request failed with status code: {response.status_code}')
            logging.error(f'Error message: {response.text}')
        return response

    def _post_call(
        self, url: str, params: Optional[Dict[str, Any]] = None, success_message: Optional[str] = None
    ) -> requests.Response:
        """Make a POST request to the specified URL.

        Args:
            url (str): The URL to make the POST request to.
            params (Optional[Dict], optional): A dictionary of parameters to pass to the URL.
                Defaults to None.
            success_message (Optional[str], optional): A message to log upon successful completion of the
                POST request. Defaults to None.

        Returns:
            requests.Response: The response from the POST request.

        """
        response = requests.post(url, json=params, headers=self.headers)

        if response.status_code == 200:
            logging.info('POST request successful!')
            logging.info(success_message)
            logging.debug(f'Response: {response.text}')
        else:
            logging.error(f'POST request failed with status code: {response.status_code}')
            raise Exception(f'Error message: {response.text}')
        return response

    def _delete_call(self, url: str) -> requests.Response:
        """Make a Delete request to the specified URL.

        Args:
            url (str): The URL to make the Delete request to.

        Returns:
            requests.Response: The response from the Delete request.

        """
        response = requests.delete(url, headers=self.headers)
        if response.status_code == 200:
            logging.info(f'Dataset {self.dataset_name} deleted successfully.')
            logging.debug(f'Response: {response.text}')
        else:
            logging.error(f'Failed to delete the resource. Status code: {response.status_code}')
            raise Exception(f'Error message: {response.text}')
        return response

    def _generate_csv(self, dataset_dir: str) -> None:
        """Generates a CSV file containing the paths to the images in the dataset.

        Args:
            dataset_dir (str): The path to the directory containing the dataset.
        """
        image_paths = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    image_path = os.path.relpath(os.path.join(root, file), dataset_dir)
                    image_paths.append(image_path)

        df = pd.DataFrame({'image_path': image_paths, 'description': '', 'subset': '', 'metadata': ''})
        df.to_csv(os.path.join(dataset_dir, 'predictions.csv'), index=False)

    def _get_df_output(self, response_content: bytes) -> DataFrame:
        """Parse the response from the CLIP job.

        Args:
            response_content (str): The response from the CLIP job.

        Returns:
            DataFrame: A DataFrame containing the parsed output from the CLIP job.
        """
        compressed_bytes = io.BytesIO(response_content)

        with tarfile.open(fileobj=compressed_bytes, mode='r:gz') as tar:
            output_tar_member = tar.getmember(self.output_path)
            output_file = tar.extractfile(output_tar_member)
            assert output_file is not None
            output_df = pd.read_json(io.BytesIO(output_file.read()), lines=True)
        return output_df

    def search_dataset(self, dataset_name: str) -> Any:
        """Search for a dataset in SambaStudio.

        Args:
            dataset_name (str): The name of the dataset to search for.
        Returns:
            dataset_id (str): The id of the searched dataset
        """
        url = self.base_url + self.datasets_url + '/search'
        params = {'dataset_name': dataset_name}
        response = self._get_call(url, params, f'Dataset {dataset_name} found in SambaStudio')
        parsed_reponse = json.loads(response.text)
        return parsed_reponse['data']['dataset_id']

    def delete_dataset(self, dataset_name: str) -> None:
        """Delete a dataset from SambaStudio.
        Args:
            dataset_name (str): The name of the dataset to delete.
        """
        dataset_id = self.search_dataset(dataset_name)
        url = self.base_url + self.datasets_url + '/' + dataset_id
        response = self._delete_call(url)
        logging.info(response.text)

    def create_dataset(self, path: str) -> str:
        """Create a dataset for openclip batch inference in SambaStudio.

        Args:
            path (str): The path to the audio files to create the dataset from.

        Returns:
            dataset_name (str): The name of the created dataset.
        """

        # create clip directory and source.json file

        dataset_name = f'{self.dataset_name}_{int(time.time())}'

        clip_directory = os.path.join(self.datasets_path, dataset_name)

        if not os.path.isdir(self.datasets_path):
            os.mkdir(self.datasets_path)

        if not os.path.isdir(clip_directory):
            logging.info(f'Datasets path: {clip_directory} not found')

            source_file_data = {'source_path': clip_directory}

            with open(self.dataset_source_file, 'w') as json_file:
                json.dump(source_file_data, json_file)

        shutil.copytree(path, clip_directory)

        self._generate_csv(clip_directory)

        # create dataset
        command = f'echo yes | snapi dataset add \
            --dataset-name {dataset_name} \
            --job_type {self.job_type} \
            --apps {self.clip_app_id} \
            --source_type {self.dataset_source_type} \
            --source_file {self.dataset_source_file} \
            --description "{self.dataset_description}"'

        os.system(command)
        logging.info(f'Creating dataset: {dataset_name}')

        return dataset_name

    def check_dataset_creation_progress(self, dataset_name: str) -> bool:
        """Check dataset creation progress of a given dataset

        Args:
            dataset_name (str): The name of the dataset to check.

        Returns:
            bool: True if the dataset is created, False otherwise.
        """
        url = self.base_url + self.datasets_url + '/' + dataset_name
        response = self._get_call(url)
        if response.json()['data']['status'] == 'Available':
            return True
        else:
            return False

    def create_load_project(self) -> Any:
        """Create or load project in SambaStudio.

        Returns:
            project_id (str): The id of the created project.
        """
        url = self.base_url + self.projects_url + '/' + self.project_name

        response = self._get_call(url, success_message=f'Project {self.project_name} found in SambaStudio')
        not_found_error_message = f'{self.project_name} not found'

        if not_found_error_message in response.text:
            logging.info(f'Project {self.project_name} not found in SambaStudio')

            url = self.base_url + self.projects_url

            params = {'project_name': self.project_name, 'description': self.project_description}

            response = self._post_call(url, params, success_message=f'Project {self.project_name} created!')

        parsed_reponse = json.loads(response.text)
        self.project_id = parsed_reponse['data']['project_id']
        return self.project_id

    def run_job(self, dataset_name: str) -> Any:
        """Run a batch inference job in SambaStudio.

        Args:
            dataset_name (str): The name of the dataset to run the job on.
        Returns:
            job_id (str): The id of the created job.
        """
        url = self.base_url + self.projects_url + self.jobs_url.format(project_id=self.project_id)

        params = {
            'task': self.job_task,
            'job_type': self.job_type,
            'job_name': f'{self.job_name}_{int(time.time())}',
            'project': self.project_id,
            'model_checkpoint': self.model_checkpoint,
            'description': self.job_description,
            'dataset': dataset_name,
        }

        response = self._post_call(url, params, success_message='Job running')
        parsed_reponse = json.loads(response.text)
        job_id = parsed_reponse['data']['job_id']

        return job_id

    def check_job_progress(self, job_id: str) -> bool:
        """Check job progress of a given job.

        Args:
            job_id (str): The id of the job to check.

        Returns:
            bool: True when the job is finished.
        """

        url = self.base_url + self.projects_url + self.jobs_url.format(project_id=self.project_id) + '/' + job_id

        status = PENDING_RDU_JOB_STATUS
        while status != SUCCESS_JOB_STATUS:
            response = self._get_call(url, success_message='Still waiting for job to finish')
            parsed_reponse = json.loads(response.text)
            status = parsed_reponse['data']['status']
            logging.info(f'Job status: {status}')
            if status == SUCCESS_JOB_STATUS:
                logging.info('Job finished!')
                break
            elif status == FAILED_JOB_STATUS:
                logging.info('Job failed!')
                return False
            time.sleep(10)

        return True

    def delete_job(self, job_id: str) -> None:
        """Delete a job from SambaStudio.

        Args:
            job_id (str): The id of the job to delete.
        """
        url = self.base_url + self.projects_url + self.jobs_url.format(project_id=self.project_id) + '/' + job_id
        response = self._delete_call(url)
        logging.info(response.text)

    def retrieve_results(self, job_id: str) -> DataFrame:
        """Retrieve results from a finished batch inference job

        Args:
            job_id (str): The id of the job to retrieve results of.
        Returns:
            df (pandas.DataFrame): The results of the batch inference job.
        """
        url = (
            self.base_url
            + self.projects_url
            + self.jobs_url.format(project_id=self.project_id)
            + '/'
            + job_id
            + self.download_results_url
        )
        response = self._get_call(url, success_message='Results downloaded!')
        df = self._get_df_output(response.content)
        return df

    def process_images(self, path: str) -> DataFrame:
        """Process and generate embedding for images in SambaStudio.

        Args:
            path (str): The path with contining images to process.
        Returns:
            df (pandas.DataFrame): The results of the batch inference job.
        """
        self.create_load_project()
        dataset_name = self.create_dataset(path=path)
        while not self.check_dataset_creation_progress(dataset_name):
            print('waiting for dataset available')
            time.sleep(5)
        job_id = self.run_job(dataset_name)
        job_finished = self.check_job_progress(job_id)
        if job_finished:
            df = self.retrieve_results(job_id)
            self.delete_job(job_id)
            self.delete_dataset(dataset_name)
            return df
        else:
            self.delete_job(job_id)
            self.delete_dataset(dataset_name)
            raise Exception('Job failed!')
