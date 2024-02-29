
import os
import io
import time
import shutil
import json
import yaml
import tarfile
import requests
import pandas as pd
from pandas import DataFrame
from dotenv import load_dotenv
import logging

load_dotenv('../export.env')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

PENDING_RDU_JOB_STATUS = 'PENDING_RDU'
SUCCESS_JOB_STATUS = 'EXIT_WITH_0'

class BatchASRProcessor():
    
    def __init__(self, config_path='./config.yaml') -> None:

        self.config = self._load_config(config_path) 
               
        self.headers = {
            'content-type': 'application/json',
            'key': os.getenv('SAMBASTUDIO_KEY'),
        }
        self.datasets_path = self.config['asr']['datasets']['datasets_path']
        self.dataset_id = None
        self.dataset_name = self.config['asr']['datasets']['dataset_name']
        self.dataset_description = self.config['asr']['datasets']['dataset_description']
        self.dataset_source_type = self.config['asr']['datasets']['dataset_source_type']
        self.dataset_source_file = self.config['asr']['datasets']['dataset_source_file']
        self.dataset_language = self.config['asr']['datasets']['dataset_language']
        
        self.asr_with_diarization_app_id = self.config['asr']['apps']['asr_with_diarization_app_id']
        self.application_field = self.config['asr']['apps']['application_field']
        
        self.base_url = self.config['asr']['urls']['base_url']
        self.datasets_url = self.config['asr']['urls']['datasets_url'] 
        self.projects_url = self.config['asr']['urls']['projects_url'] 
        self.jobs_url = self.config['asr']['urls']['jobs_url'] 
        self.download_results_url = self.config['asr']['urls']['download_results_url'] 
    
        self.project_name = self.config['asr']['projects']['project_name']
        self.project_description = self.config['asr']['projects']['project_description']
        self.project_id=None
        
        self.job_name = self.config['asr']['jobs']['job_name']
        self.job_task = self.config['asr']['jobs']['job_task']
        self.job_type = self.config['asr']['jobs']['job_type']
        self.job_description = self.config['asr']['jobs']['job_description']
        self.model_checkpoint = self.config['asr']['jobs']['model_checkpoint']
        
        self.output_path = self.config['asr']['output']['output_path']
     
    def _load_config(self, file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config   
        
    def _get_call(self, url, params = None, success_message = None):
        response = requests.get(url, params=params, headers=self.headers)

        if response.status_code == 200:
            logging.info('GET request successful!')
            logging.info(success_message)
            logging.debug(f'Response: {response.text}')
        else:
            logging.error(f'GET request failed with status code: {response.status_code}')
            logging.error(f'Error message: {response.text}')
        return response

    def _post_call(self, url, params, success_message = None):
        response = requests.post(url, json=params, headers=self.headers)

        if response.status_code == 200:
            logging.info('POST request successful!')
            logging.info(success_message)
            logging.debug(f'Response: {response.text}')
        else:
            logging.error(f'POST request failed with status code: {response.status_code}')
            raise Exception(f'Error message: {response.text}')
        return response
    
    def _delete_call(self, url):
        response = requests.delete(url, headers=self.headers)    
        if response.status_code == 200:
            logging.info(f'Dataset {self.dataset_name} deleted successfully.')
            logging.debug(f'Response: {response.text}')
        else:
            logging.error(f'Failed to delete the resource. Status code: {response.status_code}')
            raise Exception(f'Error message: {response.text}')    
        return response

    def _time_to_seconds(self, time_str):
        minutes, seconds = map(int, time_str.split(':'))
        return  minutes * 60 + seconds

    def _get_df_output(self, response_content: str) -> DataFrame:
        compressed_bytes = io.BytesIO(response_content)
        
        with tarfile.open(fileobj=compressed_bytes, mode="r:gz") as tar:
            output_tar_member = tar.getmember(self.output_path)
            output_file = tar.extractfile(output_tar_member)
            output_df = pd.read_csv(io.BytesIO(output_file.read()), names = ['audio_path', 'results_path', 'speaker', 'start_time', 'sample_duration', 'unformatted_transcript', 'formatted_transcript'])
            output_df['start_time'] = output_df.apply(lambda x: self._time_to_seconds(x['start_time']), axis = 1)
            output_df['end_time'] = output_df.apply(lambda x: x['start_time'] + int(x['sample_duration'])/16000, axis = 1)
            output_df = output_df[['start_time', 'end_time', 'speaker', 'formatted_transcript']].rename(columns={'formatted_transcript': 'text'})
        
        return output_df

    def search_dataset(self, dataset_name):
        url = self.base_url + self.datasets_url + '/search'
        params = {
            'dataset_name': dataset_name
        }
        response = self._get_call(url, params, f'Dataset {dataset_name} found in SambaStudio')
        parsed_reponse = json.loads(response.text)
        return parsed_reponse['data']['dataset_id']

    def delete_dataset(self, dataset_name):
        dataset_id = self.search_dataset(dataset_name)
        url = self.base_url + self.datasets_url + '/' + dataset_id
        response = self._delete_call(url)
        logging.info(response.text)
        
        
    def create_dataset(self, path):
                
        dataset_name = f'{self.dataset_name}_{int(time.time())}'
        
        # create pca directory and source.json file
        pca_directory = self.datasets_path + '/' + dataset_name
        
        if not os.path.isdir(self.datasets_path):
            os.mkdir(self.datasets_path) 
            
        if not os.path.isdir(pca_directory):
            logging.info(f'Datasets path: {pca_directory} wan\'t found')
            
            source_file_data = {
                "source_path": pca_directory
            }
            with open(self.dataset_source_file, 'w') as json_file:
                json.dump(source_file_data, json_file)
            os.mkdir(pca_directory)
            
            logging.info(f'PCA Directory: {pca_directory} created')
    
        # validate audio file
        audio_format = path.split('.')[-1]
        
        if audio_format == 'mp3':
            shutil.copyfile(path, pca_directory + '/pca_file.mp3')
        elif audio_format == 'wav':
            shutil.copyfile(path, pca_directory + '/pca_file.wav')
        else:
            raise Exception('Only mp3 and wav audio files supported')
        
        # create dataset
        command = f'echo yes | snapi dataset add \
            --dataset-name {dataset_name} \
            --job_type {self.job_type} \
            --apps {self.asr_with_diarization_app_id} \
            --source_type {self.dataset_source_type} \
            --source_file {self.dataset_source_file} \
            --application_field {self.application_field} \
            --language {self.dataset_language} \
            --description "{self.dataset_description}"'
        
        os.system(command)
        logging.info(f'Creating dataset: {dataset_name}')
        
        return dataset_name
                
    def check_dataset_creation_progress(self, dataset_name):
        url = self.base_url + self.datasets_url + '/' + dataset_name
        response = self._get_call(url)
        if response.json()["data"]["status"]=="Available": 
            return True
        else:
            return False
            
    def create_load_project(self):

        url = self.base_url + self.projects_url + '/' + self.project_name

        response = self._get_call(url, success_message=f'Project {self.project_name} found in SambaStudio')
        not_found_error_message = f"{self.project_name} not found"

        if not_found_error_message in response.text:
            
            logging.info(f'Project {self.project_name} wasn\'t found in SambaStudio')
            
            url = self.base_url + self.projects_url

            params = {
                'project_name': self.project_name,
                'description': self.project_description
            }

            response = self._post_call(url, params, success_message=f'Project {self.project_name} created!')

        parsed_reponse = json.loads(response.text)
        self.project_id = parsed_reponse['data']['project_id']
        return self.project_id
    
    def run_job(self, dataset_name):
        
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
    
    def check_job_progress(self, job_id):

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
            time.sleep(10)
        
        return True
    
    def delete_job(self, job_id):
        url = self.base_url +  self.projects_url + self.jobs_url.format(project_id=self.project_id) + '/' + job_id
        response = self._delete_call(url)
        logging.info(response.text)
        
    def retrieve_results(self, job_id):
        url = self.base_url + self.projects_url + self.jobs_url.format(project_id=self.project_id) + '/' + job_id + self.download_results_url
        response = self._get_call(url, success_message='Results downloaded!')
        df = self._get_df_output(response.content)
        return df
    
    def process_audio(self, path):
        self.create_load_project()
        dataset_name = self.create_dataset(path=path)
        while not self.check_dataset_creation_progress(dataset_name):
            print("waiting for dataset available")
            time.sleep(5)
        job_id = self.run_job(dataset_name)
        self.check_job_progress(job_id) 
        df = self.retrieve_results(job_id)
        self.delete_job(job_id)
        self.delete_dataset(dataset_name)
        return df