import os
import json
import yaml
from snsdk import SnSdk
import logging
from typing import Optional

SNAPI_PATH = '~/.snapi'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)


class SnsdkWrapper:
    """ "Wrapper around the SnSdk and SNAPI for E2E fine-tuning in SambaStudio"""

    """init and get configs"""

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Wrapper around the SnSdk and SNAPI for E2E fine-tuning in SambaStudio
        Args:
            config_path (str , optional): path to config path. Defaults to None.
            see a config file example in ./config.yaml
        """
        self.config = {}
        self.tenant_name = None
        self.snapi_path = SNAPI_PATH

        if config_path is not None:
            self.config = self._load_config(config_path)
            config_snapi_path = self.config['sambastudio']['snapi_path']
            if config_snapi_path is not None and len(config_snapi_path) > 0:
                self.snapi_path = self.config['sambastudio']['snapi_path']

        host_url, tenant_id, access_key = self._get_sambastudio_variables()

        self.snsdk_client = SnSdk(
            host_url=host_url,
            access_key=access_key,
            tenant_id=tenant_id,
        )

    def _load_config(self, file_path: str) -> dict:
        """Loads a YAML configuration file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: The configuration data loaded from the YAML file.
        """
        try:
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'Error: The file {file_path} does not exist.')
        except yaml.scanner.ScannerError:
            raise ValueError(f'Error: The file {file_path} contains invalid yaml.')
        return config

    def _get_sambastudio_variables(self) -> tuple:
        """Gets Sambastudio host name, tenant id and access key from Snapi folder location

        Raises:
            FileNotFoundError: raises error when the snapi config or secret file is not found
            ValueError: raises error when the snapi config file doesn't contain a correct json format

        Returns:
            tuple: host name, tenant id and access key from snapi setup
        """
        snapi_config = ''
        snapi_secret = ''

        try:
            # reads snapi config json
            snapi_config_path = os.path.expanduser(self.snapi_path) + '/config.json'
            with open(snapi_config_path, 'r') as file:
                snapi_config = json.load(file)

            # reads snapi secret txt file
            snapi_secret_path = os.path.expanduser(self.snapi_path) + '/secret.txt'
            with open(snapi_secret_path, 'r') as file:
                snapi_secret = file.read()

        except FileNotFoundError:
            raise FileNotFoundError(f'Error: The file {snapi_config_path} does not exist.')
        except ValueError:
            raise ValueError(f'Error: The file {snapi_config_path} contains invalid JSON.')

        host_name = os.getenv('HOST_NAME')
        access_key = os.getenv('ACCESS_KEY')
        tenant_name = os.getenv('TENANT_NAME')

        if (host_name is not None) and (access_key is not None) and (tenant_name is not None):
            new_snapi_config = {}
            new_snapi_config['HOST_NAME'] = host_name
            new_snapi_config['CONFIG_DIR'] = snapi_config['CONFIG_DIR']
            new_snapi_config['DISABLE_SSL_WARNINGS'] = snapi_config['DISABLE_SSL_WARNINGS']
            with open(snapi_config_path, 'w') as file:
                json.dump(new_snapi_config, file)
            with open(snapi_secret_path, 'w') as file:
                file.write(access_key)

            snapi_config_response = subprocess.run(
                ['snapi', 'config', 'set', '--tenant', 'default'],
                capture_output=True,
                text=True,
            )

            with open(snapi_config_path, 'r') as file:
                new_snapi_config = json.load(file)
            new_new_snapi_config = new_snapi_config

            tmp_snsdk_client = SnSdk(
                host_url=host_name,
                access_key=access_key,
                tenant_id=new_snapi_config['TENANT_ID'],
            )

            tenant_info_response = tmp_snsdk_client.tenant_info(tenant=new_snapi_config['TENANT_ID'])
            tenant_id = tenant_info_response['data']['tenant_id']
            new_new_snapi_config['TENANT_ID'] = tenant_id

            with open(snapi_config_path, 'w') as file:
                json.dump(new_new_snapi_config, file)

            return host_name, tenant_id, access_key

        else:
            host_name = snapi_config['HOST_NAME']
            tenant_id = snapi_config['TENANT_ID']
            access_key = snapi_secret.strip()

        return host_name, tenant_id, access_key

    """Project"""

    def search_project(self, project_name: Optional[str] = None) -> Optional[str]:
        """
        Search for a project by its name in sambastudio environment.

        Parameters:
        project_name (str, optional): The name of the project to search for. 
            If not provided, the name from the configuration is used.

        Returns:
        Optional[str]: The ID of the project if found, otherwise None.
        """
        if project_name is None:
            project_name = self.config['project']['project_name']
        search_project_response = self.snsdk_client.search_project(project_name=project_name)
        if search_project_response['status_code'] == 200:
            project_id = search_project_response['data']['project_id']
            logging.info(f"Project with name '{project_name}' found with id {project_id}")
            return project_id
        else:
            logging.info(f"Project with name '{project_name}' not found")
            return None

    def create_project(self, project_name: Optional[str] = None, project_description: Optional[str] = None) -> str:
        """
        Creates a new project in SambaStudio.

        Parameters:
        project_name (str, optional): The name of the project. If not provided, the name from the configs file is used.
        project_description (str, optional): The description of the project. 
            If not provided, the description from the configs file is used.

        Returns:
        str: The ID of the newly created or existent project.

        Raises:
        Exception: If the project creation fails.
        """
        if project_name is None:
            project_name = self.config['project']['project_name']
        if project_description is None:
            project_description = self.config['project']['project_description']
        project_id = self.search_project(project_name)
        if project_id is None:
            create_project_response = self.snsdk_client.create_project(
                project_name=project_name, description=project_description
            )
            if create_project_response['status_code'] == 200:
                project_id = create_project_response['data']['project_id']
                logging.info(f'Project with name {project_name} created with id {project_id}')
            else:
                logging.error(
                    f"Failed to create project with name '{project_name}'. Details: {create_project_response['detail']}"
                )
                raise Exception(f"Error message: {create_project_response['detail']}")
        else:
            logging.info(f"Project with name '{project_name}' already exists with id '{project_id}', using it")
        return project_id

    def list_projects(self, verbose: Optional[bool] = False) -> list[dict]:
        """
        List all projects.

        Parameters:
        verbose (bool, optional): If True, detailed information about each project is returned. 
            If False, only basic information is returned. Defaults to False.

        Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a project. 
            If verbose is True, each dictionary contains detailed information about the project. 
            If verbose is False, each dictionary contains basic information about the project.

        Raises:
        Exception: If there is an error in listing the projects.
        """
        list_projects_response = self.snsdk_client.list_projects()
        if list_projects_response['status_code'] == 200:
            projects = []
            for project in list_projects_response['data'].get('projects'):
                if verbose:
                    projects.append({k: v for k, v in project.items()})
                else:
                    projects.append(
                        {k: v for k, v in project.items() if k in ['project_name', 'project_id', 'status', 'user_id']}
                    )
            return projects
        else:
            logging.error(f"Failed to list projects. Details: {list_projects_response['detail']}")
            raise Exception(f"Error message: {list_projects_response['detail']}")

    def delete_project(self, project_name: Optional[str] = None) -> None:
        """
        Deletes a project from the SambaStudio.
        Use with caution it will delete project, and its associated jobs, checkpoints, and endpoints

        Parameters:
        project_name (str, optional): The name of the project to be deleted. 
            If not provided, the name from the configs file will be used.

        Returns:
        None

        Raises:
        Exception: If the project is not found or if there is an error in deleting the project.
        """
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name=project_name)
        if project_id is None:
            raise Exception(f"Project with name '{project_name}' not found")
        delete_project_response = self.snsdk_client.delete_project(project=project_id)
        if delete_project_response['status_code'] == 200:
            logging.info(f"Project with name '{project_name}' deleted")
        else:
            logging.error(
                f"Failed to delete project with name or id '{project_name}'. Details: {delete_project_response}"
            )
            raise Exception(f'Error message: {delete_project_response}')

    """models"""

    def list_models(self, filter: Optional[list[str]] = [], verbose: Optional[bool] = False) -> list[dict]:
        """
        List models in sambastudio based on the provided filter.

        Parameters:
        filter (list[str], optional): A list of job types to filter the models. Defaults to [].
            Should include the job types supported by the models to filter 
            example: ['train', 'batch_predict', 'deploy']
        verbose (bool, optional): If True, return detailed information about each model. Defaults to False.

        Returns:
        list[dict]: A list of dictionaries, each representing a model. 
            If verbose is True, each dictionary contains all model information. 
            Otherwise, each dictionary contains only the model's checkpoint name and ID.

        Raises:
        Exception: If there is an error in listing models.
        """

        list_models_response = self.snsdk_client.list_models()
        if list_models_response['status_code'] == 200:
            models = []
            for model in list_models_response['models']:
                if set(filter).issubset(model.get('jobTypes')):
                    if verbose:
                        models.append({k: v for k, v in model.items()})
                    else:
                        models.append({k: v for k, v in model.items() if k in ['model_checkpoint_name', 'model_id']})
            return models
        else:
            logging.error(f"Failed to list models. Details: {list_models_response['detail']}")
            raise Exception(f"Error message: {list_models_response['detail']}")

    def search_model(self, model_name: Optional[str] = None) -> Optional[str]:
        """
        Search for a model in the SambaStudio by its name.

        Parameters:
        model_name (str, optional): The name of the model to search for. 
            If not provided, the name from the configs file is used.

        Returns:
        str or None: The ID of the model if found, otherwise None.
        """
        if model_name is None:
            model_name = self.config['job']['model']
        search_model_response = self.snsdk_client.search_model(model_name=model_name)
        if search_model_response['status_code'] == 200:
            model_id = search_model_response['data']['model_id']
            logging.info(f"Model with name '{model_name}' found with id {model_id}")
            return model_id
        else:
            logging.info(f"Project with name '{model_name}' not found")
            return None

    def search_trainable_model(self, model_name: Optional[str] = None) -> Optional[str]:
        """
        Search for a trainable and deployable  model in the SambaStudio by its name.

        Parameters:
        model_name (str, optional): The name of the model to search for. 
            If not provided, the name from the configs file is used.

        Returns:
        str or None: The ID of the model if found and is trainable and deployable, otherwise None.
        """
        if model_name is None:
            model_name = self.config['job']['model']
        models = self.list_models(filter=['train', 'deploy'])
        model_id = [model['model_id'] for model in models if model['model_checkpoint_name'] == model_name]
        if len(model_id) > 0:
            logging.info(f"Model '{model_name}' with id '{model_id[0]}' available for training and deployment found")
            return model_id[0]
        else:
            logging.info(f"Model '{model_name}' available for training and deployment not found")
            return None

    """Job"""

    def run_job(
        self,
        project_name: Optional[str] = None,
        job_name: Optional[str] = None,
        job_description: Optional[str] = None,
        job_type: Optional[str] = None,
        model: Optional[str] = None,
        dataset_name: Optional[str] = None,
        parallel_instances: Optional[int] = None,
        load_state: Optional[bool] = None,
        sub_path: Optional[str] = None,
        rdu_arch: Optional[str] = None,
        hyperparams: Optional[dict] = None,
    ) -> str:
        """
        Creates a new job in an specific SambaStudio project with the given parameters.

        Args:
            project_name (str, optional). The name of the project to run the job in . 
                If not provided, the project name from the configs file will be used.
            job_name (str, optional).  The name of the job. 
                If not provided, the job name from the configs file will be used.
            job_description (str, optional).  the description for the job. 
                If not provided, the description from the configs file will be used.
            job_type (str, optional).  the type of job to run can be "train" or "batch_inference". 
                If not provided, the type from the configs file will be used.
            model (str, optional).  the name of the model to fine tune. 
                If not provided, the model name from the configs file will be used.
            dataset_name (str, optional).  the name of the dataset to finetune with. 
                If not provided, the dataset name from the configs file will be used.
            parallel_instances (int, optional). the number of instances to use for the Job. 
                If not provided, the number of instances from the configs file will be used.
            load_state (bool, optional). Only load weights from the model checkpoint, if True. 
                If not provided, the param from the configs file will be used.
            sub_path (str, optional). Folder/file path inside dataset. 
                If not provided, the sub path from the configs file will be used.
            rdu_arch (str, optional). RDU Architecture to train with. 
                If not provided, the rdu arch from the configs file will be used.
            hyperparams (dict, optional). hyperparameters for executing the job. 
                If not provided, the hyperparameters from the configs file will be used.
                hyperparams example={
                    "batch_size": 256,
                    "do_eval": False,
                    "eval_steps": 50,
                    "evaluation_strategy": "no",
                    "learning_rate": 0.00001,
                    "logging_steps": 1,
                    "lr_schedule": "fixed_lr",
                    "max_sequence_length": 4096,
                    "num_iterations": 100,
                    "prompt_loss_weight": 0.0,
                    "save_optimizer_state": True,
                    "save_steps": 50,
                    "skip_checkpoint": False,
                    "subsample_eval": 0.01,
                    "subsample_eval_seed": 123,
                    "use_token_type_ids": True,
                    "vocab_size": 32000,
                    "warmup_steps": 0,
                    "weight_decay": 0.1,
                }
        Raises:
            Exception: If the project is not found.
            Exception: If the model is not found.
            Exception: If the dataset is not found.
            Exception: If there is an error creating the job.

        Returns:
            str: the job id
        """

        # check if project selected exists
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if model selected is found and is trainable
        if model is None:
            model = self.config['job']['model']
        model_id = self.search_trainable_model(model)
        if model_id is None:
            raise Exception(f"model with name '{model}' not found")

        # check if dataset exist
        if dataset_name is None:
            dataset_name = self.config['dataset']['dataset_name']
        dataset_id = self.search_dataset(dataset_name)
        if dataset_id is None:
            raise Exception(f"dataset with name '{dataset_name}' not found")

        # create job
        create_job_response = self.snsdk_client.create_job(
            project=project_id,
            job_name=job_name or self.config.get('job', {}).get('job_name'),
            description=job_description or self.config.get('job', {}).get('job_description'),
            job_type=job_type or self.config.get('job', {}).get('job_type'),
            model_checkpoint=model_id,
            dataset=dataset_id,
            parallel_instances=parallel_instances or self.config.get('job', {}).get('parallel_instances'),
            load_state=load_state or self.config.get('job', {}).get('load_state'),
            sub_path=sub_path or self.config.get('job', {}).get('sub_path'),
            rdu_arch=rdu_arch or self.config.get('sambastudio', {}).get('rdu_arch'),
            hyperparams=json.dumps(hyperparams or self.config.get('job', {}).get('hyperparams')),
        )

        if create_job_response['status_code'] == 200:
            job_id = create_job_response['job_id']
            logging.info(
                f"Job with name '{job_name or self.config.get('job',{}).get('job_name')}' created: '{create_job_response}'"
            )
            return job_id
        else:
            logging.error(
                f"Failed to create job with name '{job_name or self.config.get('job',{}).get('job_name')}'. Details: {create_job_response}"
            )
            raise Exception(f'Error message: {create_job_response}')

    def search_job(self, job_name: Optional[str] = None, project_name: Optional[str] = None) -> Optional[str]:
        """
        Search for a job in a specific SambaStudio project.

        Parameters:
        job_name (str, optional): The name of the job to search for. 
            If not provided, the job name from the configs file is used.
        project_name (str, optional): The name of the project to search in. 
            If not provided, the project name from the configs file is used.

        Returns:
        str: The ID of the job if found, otherwise None.

        Raises:
        Exception: If the project does not exist.
        """
        if job_name is None:
            job_name = self.config['job']['job_name']

        # check if project selected exists
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # search for job
        search_job_response = self.snsdk_client.search_job(project=project_id, job_name=job_name)
        if search_job_response['status_code'] == 200:
            job_id = search_job_response['data']['job_id']
            logging.info(f"Job with name '{job_name}' in project '{project_name}' found with id '{job_id}'")
            return job_id
        else:
            logging.info(f"Job with name '{job_name}' in project '{project_name}' not found")
            return None

    def check_job_progress(self, project_name: Optional[str] = None, job_name: Optional[str] = None) -> dict:
        """
        Check the progress of a job in a specific SambaStudio project.

        Parameters:
        project_name (str, optional): The name of the project. 
            If not provided, the project name from the configs file is used.
        job_name (str, optional): The name of the job. 
            If not provided, the job name from the configs file is used.

        Returns:
        dict: A dictionary containing the job progress status.

        Raises:
        Exception: If the project or job is not found.
        """

        # check if project selected exists
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if job selected exists
        if job_name is None:
            job_name = self.config['job']['job_name']
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")

        # check job progress
        check_job_progress_response = self.snsdk_client.job_info(project=project_id, job=job_id)
        if check_job_progress_response['status_code'] == 200:
            # TODO implement logic to filter out response, blocked by authorization error
            # job_progress = {}
            # logging.info(f"Job '{job_name}' with progress status: {job_progress}")
            return job_progress
        else:
            logging.error(f'Failed to check job progress. Details: {check_job_progress_response}')
            raise Exception(f'Error message: {check_job_progress_response}')

    def list_jobs(self, project_name: Optional[str] = None, verbose: Optional[bool] = False) -> list[dict]:
        """
        List all jobs in a specific SambaStudio project.

        Parameters:
        project_name (str, optional): The name of the project. 
            If not provided, the project name from the configs file is used.  
            If the project does not exist all user jobs are returned
        verbose (bool, optional): If True, detailed information about each job is returned. 
            If False, only basic information is returned.

        Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a job. 
            If verbose is True, each dictionary contains detailed information about the job. 
            If verbose is False, each dictionary contains basic information about the job.

        Raises:
        Exception: If there is an error in listing the jobs.
        """
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            logging.info(f"Project '{project_name}' not found listing all user jobs")
        list_jobs_response = self.snsdk_client.list_jobs(project_id=project_id)
        if list_jobs_response['status_code'] == 200:
            jobs = []
            for job in list_jobs_response['jobs']:
                if verbose:
                    jobs.append({k: v for k, v in job.items()})
                else:
                    jobs.append(
                        {
                            k: v
                            for k, v in job.items()
                            if k in ['job_name', 'job_id', 'job_type', 'project_id', 'status']
                        }
                    )
            return jobs
        else:
            logging.error(f"Failed to list jobs. Details: {list_jobs_response['detail']}")
            raise Exception(f"Error message: {list_jobs_response['detail']}")

    def delete_job(self, project_name: Optional[str] = None, job_name: Optional[str] = None) -> None:
        """
        Deletes a job from a specified SambaStudio project.
        Use with caution it deletes the job and all its associated checkpoints

        Parameters:
        project_name (str, optional): The name of the project. 
            If not provided, the project name from the configuration file is used.
        job_name (str, optional): The name of the job. 
            If not provided, the job name from the configuration file is used.

        Returns:
        None

        Raises:
        Exception: If the project or job is not found.
        Exception: If there is an error in deleting the job.
        """

        # check if project selected exists
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if job selected exists
        if job_name is None:
            job_name = self.config['job']['job_name']
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")

        # delete job
        delete_job_response = self.snsdk_client.delete_job(project=project_id, job=job_id)
        if delete_job_response['status_code'] == 200:
            logging.info(f"Job with name '{job_name}' in project '{project_name}' deleted")
            # TODO check if working, blocked by authorization error
        else:
            logging.error(
                f"Failed to delete job with name '{job_name}' in project '{project_name}'. Details: {delete_job_response}"
            )
            raise Exception(f'Error message: {delete_job_response}')

    """checkpoints"""

    def list_checkpoints(
        self, project_name: Optional[str] = None, job_name: Optional[str] = None, verbose: Optional[bool] = False
    ) -> list[dict]:
        """
        List all checkpoints in a specific job within a SambaStudio project.

        Parameters:
        project_name (str, optional): The name of the project. 
            If not provided, the project name from the configs file is used.
        job_name (str, optional): The name of the job. If not provided, the job name from the configs file is used.
        verbose (bool, optional): If True, detailed information about each checkpoint is returned. 
            If False, only basic information is returned.

        Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a checkpoint. 
            If verbose is True, each dictionary contains detailed information about the checkpoint. 
            If verbose is False, each dictionary contains basic information about the checkpoint.

        Raises:
        Exception: If the project does not exist or if the job does not exist within the project.
        Exception: If there is an error in listing the checkpoints.
        """

        # check if project selected exists
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if job selected exists
        if job_name is None:
            job_name = self.config['job']['job_name']
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")

        list_checkpoints_response = self.snsdk_client.list_checkpoints(project=project_id, job=job_id)
        if list_checkpoints_response['status_code'] == 200:
            checkpoints = []
            # TODO implement logic to filter out response, blocked by authorization error
            # for checkpoint in list_checkpoints_response:
            #    if verbose:
            #        checkpoints.append({k:v for k,v in checkpoint.items()})
            #    else:
            #        checkpoints.append(
            #            {k:v for k,v in checkpoint.items() if k in ["checkpoint_id", "status"]}
            #        )
            return checkpoints
        else:
            logging.error(f'Failed to list checkpoints. Details: {list_checkpoints_response}')
            raise Exception(f'Error message: {list_checkpoints_response}')

    def promote_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        project_name: Optional[str] = None,
        job_name: Optional[str] = None,
        model_name: Optional[str] = None,
        model_description: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> str:
        """
        Promotes a model checkpoint from a specific training job to a model in SambaStudio model hub.

        Parameters:
        - checkpoint_id (str, optional): The ID of the model checkpoint to be promoted. 
            If not provided, the checkpoint id from the configs file is used.
        - project_name (str, optional): The name of the project where the model checkpoint will be promoted. 
            If not provided, the project name from the configs file is used.
        - job_name (str, optional): The name of the job where the model checkpoint will be promoted. 
            If not provided, the job name from the configs file is used.
        - model_name (str, optional): The name of the model to which the model checkpoint will be promoted. 
            If not provided, the model name from the configs file is used.
        - model_description (str, optional): The description of the model to which the checkpoint will be promoted. 
            If not provided, the model description from the configs file is used.
        - model_type (str, optional): The type of the model to which the model checkpoint will be promoted 
            either "finetuned" or "pretrained". If not provided, the model type from the configs file is used.

        Returns:
        - str: The ID of the promoted model.

        Raises:
        - Exception: If the project or job is not found.
        - Exception: If the model checkpoint ID is not provided.
        - Exception: If there is an error promoting the model checkpoint to the model.
        """

        # check if project selected exists
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if job selected exists
        if job_name is None:
            job_name = self.config['job']['job_name']
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")

        if checkpoint_id is None:
            checkpoint_id = self.config['model_checkpoint']['model_checkpoint_id']
            if not checkpoint_id:
                raise Exception('No model checkpoint_id provided')
            # TODO: check if checkpoint in list checkpoints list blocked because authorization error in lists checkpoints method
            # if checkpoint_id not in self.list_checkpoints(project_name=project_name, job_name=job_name):
            #     raise Exception(f"Checkpoint id '{checkpoint_id}' not found in job '{job_name}'")

        add_model_response = self.snsdk_client.add_model(
            project=project_name or self.config.get('project', {}).get('project_name'),
            job=job_name or self.config.get('job', {}).get('job_name'),
            model_checkpoint=checkpoint_id,
            model_checkpoint_name=model_name or self.config.get('model_checkpoint', {}).get('model_name'),
            description=model_description or self.config.get('model_checkpoint', {}).get('model_description'),
            checkpoint_type=model_type or self.config.get('model_checkpoint', {}).get('model_type'),
        )

        if add_model_response['status_code'] == 200:
            logging.info(f"Model checkpoint '{checkpoint_id}' promoted to model '{model_name}'")
            # TODO test blocked because of authorization error
            # return model_id
        else:
            logging.error(f"Failed to promote checkpoint '{checkpoint_id}' to model. Details: {add_model_response}")
            raise Exception(f'Error message: {add_model_response}')

    def delete_checkpoint(self, checkpoint: str = None) -> None:
        """
        Deletes a model checkpoint from the SambaStudio model hub.

        Parameters:
        - checkpoint (str, optional): The name/id of the checkpoint or promoted model to be deleted. 
            If not provided, the checkpoint id from the configs file is used.

        Returns:
        - None

        Raises:
        - Exception: If no model checkpoint_id is provided.
        - Exception: If there is an error deleting the model checkpoint.
        """

        # check if checkpoint is provided
        if checkpoint is None:
            checkpoint = self.config['model_checkpoint']['model_checkpoint_id']
            if not checkpoint:
                raise Exception('No model checkpoint_id provided')

        delete_checkpoint_response = self.snsdk_client.delete_checkpoint(checkpoint=checkpoint)
        if delete_checkpoint_response['status_code'] == 200:
            logging.info(f"Model checkpoint '{checkpoint}' deleted")
        else:
            logging.error(f"Failed to delete checkpoint '{checkpoint}'. Details: {delete_checkpoint_response}")
            raise Exception(f'Error message: {delete_checkpoint_response}')

    """endpoint"""

    def list_endpoints(self, project_name: Optional[str] = None, verbose: Optional[bool] = None) -> list[dict]:
        """
        List all endpoints in a specific project.

        Parameters:
        project_name (str, optional): The name of the project. 
            If not provided, the project name from the configs file is used.
        verbose (bool, optional): If True, detailed information about each endpoint is returned. 
            If False, only basic information is returned.

        Returns:
        list[dict]: A list of dictionaries, where each dictionary represents an endpoint. 
            If verbose is True, each dictionary contains detailed information about the endpoint. 
            If verbose is False, each dictionary contains basic information about the endpoint.

        Raises:
        Exception: If the project does not exist or if there is an error in listing the endpoints.
        """

        # check if project exist
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            logging.info(f"Project '{project_name}' not found listing all user endpoints")

        list_endpoints_response = self.snsdk_client.list_endpoints(project=project_id)
        if list_endpoints_response['status_code'] == 200:
            endpoints = []
            for endpoint in list_endpoints_response['endpoints']:
                if verbose:
                    endpoints.append({k: v for k, v in endpoint.items()})
                else:
                    endpoints.append(
                        {
                            k: v
                            for k, v in endpoint.items()
                            if k
                            in [
                                'name',
                                'id',
                                'project_id',
                                'status',
                            ]
                        }
                    )
            return endpoints
        else:
            logging.error(f"Failed to list endpoints. Details: {list_endpoints_response['detail']}")
            raise Exception(f"Error message: {list_endpoints_response['detail']}")

    def create_endpoint(
        self,
        project_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        endpoint_description: Optional[str] = None,
        model_name: Optional[str] = None,
        instances: Optional[str] = None,
        rdu_arch: Optional[str] = None,
        hyperparams: Optional[str] = None,
    ) -> Optional[str]:
        """
        Creates a new endpoint in a specified Smabastudio project using a specified model.

        Parameters:
        - project_name (str, optional): The name of the project. 
            If not provided, the project name from the configuration is used.
        - endpoint_name (str, optional): The name of the endpoint. 
            If not provided, the endpoint name from the configuration is used.
        - endpoint_description (str, optional): The description of the endpoint. 
            If not provided, the endpoint description from the configuration is used.
        - model_name (str, optional): The name of the model. 
            If not provided, the model name from the configuration is used.
        - instances (str, optional): The number of instances for the endpoint. 
            If not provided, the endpoint instances from the configuration is used.
        - rdu_arch (str, optional): The RDU architecture for the endpoint. 
            If not provided, the RDU architecture from the configuration is used.
        - hyperparams (str, optional): The hyperparameters for the endpoint. 
            If not provided, the hyperparameters from the configuration is used.

        Raises:
        Exception: If the project does not exist
        Exeption: If the model does not exist
        Exception: If there is an error in creating the endpoint.

        Returns:
        - str: The ID of the created endpoint if successful or the endpoint already exists. 
            If unsuccessful, None is returned.
        """

        # check if project selected exists
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if model selected exists
        if model_name is None:
            model_name = self.config['model_checkpoint']['model_name']
        model_id = self.search_model(model_name=model_name)
        if model_id is None:
            raise Exception(f"Model with name '{model_name}' not found")

        # check if endpoint selected exists
        if endpoint_name is None:
            endpoint_name = self.config['endpoint']['endpoint_name']
        endpoint_id = self.search_endpoint(project=project_id, endpoint_name=endpoint_name)
        if endpoint_id is not None:
            logging.info(f"Endpoint with name '{endpoint_name}' not created it already exist with id {endpoint_id}")
            return endpoint_id

        create_endpoint_response = self.snsdk_client.create_endpoint(
            project=project_id,
            endpoint_name=endpoint_name,
            description=endpoint_description or self.config.get('endpoint', {}).get('endpoint_description'),
            model_checkpoint=model_id,
            instances=instances or self.config.get('endpoint', {}).get('endpoint_instances'),
            rdu_arch=rdu_arch or self.config['sambastudio']['rdu_arch'],
            hyperparams=json.dumps(hyperparams or self.config.get('endpoint', {}).get('hyperparams')),
        )

        if create_endpoint_response['status_code'] == 200:
            logging.info(f"Endpoint '{endpoint_name}' created")
            endpoint_id = create_endpoint_response['id']
            return endpoint_id

        else:
            logging.error(f'Failed to create endpoint {endpoint_name}. Details: {create_endpoint_response}')
            raise Exception(f'Error message: {create_endpoint_response}')

    def search_endpoint(self, project: Optional[str] = None, endpoint_name: Optional[str] = None) -> Optional[str]:
        """
        Search for an endpoint in a specified project by its name.

        Parameters:
        - project (str, optional): The name of the project. 
            If not provided, the project name from the configuration is used.
        - endpoint_name (str, optional): The name of the endpoint.
            If not provided, the endpoint name from the configuration is used.

        Raises:
        - Exception: if there is an error listing the endpoint

        Returns:
        - str: The ID of the endpoint if found. If not found or an error occurs, None is returned.
        """
        if project is None:
            project = self.config['project']['project_name']
        if endpoint_name is None:
            endpoint_name = self.config['endpoint']['endpoint_name']
        endpoint_info_response = self.snsdk_client.endpoint_info(project=project, endpoint=endpoint_name)

        if endpoint_info_response['status_code'] == 200:
            endpoint_id = endpoint_info_response['id']
            return endpoint_id
        elif endpoint_info_response['status_code'] == 404:
            logging.info(f"Endpoint with name '{endpoint_name}' not found in project '{project}'")
            return None
        else:
            logging.error(f'Failed to retrieve information for endpoint Details: {endpoint_info_response}')
            raise Exception(f'Error message: {endpoint_info_response}')

    def get_endpoint_details(self, project_name: Optional[str] = None, endpoint_name: Optional[str] = None) -> dict:
        """
        Retrieves the details of a specified endpoint in a given SambaStudio project.

        Parameters:
        - project_name (str, optional): The name of the project. 
            If not provided, the project name from the configs file is used.
        - endpoint_name (str, optional): The name of the endpoint. 
            If not provided, the endpoint name from the configs file is used.

        Raises:
        - Exception: If the project does not exist or if the endpoint does not exist.
        - Exception: If there is an error getting endpoint details

        Returns:
        - dict: Dictionary containing the endpoint's status, URL, and environment variables for the Langchain wrapper.
        """

        # check if project selected exists
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        if endpoint_name is None:
            endpoint_name = self.config['endpoint']['endpoint_name']

        endpoint_info_response = self.snsdk_client.endpoint_info(project=project_id, endpoint=endpoint_name)

        if endpoint_info_response['status_code'] == 200:
            endpoint_url = endpoint_info_response['url']
            endpoint_details = {
                'status': endpoint_info_response['status'],
                'url': endpoint_url,
                'langchain wrapper env': {
                    'SAMBASTUDIO_BASE_URL': self.snsdk_client.host_url,
                    'SAMBASTUDIO_BASE_URI': '/'.join(endpoint_url.split('/')[1:4]),
                    'SAMBASTUDIO_PROJECT_ID': endpoint_url.split('/')[-2],
                    'SAMBASTUDIO_ENDPOINT_ID': endpoint_url.split('/')[-1],
                    'SAMBASTUDIO_API_KEY': endpoint_info_response['api_key'],
                },
            }
            return endpoint_details
        else:
            logging.error(
                f"Failed to get details for endpoint '{endpoint_name}' in project '{project_name}'. Details: {endpoint_info_response}"
            )
            raise Exception(f'Error message: {endpoint_info_response}')

    def delete_endpoint(self, project_name: Optional[str] = None, endpoint_name: Optional[str] = None) -> None:
        """
        Deletes a specified endpoint in a given SambaStudio project.

        Parameters:
        - project_name (str, optional): The name of the project. 
            If not provided, the project name from the configs file is used.
        - endpoint_name (str, optional): The name of the endpoint. 
            If not provided, the endpoint name from the configs file is used.

        Raises:
        - Exception: If the project does not exist or if the endpoint does not exist.
        - Exception: If there is an error deleting the endpoint.

        Returns:
        - None
        """

        # check if project selected exists
        if project_name is None:
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if endpoint selected exists
        if endpoint_name is None:
            endpoint_name = self.config['endpoint']['endpoint_name']
        endpoint_id = self.search_endpoint(project=project_id, endpoint_name=endpoint_name)
        if endpoint_id is None:
            raise Exception(f"Endpoint with name '{endpoint_name}' not found in project '{project_name}'")

        delete_endpoint_response = self.snsdk_client.delete_endpoint(project=project_id, endpoint=endpoint_id)

        if delete_endpoint_response['status_code'] == 200:
            logging.info(f"Endpoint '{endpoint_name}' deleted in project '{project_name}'")

        else:
            logging.error(
                f"Failed to delete endpoint '{endpoint_name}' in project '{project_name}'. Details: {delete_endpoint_response}"
            )
            raise Exception(f'Error message: {delete_endpoint_response}')

    """datasets"""

    def search_dataset(self, dataset_name: Optional[str] = True) -> Optional[str]:
        """
        Searches for a dataset in SambaStudio by its name.

        Parameters:
        - dataset_name (str, optional): The name of the dataset to search for. 
            If not provided, the dataset name from the configs file is used.

        Returns:
        - str: The ID of the dataset if found. If not found, None is returned.
        """
        if dataset_name is None:
            dataset_name = self.config['dataset']['dataset_name']
        search_dataset_response = self.snsdk_client.search_dataset(dataset_name=dataset_name)
        if search_dataset_response['status_code'] == 200:
            dataset_id = search_dataset_response['data']['dataset_id']
            logging.info(f"Dataset with name '{dataset_name}' found with id {dataset_id}")
            return dataset_id
        else:
            logging.info(f"Dataset with name '{dataset_name}' not found")
            return None

    """list datasets"""
    # {dataset.get("dataset_name"):dataset.get("id") for dataset in sambastudio.list_datasets().get("datasets")}

    """create dataset"""
    # snapi

    """list apps"""
    # available_apps = {app.get("name"): app.get("id") for app in sambastudio.list_apps().get("apps")}

    """search app"""
    # define if is this needed

    """list tenants"""
    # TODO add later as util if dont known tenant by user before instantiation
