import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from snapi.snapi import USER_AGENT  # type: ignore
from snsdk import SnSdk  # type: ignore

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)
load_dotenv(os.path.join(repo_dir, '.env'), override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)

SNAPI_PATH = '~/.snapi'

# Job types: can not combine train/evaluation with batch_predict
JOB_TYPES = [
    'train',
    'evaluation',
    'batch_predict',
]

# TODO: future support for other types
SOURCE_TYPES = ['localMachine']
SOURCE_FILE_PATH = os.path.join(utils_dir, 'fine_tuning', 'src', 'tmp_source_file.json')


class SnsdkWrapper:
    """ "Wrapper around the SnSdk and SNAPI for E2E fine-tuning in SambaStudio"""

    """init"""

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Wrapper around the SnSdk and SNAPI for E2E fine-tuning in SambaStudio
        Args:
            config_path (str , optional): path to config path. Defaults to None.
            see a config file example in ./config.yaml
        """
        self.config_path = config_path
        self.config = None
        self.snapi_path = SNAPI_PATH

        # If config is provided, load it and validate Snapi directory path
        if self.config_path is not None:
            self.config = self._load_config(self.config_path)
            config_snapi_path = self.config['sambastudio']['snapi_path']
            if config_snapi_path is not None and len(config_snapi_path) > 0:
                self.snapi_path = self.config['sambastudio']['snapi_path']

        # Get sambastudio variables to set up Snsdk
        host_url, tenant_id, access_key = self._get_sambastudio_variables()

        self.snsdk_client = SnSdk(
            host_url=host_url,
            access_key=access_key,
            tenant_id=tenant_id,
            user_agent=USER_AGENT,
        )

        # Set up cookie to avoid auth issues
        self.snsdk_client.http_session.cookies.set('tenant', tenant_id)

    """Internal methods"""

    def _set_snapi_using_env_variables(
        self,
        host_name: str,
        access_key: str,
        current_snapi_config: Dict[Any, Any],
        snapi_config_path: str,
        snapi_secret_path: str,
        tenant_name: str = 'default',
    ) -> Tuple[str, str, str]:
        """Sets Snapi using env variables. It also validates if tenant can be set in Snapi config file.
        Args:
            host_name (str): host name coming from env variables
            access_key (str): access key coming from env variables
            tenant_name (str): tenant name coming from env variables
            current_snapi_config (dict): current snapi config dictionary
            snapi_config_path (str): snapi config path
            snapi_secret_path (str): snapi secret path

        Raises:
            Exception: fails to set the specified tenant using Snapi CLI

        Returns:
            tuple: host name, access key and tenant id
        """
        # Updates snapi config file using requested Sambastudio env
        tmp_snapi_config = {}
        tmp_snapi_config['HOST_NAME'] = host_name
        tmp_snapi_config['CONFIG_DIR'] = current_snapi_config['CONFIG_DIR']
        tmp_snapi_config['DISABLE_SSL_WARNINGS'] = current_snapi_config['DISABLE_SSL_WARNINGS']
        with open(snapi_config_path, 'w') as file:
            json.dump(tmp_snapi_config, file)
        with open(snapi_secret_path, 'w') as file:
            file.write(access_key)

        # Sets default requested tenant
        snapi_config_response = subprocess.run(
            ['snapi', 'config', 'set', '--tenant', f'{tenant_name}'],
            capture_output=True,
            text=True,
        )

        # If there's an error in Snapi subprocess, show it and stop process
        if (
            ('status_code' in snapi_config_response.stdout.lower())
            and ('error occured' in snapi_config_response.stdout.lower())
            or (len(snapi_config_response.stderr) > 0)
        ):
            if len(snapi_config_response.stderr) > 0:
                error_message = snapi_config_response.stderr
            else:
                error_search = re.search(r'message:\s*(.*)', snapi_config_response.stdout)
                if error_search:
                    error_message = error_search[0]
            logging.error(f"Failed to set tenant with name '{tenant_name}'. Details: {error_message}")
            raise Exception(f'Error message: {error_message}')

        # Read updated Snapi config file
        with open(snapi_config_path, 'r') as file:
            new_snapi_config = json.load(file)

        return host_name, new_snapi_config['TENANT_ID'], access_key

    def _get_sambastudio_variables(self) -> Tuple[str, str, str]:
        """Gets Sambastudio host name, tenant id and access key from Snapi folder location

        Raises:
            FileNotFoundError: raises error when the snapi config or secret file is not found
            ValueError: raises error when the snapi config file doesn't contain a correct json format

        Returns:
            tuple: host name, tenant id and access key from snapi setup
        """
        snapi_config = {}
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

        host_name = os.getenv('SAMBASTUDIO_HOST_NAME')
        access_key = os.getenv('SAMBASTUDIO_ACCESS_KEY')
        tenant_name = os.getenv('SAMBASTUDIO_TENANT_NAME')

        if (host_name is not None) and (access_key is not None) and (tenant_name is not None):
            logging.info(f'Using env variables to set up Snsdk.')
            host_name, tenant_id, access_key = self._set_snapi_using_env_variables(
                host_name,
                access_key,
                snapi_config,
                snapi_config_path,
                snapi_secret_path,
                tenant_name,
            )

        else:
            logging.info(f'Using variables from Snapi config to set up Snsdk.')
            host_name = snapi_config['HOST_NAME']
            assert host_name is not None
            tenant_id = snapi_config['TENANT_ID']
            assert tenant_id is not None
            access_key = snapi_secret

        if access_key is None:
            access_key = ''

        return host_name, tenant_id, access_key.strip()

    def _load_config(self, file_path: str) -> Dict[str, Any]:
        """Loads a YAML configuration file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: The configuration data loaded from the YAML file.
        """
        try:
            with open(file_path, 'r') as file:
                config: Dict[str, Any] = yaml.safe_load(file)
            logging.info(f'Using config file located in {file_path}')
        except FileNotFoundError:
            raise FileNotFoundError(f'Error: The file {file_path} does not exist.')
        except yaml.scanner.ScannerError:
            raise ValueError(f'Error: The file {file_path} contains invalid yaml.')
        return config

    def _raise_error_if_config_is_none(self) -> None:
        """Raise an error if the configuration file is not provided"""
        if self.config is None:
            error_message = 'No config found. Please provide parameter values.'
            logging.error(error_message)
            raise Exception(f'Error message: {error_message}')

    def _create_source_file(self, dataset_path: str) -> None:
        """
        Create a source file for snapi dataset upload

        Args:
            dataset_path string: path to dataset
        """
        json_content = {'source_path': dataset_path}
        with open(SOURCE_FILE_PATH, 'w') as file:
            json.dump(json_content, file)

    """tenants"""

    def list_tenants(self, verbose: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Lists all tenants

        Returns:
            list | None: list of existing tenants. If there's an error, None is returned.
        """
        list_tenant_response = self.snsdk_client.list_tenants()
        if list_tenant_response['status_code'] == 200:
            tenants = []
            if verbose:
                return list_tenant_response['data']
            else:
                for tenant in list_tenant_response['data']:
                    tenants.append(
                        {
                            'tenant_id': tenant.get('tenant_id'),
                            'tenant_name': tenant.get('tenant_name'),
                        }
                    )
                return tenants
        else:
            logging.error(f"Failed to list projects. Details: {list_tenant_response['detail']}")
            raise Exception(f"Error message: {list_tenant_response['detail']}")

    def search_tenant(self, tenant_name: Optional[str]) -> Optional[str]:
        """Searches tenant

        Args:
            tenant_name (str): tenant name to search

        Returns:
            str | None: searched tenant information. If there's an error, None is returned.
        """

        tenant_info_response = self.snsdk_client.tenant_info(tenant=tenant_name)
        if tenant_info_response['status_code'] == 200:
            tenant_id = tenant_info_response['data']['tenant_id']
            logging.info(f"Tenant with name '{tenant_name}' found with id {tenant_id}")
            return tenant_id
        else:
            logging.info(f"Tenant with name '{tenant_name}' not found")
            return None

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
            self._raise_error_if_config_is_none()
            project_name = self.config['project']['project_name']
        search_project_response = self.snsdk_client.search_project(project_name=project_name)
        if search_project_response['status_code'] == 200:
            project_id = search_project_response['data']['project_id']
            logging.info(f"Project with name '{project_name}' found with id {project_id}")
            return project_id
        else:
            logging.info(f"Project with name '{project_name}' not found")
            return None

    def create_project(
        self,
        project_name: Optional[str] = None,
        project_description: Optional[str] = None,
    ) -> str:
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
            self._raise_error_if_config_is_none()
            project_name = self.config['project']['project_name']
        if project_description is None:
            self._raise_error_if_config_is_none()
            project_description = self.config['project']['project_description']

        project_id = self.search_project(project_name)
        if project_id is None:
            create_project_response = self.snsdk_client.create_project(
                project_name=project_name, description=project_description
            )
            if create_project_response['status_code'] == 200:
                project_id = create_project_response['id']
                logging.info(f'Project with name {project_name} created with id {project_id}')
            else:
                logging.error(
                    f"Failed to create project with name '{project_name}'. Details: {create_project_response['detail']}"
                )
                raise Exception(f"Error message: {create_project_response['detail']}")
        else:
            logging.info(f"Project with name '{project_name}' already exists with id '{project_id}', using it")
        return project_id

    def list_projects(self, verbose: Optional[bool] = False) -> List[Dict[str, Any]]:
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
            for project in list_projects_response.get('projects'):
                if verbose:
                    projects.append({k: v for k, v in project.items()})
                else:
                    project_info = {k: v for k, v in project.items() if k in ['name', 'id', 'status']}
                    project_info['owner'] = project['metadata']['owner']
                    projects.append(project_info)
            return projects
        else:
            logging.error(f"Failed to list projects. Details: {list_projects_response['detail']}")
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
            self._raise_error_if_config_is_none()
            project_name = self.config['project']['project_name']

        # Check if the project exists
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

    """Dataset"""

    def list_datasets(self, verbose: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Lists all datasets

        Returns:
            list | None: list of existing datasets. If there's an error, None is returned.
        """
        list_datasets_response = self.snsdk_client.list_datasets()
        if list_datasets_response['status_code'] == 200:
            datasets = []
            if verbose:
                return list_datasets_response['datasets']
            else:
                for dataset in list_datasets_response['datasets']:
                    datasets.append(
                        {
                            'id': dataset.get('id'),
                            'dataset_name': dataset.get('dataset_name'),
                        }
                    )
                return datasets
        else:
            logging.error(f"Failed to list models. Details: {list_datasets_response['detail']}")
            raise Exception(f"Error message: {list_datasets_response['detail']}")

    def search_dataset(self, dataset_name: Optional[str] = None) -> Optional[str]:
        """Searches a dataset

        Args:
            dataset_name (str): dataset name to search

        Returns:
            str | None: searched dataset information. If there's an error, None is returned.
        """
        # Decide whether using method parameters or config
        if dataset_name is None:
            self._raise_error_if_config_is_none()
            dataset_name = self.config['dataset']['dataset_name']

        # Search dataset
        search_dataset_response = self.snsdk_client.search_dataset(dataset_name=dataset_name)
        if search_dataset_response['status_code'] == 200:
            dataset_id = search_dataset_response['data']['dataset_id']
            logging.info(f"Dataset with name '{dataset_name}' found with id {dataset_id}")
            return dataset_id
        else:
            logging.info(f"Dataset with name '{dataset_name}' not found")
            return None

    def delete_dataset(self, dataset_name: Optional[str] = None) -> None:
        # Decide whether using method parameters or config
        if dataset_name is None:
            self._raise_error_if_config_is_none()
            dataset_name = self.config['dataset']['dataset_name']

        # Search dataset if exists
        dataset_id = self.search_dataset(dataset_name=dataset_name)
        if dataset_id is None:
            raise Exception(f"Dataset with name '{dataset_name}' not found")

        # Delete dataset
        delete_dataset_response = self.snsdk_client.delete_dataset(dataset=dataset_name)
        if delete_dataset_response['status_code'] == 200:
            logging.info(f"Dataset with name '{dataset_name}' deleted")
        else:
            logging.error(f"Failed to delete dataset with name '{dataset_name}'. Details: {delete_dataset_response}")
            raise Exception(f'Error message: {delete_dataset_response}')

    def _build_snapi_dataset_add_command(
        self,
        dataset_name: str,
        dataset_apps_availability: List[str],
        dataset_job_types: List[str],
        dataset_source_type: str,
        dataset_description: str,
        dataset_filetype: str,
        dataset_url: str,
        dataset_language: str,
    ) -> List[str]:
        """Builds snapi command to add a dataset to SambaStudio.
        Addresses apps and job types, since they're lists
        Args:
            dataset_name (str): dataset name
            dataset_apps_availability (list): list of apps
            dataset_job_types (list): list of job types
            dataset_source_type (str): source type
            dataset_description (str): dataset description
            dataset_filetype (str): file type
            dataset_url (str): url
            dataset_language (str): language

        Returns:
            str: snapi command ready to execute
        """
        # Get multiple job type parameters
        job_type_command_parameters = []
        for job_type in dataset_job_types:
            job_type_command_parameters.append('--job_type')
            job_type_command_parameters.append(job_type)

        # Get multiple apps parameters
        apps_command_parameters = []
        for app in dataset_apps_availability:
            apps_command_parameters.append('--apps')
            apps_command_parameters.append(app)

        command = [
            'snapi',
            'dataset',
            'add',
            '--dataset-name',
            dataset_name,
            '--description',
            dataset_description,
            '--source_type',
            dataset_source_type,
            '--language',
            dataset_language,
            '--source_file',
            SOURCE_FILE_PATH,
            '--file_type',
            dataset_filetype,
            '--url',
            dataset_url,
        ]
        command.extend(job_type_command_parameters)
        command.extend(apps_command_parameters)

        return command

    def create_dataset(
        self,
        dataset_name: Optional[str] = None,
        dataset_apps_availability: Optional[List[str]] = None,
        dataset_job_types: Optional[List[str]] = None,
        dataset_source_type: Optional[str] = None,
        dataset_path: Optional[str] = None,  # add note in readme it must be absolute
        dataset_description: Optional[str] = None,
        dataset_filetype: Optional[str] = None,
        dataset_url: Optional[str] = None,
        dataset_language: Optional[str] = None,
        dataset_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """ """
        # Decide whether using method parameters or config
        if dataset_name is None:
            self._raise_error_if_config_is_none()
            dataset_name = self.config['dataset']['dataset_name']

        # Validate if apps exist
        if dataset_apps_availability is None:
            self._raise_error_if_config_is_none()
            dataset_apps_availability = self.config['dataset']['dataset_apps_availability']
        for app_name in dataset_apps_availability:
            app_id = self.search_app(app_name)
            if app_id is None:
                raise Exception(f"App '{app_name}' not found")

        # Validate job types
        if dataset_job_types is None:
            self._raise_error_if_config_is_none()
            dataset_job_types = self.config['dataset']['dataset_job_types']
        for job_type in dataset_job_types:
            if job_type not in JOB_TYPES:
                raise Exception(f"Job type '{job_type}' not valid")

        # Validate source type
        if dataset_source_type is None:
            self._raise_error_if_config_is_none()
            dataset_source_type = self.config['dataset']['dataset_source_type']
        if dataset_source_type not in SOURCE_TYPES:
            raise Exception(f"Source type '{dataset_source_type}' not valid")

        # Decide whether using method parameters or config
        if dataset_path is None:
            self._raise_error_if_config_is_none()
            dataset_path = self.config['dataset']['dataset_path']

        # Create source file based on dataset path
        self._create_source_file(dataset_path)

        # Decide whether using method parameters or config
        if dataset_description is None:
            self._raise_error_if_config_is_none()
            dataset_description = self.config['dataset']['dataset_description']

        if dataset_filetype is None:
            self._raise_error_if_config_is_none()
            dataset_filetype = self.config['dataset']['dataset_filetype']

        if dataset_url is None:
            self._raise_error_if_config_is_none()
            dataset_url = self.config['dataset']['dataset_url']

        if dataset_language is None:
            self._raise_error_if_config_is_none()
            dataset_language = self.config['dataset']['dataset_language']

        # TODO: Metadata WIP - waiting for channel's confirmation
        if dataset_metadata is None:
            self._raise_error_if_config_is_none()
            dataset_metadata = self.config['dataset']['dataset_metadata']
        # for job_type in dataset_job_types:
        #     if job_type == "batch_predict":
        #         raise Exception(
        #             f"Metadata is not valid for dataset with job type {job_type}"
        #         )

        # Validate if dataset already exists
        dataset_id = self.search_dataset(dataset_name)

        # Create dataset if dataset is not found
        if dataset_id is None:
            command = self._build_snapi_dataset_add_command(
                dataset_name,
                dataset_apps_availability,
                dataset_job_types,
                dataset_source_type,
                dataset_description,
                dataset_filetype,
                dataset_url,
                dataset_language,
            )
            echo_response = subprocess.run(['echo', 'yes'], capture_output=True, text=True)
            snapi_response = subprocess.run(command, input=echo_response.stdout, capture_output=True, text=True)

            errors_response = (
                ('status_code' in snapi_response.stdout.lower()) and ('error occured' in snapi_response.stdout.lower())
            ) or (len(snapi_response.stderr) > 0)
            # if errors coming in response
            if errors_response:
                if len(snapi_response.stderr) > 0:
                    error_message = snapi_response.stderr
                else:
                    error_search = re.search(r'message:\s*(.*)', snapi_response.stdout)
                    if error_search:
                        error_message = error_search[0]
                logging.error(f"Failed to create dataset with name '{dataset_name}'. Details: {error_message}")
                raise Exception(f'Error message: {error_message}')
            # if there are no errors in reponse
            else:
                dataset_id = self.search_dataset(dataset_name=dataset_name)
                logging.info(f"Dataset with name '{dataset_name}' created: '{snapi_response.stdout}'")
                return dataset_id
        # Dataset found, so return dataset id
        else:
            logging.info(f"Dataset with name '{dataset_name}' already exists with id '{dataset_id}', using it")
        return dataset_id

    """app"""

    def list_apps(self, verbose: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Lists all apps

        Returns:
            list (optional) : list of existing apps. If there's an error, None is returned.
        """
        list_apps_response = self.snsdk_client.list_apps()
        if list_apps_response['status_code'] == 200:
            apps = []
            if verbose:
                apps = list_apps_response['apps']
            else:
                for app in list_apps_response['apps']:
                    apps.append({'id': app.get('id'), 'name': app.get('name')})
            return apps
        else:
            logging.error(f"Failed to list models. Details: {list_apps_response['detail']}")
            raise Exception(f"Error message: {list_apps_response['detail']}")

    def search_app(self, app_name: str) -> Optional[str]:
        """Searches an App

        Args:
            app_name (str): app name to search

        Returns:
            str (optional): searched app information. If there's an error, None is returned.
        """
        app_info_response = self.snsdk_client.app_info(app=app_name)
        if app_info_response['status_code'] == 200:
            app_id = app_info_response['apps']['id']
            logging.info(f"App with name '{app_name}' found with id {app_id}")
            return app_id
        else:
            logging.info(f"App with name '{app_name}' not found")
            return None

    """models"""

    def model_info(self, model_name: Optional[str] = None, job_type: Optional[str] = None) -> Dict[str, Any]:
        """Gets model info based on the job type specified.
        Several fields are pulled that describe different aspects of model.

        Args:
            model_name (Optional[str], optional): model name. Defaults to None.
            job_type (Optional[str], optional): job type. Defaults to None.

        Raises:
            Exception: If there is an error in getting model info.

        Returns:
            dict: dictionary containing model info.
        """
        # Decide whether using method parameters or config
        if model_name is None:
            self._raise_error_if_config_is_none()
            model_name = self.config['job']['model']
        if job_type is None:
            self._raise_error_if_config_is_none()
            job_type = self.config['job']['job_type']

        # Get model's info
        model_info_response = self.snsdk_client.model_info(model=model_name, job_type=job_type)
        if model_info_response['status_code'] == 200:
            return model_info_response
        else:
            logging.error(f"Failed to get model's info. Details: {model_info_response['message']}")
            raise Exception(f"Error message: {model_info_response['message']}")

    def list_models(
        self,
        filter_job_types: Optional[List[str]] = [],
        verbose: Optional[bool] = False,
    ) -> List[Dict[str, Any]]:
        """
        List models in sambastudio based on the provided filter.

        Parameters:
        filter_job_types (list[str], optional): A list of job types to filter the models. Defaults to [].
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
                if filter_job_types is None:
                    filter_job_types = []
                if set(filter_job_types).issubset(model.get('jobTypes')):
                    if verbose:
                        models.append({k: v for k, v in model.items()})
                    else:
                        models.append(
                            {k: v for k, v in model.items() if k in ['model_checkpoint_name', 'model_id', 'version']}
                        )
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
            self._raise_error_if_config_is_none()
            model_name = self.config['job']['model']

        search_model_response = self.snsdk_client.search_model(model_name=model_name)
        if search_model_response['status_code'] == 200:
            model_id = search_model_response['data']['model_id']
            logging.info(f"Model with name '{model_name}' found with id {model_id}")
            return model_id
        else:
            logging.info(f"Model with name '{model_name}' not found")
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
            self._raise_error_if_config_is_none()
            model_name = self.config['job']['model']

        models = self.list_models(filter_job_types=['train'])
        model_id = [model['model_id'] for model in models if model['model_checkpoint_name'] == model_name]
        if len(model_id) > 0:
            logging.info(f"Model '{model_name}' with id '{model_id[0]}' available for training and deployment found")
            return model_id[0]
        else:
            logging.info(f"Model '{model_name}' available for training and deployment not found")
            return None

    def delete_model(self, model_name: Optional[str] = None) -> None:
        """
        Deletes a model from the SambaStudio.

        Parameters:
        model_name (str, optional): The name of the model to be deleted.
            If not provided, the name from the configs file will be used.

        Returns:
        None

        Raises:
        Exception: If the model is not found or if there is an error in deleting the model.
        """
        if model_name is None:
            self._raise_error_if_config_is_none()
            model_name = self.config['model_checkpoint']['model_name']

        # Check if the model exists
        model_id = self.search_model(model_name=model_name)
        if model_id is None:
            raise Exception(f"Model with name '{model_name}' not found")

        delete_model_response = self.snsdk_client.delete_model(model=model_id)
        if delete_model_response['status_code'] == 200:
            logging.info(f"Model with name '{model_name}' deleted")
        else:
            logging.error(f"Failed to delete model with name or id '{model_name}'. Details: {delete_model_response}")
            raise Exception(f'Error message: {delete_model_response}')

    """Job"""

    def get_default_hyperparms(
        self, model_name: Optional[str] = None, job_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get hyperparameters required to run a job in a SambaStudio model.

        Parameters:
        model_name (str, optional): The name of the model to check hyperparameters.
            If not provided, the name from the configs file will be used.
        job_type (str, optional): The job type to check hyperparameters.
            If not provided, the job_Type from the configs file will be used.

        Returns:
        architecture_hyper_params (dict): list of hyper parameters per RDU
        available architecture

        Raises:
        Exception: If the model or job type is not found or if there is an error
        getting the hyperparameters.
        """
        if model_name is None:
            self._raise_error_if_config_is_none()
            model_name = self.config['model_checkpoint']['model_name']
        if job_type is None:
            self._raise_error_if_config_is_none()
            job_type = self.config['job']['job_type']
        model_info_response = self.snsdk_client.model_info(model_name, job_type)
        if model_info_response['status_code'] == 200:
            hyperparams = model_info_response.get('hyperparams').get(job_type)
            architecture_hyper_params = {}
            user_hyperparams_list = []
            for architecture, params in hyperparams.items():
                for param in params['user_params']:
                    param_dict = {
                        'field_name': param.get('FIELD_NAME'),
                        'description': param.get('DESCRIPTION'),
                        'settings': param.get('TYPE_SPECIFIC_SETTINGS').get(job_type),
                        'constrains': param.get('CONSTRAINTS'),
                    }
                    user_hyperparams_list.append(param_dict)
                architecture_hyper_params[architecture] = user_hyperparams_list
                logging.info(
                    f'Default Hyperparameters for {job_type} in {architecture} for {model_name}: \n'
                    f'''
                    {[
                        param["field_name"]+":`"+param["settings"]["DEFAULT"]+"`" 
                        for param 
                        in user_hyperparams_list
                    ]}\n
                    '''
                )
            return architecture_hyper_params
        else:
            logging.error(
                f'Failed to get "{job_type}" job hyperparameters for {model_name}. Details: {model_info_response}'
            )
            raise Exception(f'Error message: {model_info_response}')

    def run_training_job(
        self,
        project_name: Optional[str] = None,
        job_name: Optional[str] = None,
        job_description: Optional[str] = None,
        job_type: Optional[str] = None,
        model: Optional[str] = None,
        model_version: Optional[int] = None,
        dataset_name: Optional[str] = None,
        parallel_instances: Optional[int] = None,
        load_state: Optional[bool] = None,
        sub_path: Optional[str] = None,
        rdu_arch: Optional[str] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
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
            model_version (int, optional).  the version of the model to fine tune.
                If not provided, the model version from the configs file will be used.
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
            self._raise_error_if_config_is_none()
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if model selected is found and is trainable
        if model is None:
            self._raise_error_if_config_is_none()
            model = self.config['job']['model']
        model_id = self.search_trainable_model(model)
        if model_id is None:
            raise Exception(f"model with name '{model}' not found")

        # check if dataset exist
        if dataset_name is None:
            self._raise_error_if_config_is_none()
            dataset_name = self.config['job'].get('dataset_name')
            if dataset_name is None:
                dataset_name = self.config['dataset']['dataset_name']
        dataset_id = self.search_dataset(dataset_name)
        if dataset_id is None:
            raise Exception(f"dataset with name '{dataset_name}' not found")

        # check extra params passed or in config file
        if job_name is None:
            self._raise_error_if_config_is_none()
            job_name = self.config['job']['job_name']
        if model_version is None:
            self._raise_error_if_config_is_none()
            model_version = self.config['job']['model_version']
        if job_description is None:
            self._raise_error_if_config_is_none()
            job_description = self.config['job']['job_description']
        if job_type is None:
            self._raise_error_if_config_is_none()
            job_type = self.config['job']['job_type']
        if parallel_instances is None:
            self._raise_error_if_config_is_none()
            parallel_instances = self.config['job']['parallel_instances']
        if load_state is None:
            self._raise_error_if_config_is_none()
            load_state = self.config['job']['load_state']
        if sub_path is None:
            self._raise_error_if_config_is_none()
            sub_path = self.config['job']['sub_path']
        if rdu_arch is None:
            self._raise_error_if_config_is_none()
            rdu_arch = self.config['sambastudio']['rdu_arch']
        if hyperparams is None:
            self._raise_error_if_config_is_none()
            hyperparams = self.config['job']['hyperparams']

        # create job
        create_job_response = self.snsdk_client.create_job(
            project=project_id,
            job_name=job_name,
            description=job_description,
            job_type=job_type,
            model_checkpoint=model_id,
            model_version=model_version,
            dataset=dataset_id,
            parallel_instances=parallel_instances,
            load_state=load_state,
            sub_path=sub_path,
            rdu_arch=rdu_arch,
            hyperparams=json.dumps(hyperparams),
        )

        if create_job_response['status_code'] == 200:
            job_id = create_job_response['job_id']
            logging.info(f"Job with name '{job_name}' created: '{create_job_response}'")
            return job_id
        else:
            logging.error(f"Failed to create job with name '{job_name}'. Details: {create_job_response}")
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
            self._raise_error_if_config_is_none()
            job_name = self.config['job']['job_name']

        # check if project selected exists
        if project_name is None:
            self._raise_error_if_config_is_none()
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

    def check_job_progress(
        self,
        project_name: Optional[str] = None,
        job_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        wait: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Check the progress of a job in a specific SambaStudio project.

        Parameters:
        project_name (str, optional): The name of the project.
            If not provided, the project name from the configs file is used.
        job_name (str, optional): The name of the job.
            If not provided, the job name from the configs file is used.
        verbose: (bool, optional): wether to return or not full job progress status
        wait: bool, optional): if true the command will loop until job status is completed

        Returns:
        dict: A dictionary containing the job progress status.

        Raises:
        Exception: If the project or job is not found.
        """

        # check if project selected exists
        if project_name is None:
            self._raise_error_if_config_is_none()
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if job selected exists
        if job_name is None:
            self._raise_error_if_config_is_none()
            job_name = self.config['job']['job_name']
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")

        # check job progress
        while True:
            check_job_progress_response = self.snsdk_client.job_info(project=project_id, job=job_id)

            if check_job_progress_response['status_code'] == 200:
                if verbose:
                    job_progress = {k: v for k, v in check_job_progress_response['data'].items()}
                else:
                    job_progress = {
                        k: v
                        for k, v in check_job_progress_response['data'].items()
                        if k
                        in [
                            'job_name',
                            'job_id',
                            'job_type',
                            'status',
                            'time_created',
                        ]
                    }
                if job_progress.get('status') == 'EXIT_WITH_0':
                    job_progress['status'] = 'Completed'
                logging.info(f'Job `{job_name}` with progress status: {job_progress["status"]}')
            else:
                logging.error(f'Failed to check job progress. Details: {check_job_progress_response}')
                raise Exception(f'Error message: {check_job_progress_response}')

            if wait == False:
                break
            else:
                if job_progress['status'] == 'Completed':
                    break
                elif job_progress['status'] in ['EXIT_WITH_1', 'FAILED']:
                    logging.error(f'Job failed. Details: {job_progress}')
                    raise Exception(f'Job failed. Details: {job_progress}')
                time.sleep(60)

        return job_progress

    def list_jobs(self, project_name: Optional[str] = None, verbose: Optional[bool] = False) -> List[Dict[str, Any]]:
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
            self._raise_error_if_config_is_none()
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
                            if k
                            in [
                                'job_name',
                                'job_id',
                                'job_type',
                                'project_id',
                                'status',
                            ]
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
            self._raise_error_if_config_is_none()
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if job selected exists
        if job_name is None:
            self._raise_error_if_config_is_none()
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
                f"Failed to delete job with name '{job_name}' in project '{project_name}'.\
                    Details: {delete_job_response}"
            )
            raise Exception(f'Error message: {delete_job_response}')

    """checkpoints"""

    def list_checkpoints(
        self,
        project_name: Optional[str] = None,
        job_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        sort: Optional[bool] = False,
    ) -> List[Dict[str, Any]]:
        """
        List all checkpoints in a specific job within a SambaStudio project.

        Parameters:
        project_name (str, optional): The name of the project.
            If not provided, the project name from the configs file is used.
        job_name (str, optional): The name of the job. If not provided, the job name from the configs file is used.
        verbose (bool, optional): If True, detailed information about each checkpoint is returned.
            If False, only basic information is returned.
        sort (bool, optional): If True return list will be sorted by train_loss (less training loss first)

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
            self._raise_error_if_config_is_none()
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if job selected exists
        if job_name is None:
            self._raise_error_if_config_is_none()
            job_name = self.config['job']['job_name']
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")

        list_checkpoints_response = self.snsdk_client.list_checkpoints(project=project_id, job=job_id)
        if list_checkpoints_response['status_code'] == 200:
            checkpoints = []
            for checkpoint in list_checkpoints_response['data']['checkpoints']:
                if verbose or sort:
                    checkpoints.append({k: v for k, v in checkpoint.items()})
                else:
                    checkpoints.append(
                        {k: v for k, v in checkpoint.items() if k in ['checkpoint_name', 'checkpoint_id']}
                    )
            if sort:
                return sorted(checkpoints, key=lambda x: x['metrics']['single_value']['train_loss'])
            else:
                return checkpoints
        else:
            logging.error(f'Failed to list checkpoints. Details: {list_checkpoints_response}')
            raise Exception(f'Error message: {list_checkpoints_response}')

    def promote_checkpoint(
        self,
        checkpoint_name: Optional[str] = None,
        project_name: Optional[str] = None,
        job_name: Optional[str] = None,
        model_name: Optional[str] = None,
        model_description: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> str:
        """
        Promotes a model checkpoint from a specific training job to a model in SambaStudio model hub.

        Parameters:
        - checkpoint_name (str, optional): The name of the checkpoint to be promoted as model.
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
            self._raise_error_if_config_is_none()
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if job selected exists
        if job_name is None:
            job_name = self.config['job']['job_name']
            self._raise_error_if_config_is_none()
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")

        if checkpoint_name is None:
            self._raise_error_if_config_is_none()
            checkpoint_name = self.config['model_checkpoint']['checkpoint_name']
            if not checkpoint_name:
                raise Exception('No model checkpoint_id provided')
            # TODO: check if checkpoint in list checkpoints list blocked because authorization error
            # in lists checkpoints method
            # if checkpoint_id not in self.list_checkpoints(project_name=project_name, job_name=job_name):
            #     raise Exception(f"Checkpoint id '{checkpoint_id}' not found in job '{job_name}'")

        # check extra params passer or config file passed
        if model_name is None:
            self._raise_error_if_config_is_none()
            model_name = self.config['model_checkpoint']['model_name']
        if model_description is None:
            self._raise_error_if_config_is_none()
            model_description = self.config['model_checkpoint']['model_description']
        if model_type is None:
            self._raise_error_if_config_is_none()
            model_type = self.config['model_checkpoint']['model_type']

        add_model_response = self.snsdk_client.add_model(
            project=project_name,
            job=job_name,
            model_checkpoint=checkpoint_name,
            model_checkpoint_name=model_name,
            description=model_description,
            checkpoint_type=model_type,
        )

        if add_model_response['status_code'] == 200:
            logging.info(f"Model checkpoint '{checkpoint_name}' promoted to model '{model_name}'")
            return add_model_response['data']['model_id']
        else:
            logging.error(f"Failed to promote checkpoint '{checkpoint_name}' to model. Details: {add_model_response}")
            raise Exception(f'Error message: {add_model_response}')

    def delete_checkpoint(self, checkpoint: Optional[str] = None) -> None:
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
            self._raise_error_if_config_is_none()
            checkpoint = self.config['model_checkpoint']['model_checkpoint_id']
            if not checkpoint:
                raise Exception('No model checkpoint_id provided')

        delete_checkpoint_response = self.snsdk_client.delete_checkpoint(checkpoint=checkpoint)
        if delete_checkpoint_response['status_code'] == 200:
            logging.info(f"Model checkpoint '{checkpoint}' deleted")
        else:
            logging.error(f"Failed to delete checkpoint '{checkpoint}'. Details: {delete_checkpoint_response}")
            raise Exception(f'Error message: {delete_checkpoint_response}')

    def create_composite_model(
        self,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        model_list: Optional[List[str]] = None,
        rdu_required: Optional[int] = None,
    ) -> Optional[str]:
        """Create a composite model in SambaStudio

        Parameters:
        - model_name (str, optional): name of the composite model.
            If not provided, the project name from the configuration is used.
        - description (str, optional): description of the composite model. Defaults to None.
            If not provided, the project name from the configuration is used.
        - model_list (List[str], optional): list of models to include in the composite model.
            If not provided, the models from the configuration are used.
        - rdu_required (int, optional): minimum required RDU.
            If not provided, the models from the configuration are used.

        Raises:
            Exception: If one or more models on list does not exist.
            Exception: If there is an error in creating the composite model.

        Returns:
        - str: The ID of the created composite models if successful or the composite model already exists.
            If unsuccessful, None is returned.
        """

        if model_name is None:
            self._raise_error_if_config_is_none()
            model_name = self.config['composite_model']['model_name']

        if model_list is None:
            self._raise_error_if_config_is_none()
            model_list = self.config['composite_model']['model_list']

        if description is None:
            self._raise_error_if_config_is_none()
            description = self.config['composite_model']['description']

        if rdu_required is None:
            self._raise_error_if_config_is_none()
            rdu_required = self.config['composite_model']['rdu_required']

        # check if selected composite model doesn't exist already
        model_id = self.search_model(model_name=model_name)
        if model_id is None:
            # check if listed models to include in composite exist
            model_ids = []
            for model in model_list:
                model_id = self.search_model(model_name=model)
                if model_id is not None:
                    model_ids.append(model_id)
                else:
                    raise Exception(f"Model with name '{model}' does not exist.")
            logging.info(f"Models to include in composite found with ids '{list(zip(model_list,  model_ids))}")

            # create composite model
            dependencies = [{'name': model} for model in model_list]
            create_composite_model_response = self.snsdk_client.add_composite_model(
                name=model_name, description=description, dependencies=dependencies, rdu_required=rdu_required
            )

            if create_composite_model_response['status_code'] == 200:
                model_id = create_composite_model_response['model_id']
                logging.info(f'Composite model with name {model_name} created with id {model_id}')
            else:
                logging.error(
                    f'Failed to create composite model with name "{model_name}".'
                    f'Message: {create_composite_model_response["message"]}.'
                    f'Details: {create_composite_model_response["details"]}'
                )
                raise Exception(f"Error message: {create_composite_model_response['details']}")

        # if selected composite model already exists
        else:
            logging.info(f"Model with name '{model_name}' not created it already exist with id {model_id}")

        return model_id

    """endpoint"""

    def list_endpoints(
        self, project_name: Optional[str] = None, verbose: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
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
            self._raise_error_if_config_is_none()
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
        model_version: Optional[str] = None,
        instances: Optional[int] = None,
        rdu_arch: Optional[str] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Creates a new endpoint in a specified Sambastudio project using a specified model.

        Parameters:
        - project_name (str, optional): The name of the project.
            If not provided, the project name from the configuration is used.
        - endpoint_name (str, optional): The name of the endpoint.
            If not provided, the endpoint name from the configuration is used.
        - endpoint_description (str, optional): The description of the endpoint.
            If not provided, the endpoint description from the configuration is used.
        - model_name (str, optional): The name of the model.
            If not provided, the model name from the configuration is used.
        - model_version (str, optional): The version of the model.
        - instances (int, optional): The number of instances for the endpoint.
            If not provided, the endpoint instances from the configuration is used.
        - rdu_arch (str, optional): The RDU architecture for the endpoint.
            If not provided, the RDU architecture from the configuration is used.
        - hyperparams (dict, optional): The hyperparameters for the endpoint.
            If not provided, the hyperparameters from the configuration is used.

        Raises:
        Exception: If the project does not exist
        Exception: If the model does not exist
        Exception: If there is an error in creating the endpoint.

        Returns:
        - str: The ID of the created endpoint if successful or the endpoint already exists.
            If unsuccessful, None is returned.
        """

        # check if project selected exists
        if project_name is None:
            self._raise_error_if_config_is_none()
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if model selected exists
        if model_name is None:
            self._raise_error_if_config_is_none()
            if self.config.get('composite_model') is not None:
                model_name = self.config['composite_model']['model_name']
            else:
                model_name = self.config['model_checkpoint']['model_name']
        model_id = self.search_model(model_name=model_name)
        if model_id is None:
            raise Exception(f"Model with name '{model_name}' not found")

        # check if endpoint selected exists
        if endpoint_name is None:
            self._raise_error_if_config_is_none()
            endpoint_name = self.config['endpoint']['endpoint_name']
        endpoint_id = self.search_endpoint(project=project_id, endpoint_name=endpoint_name)
        if endpoint_id is not None:
            logging.info(f"Endpoint with name '{endpoint_name}' not created it already exist with id {endpoint_id}")
            return endpoint_id

        # check extra params passed or config file passed
        if self.config is not None:
            if (model_version := self.config.get('model_checkpoint', {}).get('model_version')) is None:
                model_version = model_version or '1'
        else:
            model_version = model_version or '1'
        if endpoint_description is None:
            self._raise_error_if_config_is_none()
            endpoint_description = self.config['endpoint']['endpoint_description']
        if instances is None:
            self._raise_error_if_config_is_none()
            instances = self.config['endpoint']['endpoint_instances']
        if rdu_arch is None:
            self._raise_error_if_config_is_none()
            rdu_arch = self.config['sambastudio']['rdu_arch']
        if hyperparams is None:
            self._raise_error_if_config_is_none()
            hyperparams = self.config['endpoint']['hyperparams']

        # create endpoint
        create_endpoint_response = self.snsdk_client.create_endpoint(
            project=project_id,
            endpoint_name=endpoint_name,
            description=endpoint_description,
            model_checkpoint=model_id,
            model_version=model_version,
            instances=instances,
            rdu_arch=rdu_arch,
            hyperparams=json.dumps(hyperparams),
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
            self._raise_error_if_config_is_none()
            project = self.config['project']['project_name']
        if endpoint_name is None:
            self._raise_error_if_config_is_none()
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

    def get_endpoint_details(
        self, project_name: Optional[str] = None, endpoint_name: Optional[str] = None
    ) -> Dict[str, Any]:
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
        - dict: Dictionary containing the endpoint's status, and environment variables
            for using the model with langchain wrappers.
        """

        # check if project selected exists
        if project_name is None:
            self._raise_error_if_config_is_none()
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        if endpoint_name is None:
            self._raise_error_if_config_is_none()
            endpoint_name = self.config['endpoint']['endpoint_name']

        endpoint_info_response = self.snsdk_client.endpoint_info(project=project_id, endpoint=endpoint_name)

        if endpoint_info_response['status_code'] == 200:
            endpoint_url = endpoint_info_response['url']
            endpoint_details = {
                'status': endpoint_info_response['status'],
                'url': endpoint_url,
                'langchain_wrapper_env': {
                    'SAMBASTUDIO_URL': self.snsdk_client.host_url + endpoint_url,
                    'SAMBASTUDIO_API_KEY': endpoint_info_response['api_key'],
                },
            }
            return endpoint_details
        else:
            logging.error(
                f"Failed to get details for endpoint '{endpoint_name}' in project '{project_name}'.\
                    Details: {endpoint_info_response}"
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
            self._raise_error_if_config_is_none()
            project_name = self.config['project']['project_name']
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")

        # check if endpoint selected exists
        if endpoint_name is None:
            self._raise_error_if_config_is_none()
            endpoint_name = self.config['endpoint']['endpoint_name']
        endpoint_id = self.search_endpoint(project=project_id, endpoint_name=endpoint_name)
        if endpoint_id is None:
            raise Exception(f"Endpoint with name '{endpoint_name}' not found in project '{project_name}'")

        delete_endpoint_response = self.snsdk_client.delete_endpoint(project=project_id, endpoint=endpoint_id)

        if delete_endpoint_response['status_code'] == 200:
            logging.info(f"Endpoint '{endpoint_name}' deleted in project '{project_name}'")

        else:
            logging.error(
                f"Failed to delete endpoint '{endpoint_name}' in project '{project_name}'.\
                    Details: {delete_endpoint_response}"
            )
            raise Exception(f'Error message: {delete_endpoint_response}')
