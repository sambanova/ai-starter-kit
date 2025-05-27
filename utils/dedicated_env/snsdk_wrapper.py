import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import jinja2
import yaml
from dotenv import load_dotenv
from jinja2 import meta
from jinja2.sandbox import ImmutableSandboxedEnvironment
from packaging import version

from snapi.snapi import USER_AGENT  # type: ignore
from snsdk import SnSdk  # type: ignore

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

SNAPI_PATH = '~/.snapi'

# Job types: can not combine train/evaluation with batch_predict
JOB_TYPES = [
    'train',
    'evaluation',
    'batch_predict',
]

# TODO: future support for other types
SOURCE_TYPES = ['localMachine']
SOURCE_FILE_PATH = os.path.join(current_dir, 'tmp_source_file.json')


class SnsdkWrapper:
    """ "Wrapper around the SnSdk and SNAPI for E2E fine-tuning in SambaStudio"""

    """init"""

    def __init__(self, config_path: Optional[str] = None, verbose: Optional[bool] = True) -> None:
        """Wrapper around the SnSdk and SNAPI for E2E fine-tuning in SambaStudio
        Args:
            config_path (str , optional): path to config path. Defaults to None.
                    see a config file example in ./config.yaml
            verbose (bool): show informational logs
        """
        self.config_path = config_path
        self.config = None
        self.snapi_path = SNAPI_PATH

        # If config is provided, load it and validate Snapi directory path
        if self.config_path is not None:
            self.config = self._load_config(self.config_path, verbose)
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

        # check .snapi folder exist if not create it, then write config and secret
        os.makedirs(os.path.dirname(snapi_config_path), exist_ok=True)

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
                # if all lines in stderr are warnings don't raise
                if all('warning' in line.lower() for line in error_message.splitlines()):
                    logging.warning(f"Tenant '{tenant_name}' set with warnings: {error_message}")
                else:
                    logging.error(f"Failed to set tenant with name '{tenant_name}'. Details: {error_message}")
                    raise Exception(f'Error message: {error_message}')
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

    def _get_sambastudio_variables(self, verbose: Optional[bool] = True) -> Tuple[str, str, str]:
        """Gets Sambastudio host name, tenant id and access key from environment or Snapi folder location
        Args:
            verbose (bool): show informational logs
        Raises:
            FileNotFoundError: raises error when the snapi config or secret file is not found
            ValueError: raises error when the snapi config file doesn't contain a correct json format

        Returns:
            tuple: host name, tenant id and access key from snapi setup
        """
        snapi_config_base = {'CONFIG_DIR': './', 'DISABLE_SSL_WARNINGS': 'false'}
        snapi_secret = ''

        host_name = os.getenv('SAMBASTUDIO_HOST_NAME')
        access_key = os.getenv('SAMBASTUDIO_ACCESS_KEY')
        tenant_name = os.getenv('SAMBASTUDIO_TENANT_NAME')

        snapi_config_path = os.path.expanduser(self.snapi_path) + '/config.json'
        snapi_secret_path = os.path.expanduser(self.snapi_path) + '/secret.txt'

        # if environment variables set, .snapi folder is created or overwritten
        if (host_name is not None) and (access_key is not None) and (tenant_name is not None):
            if verbose:
                logging.info(f'Using env variables to set up SNSDK and SNAPI.')
            host_name, tenant_id, access_key = self._set_snapi_using_env_variables(
                host_name=host_name,
                access_key=access_key,
                current_snapi_config=snapi_config_base,
                snapi_config_path=snapi_config_path,
                snapi_secret_path=snapi_secret_path,
                tenant_name=tenant_name,
            )

        # in other case try to get the current .snapi folder data
        else:
            try:
                # reads snapi config json
                with open(snapi_config_path, 'r') as file:
                    snapi_config = json.load(file)

                # reads snapi secret txt file
                with open(snapi_secret_path, 'r') as file:
                    snapi_secret = file.read()

            except FileNotFoundError:
                raise FileNotFoundError(f'Error: The file {snapi_config_path} does not exist.')
            except ValueError:
                raise ValueError(f'Error: The file {snapi_config_path} contains invalid JSON.')

            logging.info(f'Using variables from .snapi config to set up Snsdk.')

            host_name = snapi_config['HOST_NAME']
            assert host_name is not None
            tenant_id = snapi_config['TENANT_ID']
            assert tenant_id is not None
            access_key = snapi_secret

        if access_key is None:
            access_key = ''

        return host_name, tenant_id, access_key.strip()

    def _load_config(self, file_path: str, verbose: Optional[bool] = True) -> Dict[str, Any]:
        """Loads a YAML configuration file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: The configuration data loaded from the YAML file.
        """
        try:
            with open(file_path, 'r') as file:
                config: Dict[str, Any] = yaml.safe_load(file)
            if verbose:
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
            logging.error(f'Failed to list projects. Details: {list_tenant_response["detail"]}')
            raise Exception(f'Error message: {list_tenant_response["detail"]}')

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
                raise Exception(f'Error message: {create_project_response["detail"]}')
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
            logging.error(f'Failed to list projects. Details: {list_projects_response["detail"]}')
            logging.error(f'Failed to list projects. Details: {list_projects_response["detail"]}')
            raise Exception(f'Error message: {list_projects_response["detail"]}')

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
            logging.error(f'Failed to list models. Details: {list_datasets_response["detail"]}')
            raise Exception(f'Error message: {list_datasets_response["detail"]}')

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
        datasets=self.list_datasets()
        if dataset_name in [dataset["dataset_name"] for dataset in datasets]:
            dataset_id = [dataset["id"] for dataset in datasets if dataset["dataset_name"]==dataset_name][0]
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
            logging.info(f'running snapi add dataset command:\n {" ".join(command)}\nThis could take a while')
            echo_response = subprocess.run(['echo', 'yes'], capture_output=True, text=True)
            snapi_response = subprocess.run(command, input=echo_response.stdout, capture_output=True, text=True)

            errors_response = (
                ('status_code' in snapi_response.stdout.lower()) and ('error occured' in snapi_response.stdout.lower())
            ) or (len(snapi_response.stderr) > 0)
            # if errors coming in response
            if errors_response:
                if len(snapi_response.stderr) > 0:
                    error_message = snapi_response.stdout + snapi_response.stderr
                    # if all lines in stderr are warnings dont raise
                    if all('warning' in line.lower() for line in error_message.splitlines()):
                        logging.warning(f"dataset with name '{dataset_name} created with warnings: {error_message}")
                    else:
                        logging.error(f"Failed to create dataset with name '{dataset_name}'. Details: {error_message}")
                        raise Exception(f'Error message: {error_message}')
                else:
                    error_search = re.search(r'message:\s*(.*)', snapi_response.stdout)
                    error_message = snapi_response.stdout + snapi_response.stderr
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
            logging.error(f'Failed to list models. Details: {list_apps_response["detail"]}')
            raise Exception(f'Error message: {list_apps_response["detail"]}')

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
    
    """app - BYOC"""
    
    def get_suitable_apps(
        self, checkpoints: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None, verbose: Optional[bool] = True
    ) -> List[List[Dict[str, Any]]]:
        """
        find suitable sambastudio apps for the given checkpoints

        Parameters:
            - checkpoints (list of dict or dict, optional): checkpoints.
                if not set checkpoints in config.yaml file will be used
            - verbose (bool): show informational logs
        """
        if isinstance(checkpoints, dict):
            checkpoints = [checkpoints]
        if checkpoints is None:
            self._raise_error_if_config_is_none()
            checkpoints = self.config['checkpoints']

        sstudio_models = self.list_models(verbose=True)

        checkpoints_suitable_apps = []

        for checkpoint in checkpoints:
            assert isinstance(checkpoint, dict)
            suitable_apps: Set[str] = set()
            for model in sstudio_models:
                params = model.get('params', {})
                if params is not None:
                    model_params = params.get('invalidates_checkpoint')
                    if model_params is not None:
                        # if byoc_params["model_arch"].lower() in model["architecture"].lower():
                        if str(checkpoint['param_count']) + 'b' == model_params.get('model_parameter_count'):
                            if checkpoint['vocab_size'] == model_params.get('vocab_size'):
                                if checkpoint['seq_length'] == model_params.get('max_seq_length'):
                                    suitable_apps.add(model['app_id'])
            app_list = self.list_apps()
            named_suitable_apps = []
            assert app_list is not None
            for app in app_list:
                if app['id'] in suitable_apps:
                    named_suitable_apps.append(app)
            if verbose:
                logging.info(f'Checkpoint {checkpoint["model_name"]} suitable apps:' + '\n' + f'{named_suitable_apps}')
            checkpoints_suitable_apps.append(named_suitable_apps)
        return checkpoints_suitable_apps

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
            raise Exception(f'Error message: {model_info_response["message"]}')

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
            logging.error(f'Failed to list models. Details: {list_models_response["detail"]}')
            raise Exception(f'Error message: {list_models_response["detail"]}')

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

    """models - speculative decoding"""
    
    def create_spec_decoding_model(
        self,
        model_name: Optional[str] = None,
        target_model: Optional[str] = None,
        target_model_version: Optional[str] = None,
        draft_model: Optional[str] = None,
        draft_model_version: Optional[str] = None,
        rdu_arch: Optional[str] = None,
        job_type: Optional[str] = None,
    ) -> Optional[str]:
        """
        Creates a speculative decoding model pair.

        Args:
            - model_name (str): The name of the new model. (e.g. Meta-Llama-3.3-70B-Instruct-SD)
            - target_model (str): The name of the target model. (e.g. Meta-Llama-3.3-70B-Instruct)
            - target_model_version (str): The version of the target model. (e.g. 1)
            - draft_model (str): The name of the draft model. (e.g. Meta-Llama-3-8B-Instruct)
            - draft_model_version (str): The version of the draft model. (e.g. 1)
            - rdu_arch (str): The RDU architecture. (e.g. SN40L-8)
            - job_type (str): The type of job. (e.g. deploy)

        Returns:
            - The ID of the newly created model.

        Raises:
            - Exception: If the target or draft model does not exist, or if the target model does not support
            speculative decoding (e.g. models are not compatible for spec decoding).
            - Exception: If there is an error during the validation or creation process.
        """

        if model_name is None:
            self._raise_error_if_config_is_none()
            model_name = self.config['spec_decoding']['model_name']

        if target_model is None:
            self._raise_error_if_config_is_none()
            target_model = self.config['spec_decoding']['target_model']

        if target_model_version is None:
            self._raise_error_if_config_is_none()
            target_model_version = self.config['spec_decoding']['target_model_version']

        if draft_model is None:
            self._raise_error_if_config_is_none()
            draft_model = self.config['spec_decoding']['draft_model']

        if draft_model_version is None:
            self._raise_error_if_config_is_none()
            draft_model_version = self.config['spec_decoding']['draft_model_version']

        if rdu_arch is None:
            self._raise_error_if_config_is_none()
            rdu_arch = self.config['sambastudio']['rdu_arch']

        if job_type is None:
            self._raise_error_if_config_is_none()
            job_type = self.config['job']['job_type']

        # check if models exist
        target_model_query = self.search_model(target_model)
        if target_model_query is None:
            raise Exception(f"Model with name '{target_model}' does not exist.")
        draft_model_query = self.search_model(draft_model)
        if draft_model_query is None:
            raise Exception(f"Model with name '{draft_model}' does not exist.")

        target_model_info_response = self.model_info(target_model, job_type)
        if target_model_info_response['hyperparams'][job_type][rdu_arch]['supports_speculative_decoding'] == False:
            raise Exception(f"Model with name '{target_model}' does not support speculative decoding.")

        # check if models are compatible
        snapi_validate_spec_decoding_command = self._build_snapi_validate_spec_decoding(
            target_model, target_model_version, draft_model, draft_model_version, rdu_arch
        )

        echo_response = subprocess.run(['echo', 'yes'], capture_output=True, text=True)
        snapi_response = subprocess.run(
            snapi_validate_spec_decoding_command, input=echo_response.stdout, capture_output=True, text=True
        )

        error_message = 'Failed to validate speculative decoding. Details: '
        errors_response = self._handle_error_spec_decoding(snapi_response, error_message)

        if errors_response is None:
            logging.info(f"Speculative decoding validation: '{snapi_response.stdout}'")

        snapi_create_spec_decoding_command = self._build_snapi_create_spec_decoding_pair(
            model_name, target_model, target_model_version, draft_model, draft_model_version, rdu_arch
        )

        echo_response = subprocess.run(['echo', 'yes'], capture_output=True, text=True)
        snapi_response = subprocess.run(
            snapi_create_spec_decoding_command, input=echo_response.stdout, capture_output=True, text=True
        )

        error_message = 'Failed to create speculative decoding. Details: '
        errors_response = self._handle_error_spec_decoding(snapi_response, error_message)

        if errors_response is None:
            logging.info(f"Speculative decoding creation message: '{snapi_response.stdout}'")
            new_model_id = self.search_model(model_name)
            return new_model_id

    def _handle_error_spec_decoding(self, snapi_response: any, custom_error_message: str) -> None:
        """
        Handles error decoding for a given SNAPI response.

        Checks the response for internal server errors, validation failures, or non-empty stderr.
        If an error is detected, logs the error message and raises an exception.
        If all lines in the error message are warnings, logs a warning instead of raising an exception.

        Args:
            snapi_response (any): The response from the SNAPI.
            custom_error_message (str): A custom error message to include in the exception.

        Raises:
            - Exception: custom error message.

        """
        errors_response = (
            ('internal server error' in snapi_response.stdout.lower())
            and ('failed to validate' in snapi_response.stdout.lower())
        ) or (len(snapi_response.stderr) > 0)
        # if errors coming in response
        if errors_response:
            error_message = snapi_response.stdout
            # if all lines in stderr are warnings dont raise
            if all('warning' in line.lower() for line in error_message.splitlines()):
                logging.warning(f'message with warning: {error_message}')
            else:
                logging.error(error_message + f'{error_message}')
                raise Exception(f'Error message: {error_message}')
        # if there are no errors in reponse
        else:
            return None

    def _build_snapi_validate_spec_decoding(
        self,
        target_model: Optional[str] = None,
        target_model_version: Optional[int] = None,
        draft_model: Optional[str] = None,
        draft_model_version: Optional[int] = None,
        rdu_arch: Optional[str] = None,
    ) -> List[str]:
        """
        Builds the SNAPI command to validate speculative decoding for a pair of models.

        Args:
            - target_model (str): The name of the target model.
            - target_model_version (int): The version of the target model.
            - draft_model (str): The name of the draft model.
            - draft_model_version (int): The version of the draft model.
            - rdu_arch (str): The RDU architecture.

        Returns:
            - The SNAPI command to validate speculative decoding.
        """
        command = [
            'snapi',
            'model',
            'validate-spec-decoding',
            '--target',
            target_model,
            '--target-version',
            target_model_version,
            '--draft',
            draft_model,
            '--draft-version',
            draft_model_version,
            '--rdu-arch',
            rdu_arch,
        ]

        return command

    def _build_snapi_create_spec_decoding_pair(
        self,
        model_name: Optional[str] = None,
        target_model: Optional[str] = None,
        target_model_version: Optional[int] = None,
        draft_model: Optional[str] = None,
        draft_model_version: Optional[int] = None,
        rdu_arch: Optional[str] = None,
    ) -> List[str]:
        """
        Builds the SNAPI command to create a speculative decoding pair.

        Args:
            - model_name (str): The name of the new model.
            - target_model (str): The name of the target model.
            - target_model_version (int): The version of the target model.
            - draft_model (str): The name of the draft model.
            - draft_model_version (int): The version of the draft model.
            - rdu_arch (str): The RDU architecture.

        Returns:
            - The SNAPI command to create a speculative decoding pair.
        """
        command = [
            'snapi',
            'model',
            'create-sd-pair',
            '--name',
            model_name,
            '--target',
            target_model,
            '--target-version',
            target_model_version,
            '--draft',
            draft_model,
            '--draft-version',
            draft_model_version,
            '--rdu-arch',
            rdu_arch,
        ]

        return command
    
    """models - bundles"""
    
    def create_composite_model(
        self,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        model_list: Optional[List[str]] = None,
        rdu_required: Optional[int] = None,
        verbose: Optional[bool] = True,
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
        - verbose (bool): show informational logs

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
            if verbose:
                logging.info(f"Models to include in composite found with ids '{list(zip(model_list, model_ids))}")

            # create composite model
            dependencies = [{'name': model} for model in model_list]
            create_composite_model_response = self.snsdk_client.add_composite_model(
                name=model_name,
                description=description,
                dependencies=dependencies,
                rdu_required=rdu_required,
                config_params={},
                app='',
            )

            if create_composite_model_response['status_code'] == 200:
                model_id = create_composite_model_response['model_id']
                if verbose:
                    logging.info(f'Composite model with name {model_name} created with id {model_id}')
            else:
                logging.error(
                    f'Failed to create composite model with name "{model_name}".'
                    f'Message: {create_composite_model_response["message"]}.'
                    f'Details: {create_composite_model_response["details"]}'
                )
                raise Exception(f'Error message: {create_composite_model_response["details"]}')

        # if selected composite model already exists
        else:
            if verbose:
                logging.info(f"Model with name '{model_name}' not created it already exist with id {model_id}")

        return model_id
    
    """models - BYOC utils"""
    
    def find_config_params(
        self,
        checkpoint_paths: Optional[Union[List[str], str]] = None,
        update_config_file: bool = False,
        verbose: Optional[bool] = True,
    ) -> List[Dict[str, Any]]:
        """
        Finds and returns the model architecture, sequence length, and vocabulary size for config.json files
        in given checkpoint paths.

        Parameters:
            - checkpoint_paths (list of str or str, optional): checkpoint paths.
                if not set config paths in config,yaml file will be used
            - update_config_file (bool, optional): Whether to update the config file
                with the found parameters. Defaults to False.
            - verbose (bool): show informational logs

        Returns:
            checkpoint_parameters (list): list of dicts with the model_arch,
                seq_length and vocab size found for each model
        """
        if isinstance(checkpoint_paths, str):
            checkpoint_paths = [checkpoint_paths]

        if checkpoint_paths is None:
            self._raise_error_if_config_is_none()
            checkpoint_paths = [checkpoint['checkpoint_path'] for checkpoint in self.config['checkpoints']]

        checkpoint_params = []
        for checkpoint_path in checkpoint_paths:
            with open(os.path.join(checkpoint_path, 'config.json')) as file:
                checkpoint_config = json.load(file)
                checkpoint_params.append(
                    {
                        'model_arch': checkpoint_config['model_type'],
                        'seq_length': checkpoint_config['max_position_embeddings'],
                        'vocab_size': checkpoint_config['vocab_size'],
                    }
                )
                if verbose:
                    logging.info(f'Params for checkpoint in {checkpoint_path}:\n{checkpoint_params}')

        if self.config is not None:
            checkpoints = []
            for checkpoint, params in zip(self.config['checkpoints'], checkpoint_params):
                checkpoint['model_arch'] = params['model_arch']
                checkpoint['seq_length'] = params['seq_length']
                checkpoint['vocab_size'] = params['vocab_size']
                checkpoints.append(checkpoint)
            self.config['checkpoints'] = checkpoints
            if verbose:
                logging.info(f'config updated with checkpoints parameters')

            if update_config_file:
                self._raise_error_if_config_is_none()
                assert isinstance(self.config_path, str)
                with open(self.config_path, 'w') as outfile:
                    yaml.dump(self.config, outfile)
                if verbose:
                    logging.info(f'config file updated with checkpoints parameters')

        return checkpoint_params

    def check_chat_templates(
        self,
        test_messages: List[str],
        checkpoint_paths: Optional[Union[List[str], str]] = None,
        verbose: Optional[bool] = True,
    ) -> None:
        """
        Checks the chat templates for the given checkpoint paths.

        Reads the tokenizer config file for each checkpoint path, extracts the chat template,
        and checks if it can be rendered with the provided test messages.

        Parameters:
            - test_messages (List[str]): A list of test messages to use for rendering the chat template.
            - checkpoint_paths (list of str or str, optional): checkpoint paths.
                if not set config paths in config.yaml file will be used
            - verbose (bool): show informational logs

        Returns:
            None
        """
        if isinstance(checkpoint_paths, str):
            checkpoint_paths = [checkpoint_paths]

        if checkpoint_paths is None:
            self._raise_error_if_config_is_none()
            checkpoint_paths = [checkpoint['checkpoint_path'] for checkpoint in self.config['checkpoints']]

        for checkpoint_path in checkpoint_paths:
            with open(os.path.join(checkpoint_path, 'tokenizer_config.json')) as file:
                tokenizer_config = json.load(file)
                chat_template = tokenizer_config.get('chat_template')
            if isinstance(chat_template, str):
                if verbose:
                    logging.info(f'Raw chat template for checkpoint in {checkpoint_path}:\n{chat_template}\n')
                if version.parse(jinja2.__version__) <= version.parse('3.0.0'):
                    raise ImportError(
                        'apply_chat_template requires jinja2>=3.0.0 to be installed. Your version is '
                        f'{jinja2.__version__}.'
                    )
                try:
                    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
                    undeclared_variables = meta.find_undeclared_variables(jinja_env.parse(chat_template))
                    special_tokens_map = {'messages': test_messages, **tokenizer_config}
                    missing_variables = set(undeclared_variables) - set(special_tokens_map.keys())
                    if len(missing_variables) > 0:
                        logging.error(f'Missing variables to render template: {missing_variables}')
                    else:
                        compiled_template = jinja_env.from_string(chat_template)
                        rendered = compiled_template.render(**special_tokens_map)
                        logging.info(f'Rendered template with input test messages:\n\n{rendered}')
                except Exception as e:
                    logging.error(f'Failed to render template: {str(e)}')
            else:
                logging.error(f'Raw chat template for checkpoint in {checkpoint_path}: is not a string')

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
            model_name = self.config['job']['model']
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
                    f"""
                    {
                        [
                            param['field_name'] + ':`' + param['settings']['DEFAULT'] + '`'
                            for param in user_hyperparams_list
                        ]
                    }\n
                    """
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
            logging.error(f'Failed to list jobs. Details: {list_jobs_response["detail"]}')
            raise Exception(f'Error message: {list_jobs_response["detail"]}')

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
            for checkpoint in list_checkpoints_response['checkpoints']:
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

    """checkpoints - BYOC"""
    
    def _build_snapi_import_model_create_command(
        self,
        model_name: str,
        model_arch: str,
        param_count: int,
        seq_length: int,
        vocab_size: int,
        checkpoint_path: str,
        app_id: str,
        publisher: Optional[str] = None,
        description: Optional[str] = None,
        ignore_transformers_version: Optional[bool] = True,
    ) -> List[str]:
        """Build the command to import a model into Snapi

        Parameters:
        - model_name (str): name of the model
        - model_arch (str): architecture of the model
        - param_count (int): number of parameters in the model in billions of parameters
        - seq_length (int): sequence length of the model
        - vocab_size (int): vocabulary size of the model
        - checkpoint_path (str): path to the checkpoint folder
        - app_id (str): id of the application
        - publisher (str, optional): publisher of the model. Defaults to None.
        - description (str, optional): description of the model. Defaults to None.

        Returns:
        - str: command to byoc with Snapi
        """

        command = [
            'snapi',
            'import',
            'model',
            'create',
            '--model-name',
            model_name,
            '--app',
            app_id,
            '--source-type',
            'LOCAL',
            '--source-path',
            checkpoint_path,
            '--model-arch',
            model_arch,
            '--parameter-count',
            str(param_count) + 'b',
            '--sequence-length',
            str(seq_length),
            '--vocab-size',
            str(vocab_size),
            '-ni',  # -ni flag is used to  use non interactive mode
        ]

        # --publisher and --description flags and values are optional and only sent
        #   if provided in the config file or arguments
        if publisher is not None:
            if len(publisher) > 0:
                command.extend(['--publisher', publisher])
        if description is not None:
            if len(description) > 0:
                command.extend(['--description', description])
        if ignore_transformers_version:
            command.extend(['--ignore-transformers-version'])

        return command

    def upload_checkpoint(
        self,
        model_name: str,
        model_arch: str,
        param_count: int,
        seq_length: int,
        vocab_size: int,
        checkpoint_path: str,
        app_id: str,
        publisher: str = '',
        description: str = '',
        retries: int = 3,
        ignore_transformers_version: bool = True,
        verbose: Optional[bool] = True,
    ) -> Optional[str]:
        """Upload the checkpoint to Snapi

        Parameters:
        - model_name (str): name of the model.
        - model_arch (str): architecture of the model.
        - param_count (int): number of parameters in the model in billions of parameters.
        - seq_length (int): sequence length of the model.
        - vocab_size (int): vocabulary size of the model.
        - checkpoint_path (str): path to the checkpoint folder.
        - app_id (str): id of the application.
        - publisher (str, optional): publisher of the model. Defaults to "".
        - description (str, optional): description of the model. Defaults to "".
        - retries (int): max number of retries to upload a checkpoint when
            upload process fails. Defaults to 3.
        - verbose (bool): show informational logs

        Raises:
            Exception: If checkpoint upload fails

        Returns:
        - str: model id if model uploaded successfully. None otherwise.
        """

        # validate if model already exist
        model_id = self.search_model(model_name=model_name)
        if model_id is None:
            # Validate if path exist
            if not os.path.isdir(checkpoint_path):
                logging.error(f'path: {checkpoint_path} for checkpoint with name: {model_name} does not exist')
                raise ValueError(f'{checkpoint_path} does not exist')

            # Validate if app exist
            found_app_id = self.search_app(app_id)
            if found_app_id is None:
                logging.error(f'app: {app_id} for checkpoint with name: {model_name} does not exist')
                raise ValueError(f'app: {app_id} does not exist')

            # upload the checkpoint
            command = self._build_snapi_import_model_create_command(
                model_name,
                model_arch,
                param_count,
                seq_length,
                vocab_size,
                checkpoint_path,
                app_id,
                publisher,
                description,
                ignore_transformers_version,
            )
            # attempt to upload retries times
            for i in range(retries + 1):
                # execute snapi import model create command
                if verbose:
                    logging.info(f'running snapi upload command:\n {" ".join(command)}\nThis could take a while')
                snapi_response = subprocess.run(command, capture_output=True, text=True)

                # check if errors in execution
                errors_response = (
                    ('aborted' in snapi_response.stdout.lower()) and ('error occurred' in snapi_response.stdout.lower())
                ) or (len(snapi_response.stderr) > 0)

                # capture errors coming in response
                if errors_response:
                    if len(snapi_response.stderr) > 0:
                        error_message = snapi_response.stderr
                        # if all lines in stderr are warnings set errors_response as false
                        if all('warning' in line.lower() for line in error_message.splitlines()):
                            logging.warning(f'Warnings when uploading checkpoint : {error_message}')
                            errors_response = False
                        else:
                            logging.error(snapi_response.stdout)
                            logging.error('Error uploading model checkpoint Process returned a non-zero exit code.')
                            logging.error(error_message)
                    else:
                        error_search = re.search(r'Upload Error\s*(. *)', snapi_response.stdout)
                        if error_search:
                            error_message = error_search[0]
                        logging.error(snapi_response.stdout)
                        logging.error('Error uploading model checkpoint Process returned a non-zero exit code.')
                        logging.error(error_message)

                if errors_response:
                    if i < retries:
                        if verbose:
                            logging.info(f'Retrying upload for {i + 1} time...')
                        continue
                    else:
                        raise Exception(
                            f'Error uploading model checkpoint after {retries} attempts\nmax retries exceed'
                        )
                # if there are no errors in response
                else:
                    model_id = self.search_model(model_name=model_name)
                    if verbose:
                        logging.info(f"Model checkpoint with name '{model_name}' created it with id {model_id}")
                    break

        # if checkpoint already exists
        else:
            logging.info(f"Model checkpoint with name '{model_name}' not created it already exist with id {model_id}")

        return model_id

    def upload_checkpoints(
        self,
        checkpoints: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        max_parallel_jobs: int = 4,
        retries: int = 3,
        verbose: Optional[bool] = True,
    ) -> str:
        """
        Upload checkpoints to sambastudio

        Parameters:
        - checkpoints (Optional[List[Dict[str, Any]]], optional): list of checkpoints.
            If not provided, all checkpoints from config will be uploaded
        - max_parallel_jobs (int): maximum number of upload parallel jobs. Defaults to 4.
        - retries (int): max number of retries to upload a checkpoint when
            upload process fails. Defaults to 3.
        - verbose (bool): show informational logs
        """

        if checkpoints is None:
            self._raise_error_if_config_is_none()
            checkpoints = self.config['checkpoints']

        if isinstance(checkpoints, Dict):
            checkpoints = [checkpoints]

        futures = {}
        with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
            # Submit all the upload tasks to the executor
            for checkpoint in checkpoints:
                # Create a future for the upload_checkpoint function
                future = executor.submit(
                    self.upload_checkpoint,
                    checkpoint['model_name'],
                    checkpoint['model_arch'],
                    checkpoint['param_count'],
                    checkpoint['seq_length'],
                    checkpoint['vocab_size'],
                    checkpoint['checkpoint_path'],
                    checkpoint['app_id'],
                    checkpoint.get('publisher', ''),
                    checkpoint.get('description', ''),
                    retries,
                )
                futures[checkpoint['model_name']] = future  # Add to the futures dictionary

            # Wait for all tasks to complete and handle exceptions
            models = []
            for model_name, future in futures.items():
                try:
                    result = future.result()  # This will raise the exception if the thread raised one
                    models.append({'name': model_name, 'id': result})
                    if verbose:
                        logging.info(f'Checkpoint for model {model_name} finished successfully with result {result} ')
                except Exception as e:
                    logging.error(f'Error uploading checkpoint for model {model_name}: {e}', exc_info=True)
        return models

    def get_checkpoints_status(
        self,
        model_names: Optional[Union[List[str], str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get status of uploaded checkpoints

        Parameters:
        - model_names (Optional[List[str]], optional): list of model names.

        Return
        - model_statuses: list of model checkpoints status
        """
        if model_names is None:
            self._raise_error_if_config_is_none()
            model_names = [checkpoint['model_name'] for checkpoint in self.config['checkpoints']]
        if isinstance(model_names, str):
            model_names = [model_names]

        model_statuses = []
        for model in model_names:
            model_status = self.snsdk_client.import_status(model_id=model)
            model_statuses.append(model_status)
        return model_statuses    

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
            logging.error(f'Failed to list endpoints. Details: {list_endpoints_response["detail"]}')
            raise Exception(f'Error message: {list_endpoints_response["detail"]}')

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
        # TODO: enable inference_api_openai_compatible when available
        create_endpoint_response = self.snsdk_client.create_endpoint(
            project=project_id,
            endpoint_name=endpoint_name,
            description=endpoint_description,
            model_checkpoint=model_id,
            model_version=model_version,
            instances=instances,
            rdu_arch=rdu_arch,
            hyperparams=json.dumps(hyperparams),
            inference_api_openai_compatible=False,
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
