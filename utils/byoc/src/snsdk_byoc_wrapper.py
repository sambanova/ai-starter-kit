import json
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Union

import jinja2
import yaml
from dotenv import load_dotenv
from jinja2 import meta
from jinja2.sandbox import ImmutableSandboxedEnvironment
from packaging import version

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)

from utils.fine_tuning.src.snsdk_wrapper import SnsdkWrapper

load_dotenv(os.path.join(repo_dir, '.env'), override=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)


class BYOC(SnsdkWrapper):
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Wrapper around the SnSdk and SNAPI for BYOC in SambaStudio

        Parameters:
            config_path (str , optional): path to config path. Defaults to None.
            see a config file example in ./config.yaml
        """

        # Initialize SnsdkWrapper with sambastudio config (initialize snsdk client and snapi path)
        super().__init__(config_path=config_path)

    def find_config_params(
        self, checkpoint_paths: Optional[Union[List[str], str]] = None, update_config_file: bool = False
    ) -> None:
        """
        Finds and returns the model architecture, sequence length, and vocabulary size for config.json files
        in given checkpoint paths.

        Parameters:
            checkpoint_paths (list of str or str, optional): checkpoint paths.
                if not set config paths in config,yaml file will be used
            update_config_file (bool, optional): Whether to update the config file
                with the found parameters. Defaults to False.
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
                logging.info(f'Params for checkpoint in {checkpoint_path}:\n{checkpoint_params}')

        if self.config is not None:
            checkpoints = []
            for checkpoint, params in zip(self.config['checkpoints'], checkpoint_params):
                checkpoint['model_arch'] = params['model_arch']
                checkpoint['seq_length'] = params['seq_length']
                checkpoint['vocab_size'] = params['vocab_size']
                checkpoints.append(checkpoint)
            self.config['checkpoints'] = checkpoints
            logging.info(f'config updated with checkpoints parameters')

            if update_config_file:
                self._raise_error_if_config_is_none()
                assert isinstance(self.config_path, str)
                with open(self.config_path, 'w') as outfile:
                    yaml.dump(self.config, outfile)
                logging.info(f'config file updated with checkpoints parameters')

    def check_chat_templates(
        self, test_messages: List[str], checkpoint_paths: Optional[Union[List[str], str]] = None
    ) -> None:
        """
        Checks the chat templates for the given checkpoint paths.

        Reads the tokenizer config file for each checkpoint path, extracts the chat template,
        and checks if it can be rendered with the provided test messages.

        Parameters:
            test_messages (List[str]): A list of test messages to use for rendering the chat template.
            checkpoint_paths (list of str or str, optional): checkpoint paths.
                if not set config paths in config.yaml file will be used

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

    def get_suitable_apps(self, checkpoints: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None) -> None:
        """
        find suitable sambastudio apps for the given checkpoints

        Parameters:
            checkpoints (list of dict or dict, optional): checkpoints.
                if not set checkpoints in config.yaml file will be used
        """
        if isinstance(checkpoints, dict):
            checkpoints = [checkpoints]
        if checkpoints is None:
            self._raise_error_if_config_is_none()
            checkpoints = self.config['checkpoints']

        sstudio_models = self.list_models(verbose=True)

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
                    named_suitable_apps.append(str(app))

            formatted_suitable_apps = '\n'.join(named_suitable_apps)
            logging.info(f'Checkpoint {checkpoint["model_name"]} suitable apps:' + '\n' + f'{formatted_suitable_apps}')

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
            )
            # attempt to upload retries times
            for i in range(retries + 1):
                # execute snapi import model create command
                snapi_response = subprocess.run(command, capture_output=True, text=True)

                # check if errors in execution
                errors_response = (
                    ('Aborted' in snapi_response.stdout.lower()) and ('error occured' in snapi_response.stdout.lower())
                ) or (len(snapi_response.stderr) > 0)

                # capture errors coming in response
                if errors_response:
                    if len(snapi_response.stderr) > 0:
                        error_message = snapi_response.stderr
                    else:
                        error_search = re.search(r'Upload Error\s*(.*)', snapi_response.stdout)
                        if error_search:
                            error_message = error_search[0]
                    logging.error('Error uploading model checkpoint Process returned a non-zero exit code.')
                    logging.error(error_message)
                    if i < retries:
                        logging.info(f'Retrying upload for {i+1} time...')
                        continue
                    else:
                        raise Exception(
                            f'Error uploading model checkpoint after {retries} attempts\n' 'max retries exceed'
                        )

                # if there are no errors in response
                else:
                    model_id = self.search_model(model_name=model_name)
                    logging.info(f"Model checkpoint with name '{model_name}' created it with id {model_id}")
                    break

        # if checkpoint already exists
        else:
            logging.info(f"Model checkpoint with name '{model_name}' not created it already exist with id {model_id}")

        return model_id

    def upload_checkpoints(
        self, checkpoints: Optional[List[Dict[str, Any]]] = None, max_parallel_jobs: int = 4, retries: int = 3
    ) -> None:
        """
        Upload checkpoints to sambastudio

        Parameters:
        - checkpoints (Optional[List[Dict[str, Any]]], optional): list of checkpoints.
            If not provided, all checkpoints from config will be uploaded
        - max_parallel_jobs (int): maximum number of upload parallel jobs. Defaults to 4.
        - retries (int): max number of retries to upload a checkpoint when
            upload process fails. Defaults to 3.
        """

        if checkpoints is None:
            self._raise_error_if_config_is_none()
            checkpoints = self.config['checkpoints']

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
            for model_name, future in futures.items():
                try:
                    result = future.result()  # This will raise the exception if the thread raised one
                    logging.info(f'Checkpoint for model {model_name} finished successfully with result {result} ')
                except Exception as e:
                    logging.error(f'Error uploading checkpoint for model {model_name}: {e}', exc_info=True)

    def get_checkpoints_status(self, model_names: Optional[List[str]] = None) -> None:
        """
        Get status of uploaded checkpoints

        Parameters:
        - model_names (Optional[List[str]], optional): list of model names.
        """
        if model_names is None:
            self._raise_error_if_config_is_none()
            model_names = [checkpoint['model_name'] for checkpoint in self.config['checkpoints']]
        for model in model_names:
            model_status = self.snsdk_client.import_status(model_id=model)
            logging.info(f'model {model} status: \n {model_status}')

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

        if model_name is None:
            self._raise_error_if_config_is_none()
            model_name = self.config['composite_model']['model_name']

        if model_version is None:
            model_version = '1'

        return super().create_endpoint(
            project_name,
            endpoint_name,
            endpoint_description,
            model_name,
            model_version,
            instances,
            rdu_arch,
            hyperparams,
        )
