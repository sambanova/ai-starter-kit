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
    ) -> List[Dict[str, Any]]:
        """
        Finds and returns the model architecture, sequence length, and vocabulary size for config.json files
        in given checkpoint paths.

        Parameters:
            checkpoint_paths (list of str or str, optional): checkpoint paths.
                if not set config paths in config,yaml file will be used
            update_config_file (bool, optional): Whether to update the config file
                with the found parameters. Defaults to False.

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

        return checkpoint_params

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

    def get_suitable_apps(
        self, checkpoints: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> List[List[Dict[str, Any]]]:
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
            logging.info(f'Checkpoint {checkpoint["model_name"]} suitable apps:' + '\n' + f'{named_suitable_apps}')
            checkpoints_suitable_apps.append(named_suitable_apps)
        return checkpoints_suitable_apps

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
                        logging.info(f'Retrying upload for {i + 1} time...')
                        continue
                    else:
                        raise Exception(
                            f'Error uploading model checkpoint after {retries} attempts\nmax retries exceed'
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
        self,
        checkpoints: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        max_parallel_jobs: int = 4,
        retries: int = 3,
    ) -> str:
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
                    logging.info(f'Checkpoint for model {model_name} finished successfully with result {result} ')
                except Exception as e:
                    logging.error(f'Error uploading checkpoint for model {model_name}: {e}', exc_info=True)
        return models

    def get_checkpoints_status(self, model_names: Optional[Union[List[str], str]] = None) -> List[Dict[str, Any]]:
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
            logging.info(f'model {model} status: \n {model_status}')
        return model_statuses
