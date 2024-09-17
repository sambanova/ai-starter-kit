import subprocess
import itertools
import yaml
import random
import time
import os
import logging

class GeneralModelTestObject:
    """
    A class for managing and testing AI models using SNAPI.
    """

    def __init__(self, project, chiparch):
        """
        Initialize the GeneralModelTestObject.

        Args:
            project (str): The project name.
            chiparch (str): The chip architecture.
        """
        self.project = project 
        self.arch = chiparch
        self.logger = logging.getLogger(__name__)
        self.forbidden_sweep = {
            'do_eval', 'evaluation_strategy', 'lr_schedule', 
            'save_optimizer_state', 'model_arch_type',
            'use_token_type_ids', 'truncate_pattern',
            'scheduler_type', 'normalize', 'use_lm_decoding',
            'use_number_transcriber', 'prediction_handler',
            'max_seq_length'
        }

    def load_config(self, config_file):
        """
        Load configuration from a YAML file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            dict: Loaded configuration.
        """
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_file}")
            return {}
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            return {}

    def run_command(self, command, timeout=300):
        """
        Run a shell command with timeout and error handling.

        Args:
            command (str): The command to run.
            timeout (int): Timeout in seconds.

        Returns:
            tuple: (success, output, error)
        """
        try:
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after {timeout} seconds: {command}")
            return False, "", "Timeout"
        except Exception as e:
            self.logger.error(f"Error running command: {command}\nError: {str(e)}")
            return False, "", str(e)

    def model_endpoint(self, modelName):
        """
        Create and check the status of a model endpoint.

        Args:
            modelName (str): Name of the model.

        Returns:
            tuple: (success, api_key, url)
        """
        ins = 1
        jobname = f"{modelName.replace(' ','-').replace('_','-')}-{ins}ins-endpoint-{self.arch.lower()}".lower()

        create_command = f"snapi endpoint create -p {self.project} -n {jobname} -m '{modelName}' -a {self.arch} -i {ins}"
        success, output, error = self.run_command(create_command)

        if not success:
            self.logger.error(f"Failed to create endpoint for {modelName}: {error}")
            return False, None, None

        self.logger.info(f"Endpoint creation initiated for {modelName}")
        time.sleep(5)  # Wait for 5 seconds before checking status

        info_command = f"snapi endpoint info -p {self.project} -e {jobname}"
        success, output, error = self.run_command(info_command)

        if not success:
            self.logger.error(f"Failed to get endpoint info for {modelName}: {error}")
            return False, None, None

        api_key = None
        url = None
        status = None

        for line in output.splitlines():
            if 'API Key' in line and 'Keys' not in line:
                api_key = line.split(': ')[-1]
            elif 'Status' in line:
                status = line.split(': ')[-1]
            elif 'URL' in line:
                url = line.split(': ')[-1]

        if status == 'Live':
            return True, api_key, url
        elif status == 'Failed':
            self.logger.error(f"Endpoint creation failed for {modelName}")
            return False, api_key, url
        else:
            self.logger.info(f"Endpoint status for {modelName}: {status}. Waiting for it to go live...")
            return self._wait_for_endpoint(jobname, api_key, url)

    def _wait_for_endpoint(self, jobname, api_key, url, max_attempts=18, wait_time=10):
        """
        Wait for an endpoint to become live.

        Args:
            jobname (str): Name of the job.
            api_key (str): API key.
            url (str): Endpoint URL.
            max_attempts (int): Maximum number of status check attempts.
            wait_time (int): Time to wait between attempts in seconds.

        Returns:
            tuple: (success, api_key, url)
        """
        command = f"snapi endpoint info -p {self.project} -e {jobname}"

        for _ in range(max_attempts):
            success, output, _ = self.run_command(command)
            if success:
                for line in output.splitlines():
                    if 'Status' in line:
                        status = line.split(': ')[-1]
                        if status == 'Live':
                            self.logger.info(f"Endpoint {jobname} is now live")
                            return True, api_key, url
                        elif status == 'Failed':
                            self.logger.error(f"Endpoint {jobname} failed to go live")
                            return False, api_key, url
            
            time.sleep(wait_time)

        self.logger.error(f"Endpoint {jobname} did not go live within the expected time")
        return False, api_key, url

    def model_config_generation_prepare(self, config_file, sample_num=1):
        """
        Prepare model configuration based on a config file.

        Args:
            config_file (str): Path to the configuration file.
            sample_num (int): Number of samples to generate for each parameter.

        Returns:
            dict: Prepared model configuration.
        """
        config_dict = self.load_config(config_file)
        if not config_dict:
            return {}

        res_config_dict = {}

        for k, v in config_dict.items():
            if not isinstance(v, dict):
                continue

            if 'values' in v:
                res_config_dict[k] = v['values'] if sample_num > 1 else [v['values'][0]]
                if k == 'evaluation_strategy':
                    res_config_dict[k] = ['steps']
            elif len(v) == 1:
                key, value = next(iter(v.items()))
                if key in ['ge', 'gt', 'le', 'lt']:
                    res_config_dict[k] = self._generate_random_values(key, value, sample_num, k)
            elif len(v) == 2:
                keys = list(v.keys())
                lower_bound = float(v[keys[0]]) + 1e-9
                upper_bound = float(v[keys[1]])
                res_config_dict[k] = [
                    int(random.uniform(lower_bound, upper_bound)) if upper_bound != 1 
                    else random.uniform(lower_bound, upper_bound) 
                    for _ in range(sample_num)
                ]

        return res_config_dict

    def _generate_random_values(self, key, value, sample_num, param_name):
        """
        Generate random values based on the constraint key and value.

        Args:
            key (str): Constraint key ('ge', 'gt', 'le', 'lt').
            value (str): Constraint value.
            sample_num (int): Number of samples to generate.
            param_name (str): Name of the parameter.

        Returns:
            list: Generated random values.
        """
        try:
            value = float(value)
            if key in ['ge', 'gt']:
                upper = value + 100 if 'step' not in param_name else value + 2
                return [random.uniform(value, upper) for _ in range(sample_num)]
            elif key in ['le', 'lt']:
                return [random.uniform(0, value) for _ in range(sample_num)]
        except ValueError:
            self.logger.warning(f"Invalid value for parameter {param_name}: {value}")
            return [0] * sample_num

    def model_config_generation(self, modelName, configFileList, sampleNum=1, artifacts_path='../../artifacts/hf/'):
        """
        Generate model configurations based on a list of config files.

        Args:
            modelName (str): Name of the model.
            configFileList (list): List of configuration file paths.
            sampleNum (int): Number of samples to generate for each parameter.

        Returns:
            tuple: (training_hf_l, inference_hf_l)
        """
        training_hf_l = []
        inference_hf_l = []

        if not configFileList:
            return training_hf_l, inference_hf_l

        for configFile in configFileList:
            resConfigDict = self.model_config_generation_prepare(configFile, sampleNum)

            key_l = list(resConfigDict.keys())
            hf_l = [v for k, v in resConfigDict.items()]
            hfcombo_l = list(itertools.product(*hf_l))

            testType = 'training' if 'train' in configFile else 'inference'

            for i, hfitem in enumerate(hfcombo_l):
                modelName = modelName.replace(' ', '_')
                filename = f"{artifacts_path}/{modelName}_{testType}_testcase{i+1}.yaml"

                if testType == 'training':
                    training_hf_l.append(filename)
                elif testType == 'inference':
                    inference_hf_l.append(filename)

                with open(filename, "w") as f:
                    for i, v in enumerate(key_l):
                        f.write(f"{v}: {hfitem[i]}\n")

        return training_hf_l, inference_hf_l

    def model_config_generation_new(self, modelName, configFileList, MODE='NORMAL', artifacts_path='../../artifacts/hf/'):
        """
        Generate new model configurations based on a list of config files.

        Args:
            modelName (str): Name of the model.
            configFileList (list): List of configuration file paths.
            MODE (str): Mode of operation ('NORMAL', 'ND', 'CKPT').

        Returns:
            tuple: (training_hf_l, inference_hf_l)
        """
        training_hf_l = []
        inference_hf_l = []

        modelName = modelName.replace(' ', '_')

        if not configFileList:
            return training_hf_l, inference_hf_l

        for configFile in configFileList:
            resConfigDict = {}

            config_dict = self.load_config(configFile)
            if not config_dict:
                continue

            for k, v in config_dict.items():
                if not isinstance(v, dict):
                    continue

                if 'values' in v and k not in self.forbidden_sweep:
                    if k == 'vocab_size' and '300k' not in modelName and 'GPT' in modelName:
                        resConfigDict[k] = ['50260']
                    elif k == 'max_seq_length' and '8k' in modelName:
                        resConfigDict[k] = ['8192']
                    elif k == 'max_seq_length' and '2k' in modelName:
                        resConfigDict[k] = ['2048']
                    elif k == 'max_seq_length' and ('2k' not in modelName and '8k' not in modelName and 'Llama' not in modelName and 'Hubert' not in modelName and '1.5B' not in modelName and 'GPT_13B_Base_Model' not in modelName):
                        resConfigDict[k] = ['2048']
                    elif k == 'skip_checkpoint':
                        resConfigDict[k] = ['true']
                    else:
                        if v['values']:
                            resConfigDict[k] = v['values']

            key_l = list(resConfigDict.keys())
            hf_l = [v for k, v in resConfigDict.items()]
            hfcombo_l = list(itertools.product(*hf_l))

            testType = 'training' if 'train' in configFile else 'inference'

            for i, hfitem in enumerate(hfcombo_l):
                filename = f"{artifacts_path}/{modelName}_{testType}"

                if not key_l:
                    filename += ".yaml"
                else:
                    for i, v in enumerate(key_l):
                        if i < len(key_l) - 1:
                            filename += f"_{v.replace('_','')}{hfitem[i]}"
                        else:
                            filename += f"_{v.replace('_','')}{hfitem[i]}.yaml"

                if testType == 'training':
                    training_hf_l.append(filename)
                elif testType == 'inference':
                    inference_hf_l.append(filename)

                with open(filename, "w") as f:
                    if not key_l:
                        if 'train' in filename:
                            if MODE == 'ND':
                                f.write(f"num_iterations: 10\nlogging_steps: 1\ndo_eval: false\nskip_checkpoint: true\n")
                            elif 'CKPT' in MODE:
                                f.write(f"num_iterations: 10\nlogging_steps: 1\ndo_eval: false\n")
                                if '1.5b' not in modelName.lower():
                                    f.write(f"skip_checkpoint: false\n")
                                f.write(f"save_optimizer_state: true\nsave_steps: 10\nevaluation_strategy: no\n")
                            else:
                                if any(model in modelName.lower() for model in ['llama-2-13b', 'llama-2-7b-16k', 'llama-2-7b-chat-16k']):
                                    f.write(f"num_iterations: 10\n")
                                else:
                                    f.write(f"num_iterations: 10\n")
                    else:
                        for i, v in enumerate(key_l):
                            if 'CKPT' not in MODE:
                                f.write(f"{v}: {hfitem[i]}\n")
                            if 'train' in filename and ('gpt' in filename.lower() or 'llama' in filename.lower()):
                                if MODE == 'ND':
                                    f.write(f"num_iterations: 10\nlogging_steps: 1\ndo_eval: false\nskip_checkpoint: true\n")
                                elif 'CKPT' in MODE:
                                    f.write(f"num_iterations: 10\nlogging_steps: 1\ndo_eval: false\n")
                                    if '1.5b' not in modelName.lower():
                                        f.write(f"skip_checkpoint: false\n")
                                    f.write(f"save_optimizer_state: true\nsave_steps: 10\nevaluation_strategy: no\n")
                                else:
                                    if any(model in modelName.lower() for model in ['llama-2-13b', 'llama-2-7b-16k', 'llama-2-7b-chat-16k']):
                                        f.write(f"num_iterations: 10\n")
                                    else:
                                        f.write(f"num_iterations: 10\n")
                                    f.write(f"skip_checkpoint: true\n")

        return training_hf_l, inference_hf_l

    def model_training(self, fileHandler, training_hf_l, modelName, dataName, RDU=1, SWEEP='N'):
        """
        Run model training jobs.

        Args:
            fileHandler: File handler for logging.
            training_hf_l (list): List of training hyperparameter files.
            modelName (str): Name of the model.
            dataName (str): Name of the dataset.
            RDU (int): Number of RDUs to use.
            SWEEP (str): Sweep mode.

        Returns:
            fileHandler: Updated file handler.
        """
        if len(training_hf_l) == 0 or SWEEP == 'N':
            if self._is_data_mismatch(modelName, dataName):
                self._log_data_mismatch(fileHandler, modelName, dataName, SWEEP)
                return fileHandler

            command = self._build_training_command(modelName, dataName, RDU, SWEEP)
            res = self._run_snapi_command(command)
            self._log_training_result(fileHandler, modelName, res, dataName, SWEEP, RDU, command)
            return fileHandler

        for hf in training_hf_l:
            if self._is_data_mismatch(modelName, dataName, hf):
                continue

            jobname = self._generate_job_name(hf, dataName, RDU, SWEEP)
            command = self._build_training_command(modelName, dataName, RDU, SWEEP, hf, jobname)
            res = self._run_snapi_command(command)
            self._log_training_result(fileHandler, modelName, res, dataName, SWEEP, RDU, command, hf, jobname)

        return fileHandler

    def _is_data_mismatch(self, modelName, dataName, hf=None):
        """Check if there's a mismatch between model and data."""
        if 'GPT' in (hf or modelName):
            if ('8k' in (hf or modelName).lower() or '8192' in (hf or modelName).lower()) and '8k' not in dataName.lower():
                return True
            if ('2k' in (hf or modelName).lower() or '2048' in (hf or modelName).lower()) and '8k' in dataName.lower():
                return True
        return False

    def _log_data_mismatch(self, fileHandler, modelName, dataName, SWEEP):
        """Log a data mismatch error."""
        message = f"{self.project},{modelName},{modelName.replace(' ', '_')}_training_sweep_{SWEEP}_{dataName},Training,'DataMismatch',{dataName},{SWEEP},None,N/A\n"
        fileHandler.write(message)
        self.logger.warning(f"Data mismatch detected for {modelName} with {dataName}")

    def _build_training_command(self, modelName, dataName, RDU, SWEEP, hf=None, jobname=None):
        """Build the SNAPI command for training."""
        command = f"snapi job create -p {self.project} "
        command += f"-j {jobname or modelName.replace(' ', '_')}_training_sweep_{SWEEP}_{dataName}_rdu{RDU} "
        command += f"-t train -m '{modelName}' -d {dataName} -r {RDU} -a {self.arch} "
        if hf and SWEEP != 'N':
            command += f"-hf {hf} "
        else:
            command += '''-hp '{"num_iterations":"10", "skip_checkpoint": "true", "save_optimizer_state": "false"}' '''
        return command

    def _run_snapi_command(self, command):
        """Run a SNAPI command and return the result."""
        success, output, error = self.run_command(command)
        if not success:
            self.logger.error(f"SNAPI command failed: {error}")
            return 'Failed'
        if 'Successfully created' in output:
            return 'Succeed'
        elif 'Duplicate job' in output:
            return 'Duplicated'
        else:
            return 'Failed'

    def _log_training_result(self, fileHandler, modelName, res, dataName, SWEEP, RDU, command, hf=None, jobname=None):
        """Log the result of a training job."""
        log_message = f"{self.project},{modelName},{jobname or modelName.replace(' ', '_')}_training_sweep_{SWEEP}_{dataName}_rdu{RDU},Training,{res},{dataName},{SWEEP},{hf or 'None'},{command}\n"
        fileHandler.write(log_message)
        self.logger.info(f"Training job result for {modelName}: {res}")

    def _generate_job_name(self, hf, dataName, RDU, SWEEP):
        """Generate a job name based on the hyperparameter file and other parameters."""
        return f"{hf.split('.yaml')[0].split('../../artifacts/hf/')[-1]}_{dataName.replace(' ','').replace('_','')}_{RDU}rdu_sweep_{SWEEP}"

    def model_inference(self, fileHandler, inference_hf_l, modelName, dataName, SWEEP='N'):
        """
        Run model inference jobs.

        Args:
            fileHandler: File handler for logging.
            inference_hf_l (list): List of inference hyperparameter files.
            modelName (str): Name of the model.
            dataName (str): Name of the dataset.
            SWEEP (str): Sweep mode.

        Returns:
            fileHandler: Updated file handler.
        """
        if len(inference_hf_l) == 0 or SWEEP == 'N':
            command = self._build_inference_command(modelName, dataName)
            res = self._run_snapi_command(command)
            self._log_inference_result(fileHandler, modelName, res, dataName, SWEEP, command)
            return fileHandler

        for hf in inference_hf_l:
            jobname = self._generate_inference_job_name(hf, dataName, SWEEP)
            command = self._build_inference_command(modelName, dataName, SWEEP, hf, jobname)
            res = self._run_snapi_command(command)
            self._log_inference_result(fileHandler, modelName, res, dataName, SWEEP, command, hf, jobname)

        return fileHandler

    def _build_inference_command(self, modelName, dataName, SWEEP='N', hf=None, jobname=None):
        """Build the SNAPI command for inference."""
        command = f"snapi job create -p {self.project} "
        command += f"-j {jobname or modelName.replace(' ', '_')}_inference_sweep_{SWEEP}_{dataName} "
        command += f"-t batch_predict -m '{modelName}' -d {dataName} -a {self.arch} "
        if hf and SWEEP != 'N' and os.path.getsize(hf) > 0:
            command += f"-hf {hf} "
        return command

    def _log_inference_result(self, fileHandler, modelName, res, dataName, SWEEP, command, hf=None, jobname=None):
        """Log the result of an inference job."""
        log_message = f"{self.project},{modelName},{jobname or modelName.replace(' ', '_')}_inference_sweep_{SWEEP}_{dataName},Inference,{res},{dataName},{SWEEP},{hf or 'None'},{command}\n"
        fileHandler.write(log_message)
        self.logger.info(f"Inference job result for {modelName}: {res}")

    def _generate_inference_job_name(self, hf, dataName, SWEEP):
        """Generate a job name for inference based on the hyperparameter file and other parameters."""
        return f"{hf.split('.yaml')[0].split('../../artifacts/hf/')[-1]}_{dataName.replace(' ','').replace('_','')}_sweep_{SWEEP}"

    def model_ndscreening(self, fileHandler, training_hf_l, modelName, dataName, RDU=1, NUMRDU=8):
        """
        Run model screening jobs.

        Args:
            fileHandler: File handler for logging.
            training_hf_l (list): List of training hyperparameter files.
            modelName (str): Name of the model.
            dataName (str): Name of the dataset.
            RDU (int): Number of RDUs to use.
            NUMRDU (int): Total number of RDUs available.

        Returns:
            tuple: (fileHandler, jobname_list)
        """
        if len(training_hf_l) == 0:
            self.logger.warning(f"No training hyperparameter files found for model {modelName}.")
            return None, None

        jobname_list = []
        numtests = int(NUMRDU/RDU)

        for hf in training_hf_l:
            if self._is_data_mismatch(modelName, dataName, hf):
                continue

            if ('8k' in hf or '8192' in hf) and 'GPT' in hf:
                dataName = 'GPT_13B_8k_SS_Toy_Training_Dataset'

            for i in range(numtests):
                jobname = self._generate_screening_job_name(hf, dataName, RDU, i)
                command = self._build_screening_command(modelName, dataName, RDU, hf, jobname)
                res = self._run_snapi_command(command)
                self._log_screening_result(fileHandler, modelName, res, dataName, hf, command, jobname)
                if res == 'Succeed':
                    jobname_list.append(jobname)

        return fileHandler, jobname_list

    def _generate_screening_job_name(self, hf, dataName, RDU, i):
        """Generate a job name for screening based on the hyperparameter file and other parameters."""
        return f"{hf.split('.yaml')[0].split('../../artifacts/hf/')[-1]}_{dataName.replace(' ','').replace('_','')}_{RDU}rdu_test_{i}"

    def _build_screening_command(self, modelName, dataName, RDU, hf, jobname):
        """Build the SNAPI command for screening."""
        return f"snapi job create -p {self.project} -j {jobname} -t train -m '{modelName}' -d {dataName} -hf {hf} -r {RDU} -a {self.arch}"

    def _log_screening_result(self, fileHandler, modelName, res, dataName, hf, command, jobname):
        """Log the result of a screening job."""
        log_message = f"{self.project},{modelName},{jobname},Training,{res},{dataName},{hf},{command}\n"
        fileHandler.write(log_message)
        self.logger.info(f"Screening job result for {modelName}: {res}")

    def model_loadckpt(self, hf, modelName, dataName, RDU=1):
        """
        Load a checkpoint and run a training job.

        Args:
            hf (str): Hyperparameter file path.
            modelName (str): Name of the model.
            dataName (str): Name of the dataset.
            RDU (int): Number of RDUs to use.

        Returns:
            str: Result of the job ('Success', 'Fail', 'InitTrainingFailed', or 'NoCKPT')
        """
        if ('8k' in hf or '8192' in hf) and 'GPT' in hf:
            dataName = 'GPT_13B_8k_SS_Toy_Training_Dataset'

        jobname = self._generate_loadckpt_job_name(hf, dataName, RDU)
        command = self._build_loadckpt_command(modelName, dataName, RDU, hf, jobname)
        
        res = self._run_snapi_command(command)
        if res != 'Succeed':
            return 'InitTrainingFailed'

        self.logger.info(f"Initial training job created for {modelName}")

        ckptid = self._wait_for_job_completion_and_get_checkpoint(jobname)
        if not ckptid:
            return 'NoCKPT'

        newmodel = f"test-{ckptid}"
        self._save_checkpoint_to_modelhub(ckptid, newmodel, jobname)

        self._delete_job(jobname)

        # Start a new job with the loaded checkpoint
        command = self._build_loadckpt_command(newmodel, dataName, RDU, hf, jobname, load_checkpoint=True)
        res = self._run_snapi_command(command)
        if res != 'Succeed':
            return 'Fail'

        final_result = self._wait_for_job_completion(jobname)

        self._delete_job(jobname)
        self._delete_model(newmodel)

        return final_result

    def _generate_loadckpt_job_name(self, hf, dataName, RDU):
        """Generate a job name for checkpoint loading based on the hyperparameter file and other parameters."""
        return f"{hf.split('.yaml')[0].split('../../artifacts/hf/')[-1]}_{dataName.replace(' ','').replace('_','')}_{RDU}rdu"

    def _build_loadckpt_command(self, modelName, dataName, RDU, hf, jobname, load_checkpoint=False):
        """Build the SNAPI command for checkpoint loading."""
        command = f"snapi job create -p {self.project} -j {jobname} -t train -m '{modelName}' -d {dataName} -hf {hf} -r {RDU} -a {self.arch}"
        if load_checkpoint:
            command += " -l"
        return command

    def _wait_for_job_completion_and_get_checkpoint(self, jobname):
        """Wait for a job to complete and return the checkpoint ID if successful."""
        while True:
            status = self._get_job_status(jobname)
            if status == 'STOPPED':
                return None
            elif status in ['EXIT_WITH_0', 'FAILED']:
                break
            time.sleep(120)

        if status == 'EXIT_WITH_0':
            return self._get_checkpoint_id(jobname)
        return None

    def _get_job_status(self, jobname):
        """Get the status of a job."""
        command = f"snapi job info -p {self.project} -j {jobname}"
        success, output, _ = self.run_command(command)
        if success:
            for line in output.splitlines():
                if 'Status' in line:
                    return line.split(': ')[-1].strip()
        return None

    def _get_checkpoint_id(self, jobname):
        """Get the checkpoint ID for a completed job."""
        command = f"snapi checkpoint list -p {self.project} -j {jobname}"
        success, output, _ = self.run_command(command)
        if success:
            for line in output.splitlines():
                parts = line.split()
                if parts:
                    return parts[0]
        return None

    def _save_checkpoint_to_modelhub(self, ckptid, newmodel, jobname):
        """Save a checkpoint to the model hub."""
        command = f"snapi model add -m {ckptid} -n {newmodel} -p {self.project} -j {jobname} -t pretrained"
        self.run_command(command)
        self.logger.info(f"Checkpoint {ckptid} saved as model {newmodel}")
        time.sleep(1000 if '70' not in jobname.lower() else 3000)

    def _delete_job(self, jobname):
        """Delete a job."""
        command = f"snapi job delete -p {self.project} -j {jobname}"
        self.run_command(command)
        self.logger.info(f"Job {jobname} deleted")
        time.sleep(120)

    def _delete_model(self, modelname):
        """Delete a model."""
        command = f"snapi model remove -m {modelname}"
        self.run_command(command)
        self.logger.info(f"Model {modelname} deleted")

    def _wait_for_job_completion(self, jobname):
        """Wait for a job to complete and return the final result."""
        while True:
            status = self._get_job_status(jobname)
            if status == 'STOPPED':
                return 'STOPPED'
            elif status == 'FAILED':
                return 'Fail'
            elif status == 'EXIT_WITH_0':
                return 'Success'
            time.sleep(120)

# End of GeneralModelTestObject class