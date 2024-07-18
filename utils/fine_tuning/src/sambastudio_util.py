import os
import json
import yaml
from snsdk import SnSdk
import logging
from typing import Optional

SNAPI_PATH = "~/.snapi"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

class SnsdkWrapper:
    """init"""

    def __init__(self, config_path=None) -> None:
        self.config = self._load_config(config_path)
        self.dataset_id = "d80f6355-af9d-406b-95c5-c31854f36f2c"
        #self.snapi_path = self.config["sambastudio"]["snapi_path"]

        #if self.snapi_path is None or len(self.snapi_path) == 0:
        #    self.snapi_path = SNAPI_PATH

        #host_url, tenant_id, access_key = self._get_sambastudio_variables()

        self.snsdk_client = SnSdk(
            host_url="https://sjc3-demo1.sambanova.net/",
            access_key="8f73e0b3e0026c2c15996c6ebf3312b3af138d040bae072c292402092b155b84",
            tenant_id="41ceaded-9f08-47ae-aa02-15f39c899618",
        )

        #tenant = self.search_tenant(tenant_id)
        #if tenant is not None:
        #    self.tenant_name = tenant["tenant_name"]
        #else:
        #    raise ValueError(f"Tenant {tenant_id} not found")

    def _load_config(self, file_path):
        """Loads a YAML configuration file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: The configuration data loaded from the YAML file.
        """
        try:
            with open(file_path, "r") as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file {file_path} does not exist.")
        except yaml.scanner.ScannerError:
            raise ValueError(f"Error: The file {file_path} contains invalid yaml.")
        return config

    """Project"""

    def search_project(self, project_name: str):
        search_project_response=self.snsdk_client.search_project(project_name=project_name)
        if search_project_response["status_code"]==200:
            project_id = search_project_response["data"]["project_id"]
            logging.info(f"Project with name '{project_name}' found with id {project_id}")
            return project_id
        else:
            logging.info(f"Project with name '{project_name}' not found")
            return None

    def create_project(
        self, project_name: Optional[str] = None, 
        project_description: Optional[str] = None
        ):
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        if project_description is None:
            project_description = self.config.get["project"]["project_description"]
        project_id=self.search_project(project_name)
        if project_id is None:
            create_project_response = self.snsdk_client.create_project(    
                project_name=project_name,
                description=project_description
                )
            if create_project_response["status_code"]==200:
                project_id = create_project_response["data"]["project_id"]
                logging.info(f"Project with name {project_name} created with id {project_id}")
            else:
                logging.error(f"Failed to create project with name '{project_name}'. Details: {create_project_response['detail']}")
                raise Exception(f"Error message: {create_project_response['detail']}")
        else:
            logging.info(f"Project with name '{project_name}' already exists with id '{project_id}', using it")
        return project_id
    
    def list_projects(self):
        list_projects_response = self.snsdk_client.list_projects()
        if list_projects_response["status_code"] == 200:
            projects=[]
            for project in list_projects_response["data"].get("projects"):
                projects.append({k:v for k,v in project.items() if k in ["project_name", "project_id","status","user_id"]})
            return projects
        else:
            logging.error(f"Failed to list projects. Details: {list_projects_response['detail']}")
            raise Exception(f"Error message: {list_projects_response['detail']}")
        
    def delete_project(self,  project_name:Optional[str] = None, project_id:Optional[str] = None):
        if project_id is not None:
            project=project_id
        elif project_name is not None:
            project=project_name
        else:
            project = self.config["project"]["project_name"]
        delete_project_response = self.snsdk_client.delete_project(project=project)
        if delete_project_response["status_code"] == 200:
            logging.info(f"Project with name or id '{project_name}' deleted")
        else:
            logging.error(f"Failed to delete project with name or id '{project_name}'. Details: {delete_project_response}")
            raise Exception(f"Error message: {delete_project_response}")
        
    """Job"""
    
    def run_job(self,
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
                ):
        # check if project selected exists
        if project_name is not None:
            project_id = self.search_project(project_name)
        else:
            project_id = self.search_project(self.config["project"]["project_name"])
            
        if project_id is None:
            raise Exception(f"project with name '{project_name}' not found")
        
        # check if model selected is found and is trainable
        if model is not None:
            model_id = self.search_trainable_model(model)
        else: 
            model_id = self.search_trainable_model(self.config["job"]["model"])
            
        if model_id is None:
            raise Exception(f"model with name '{model}' not found")
            
        # check if dataset exist
        if dataset_name is not None:
            dataset_id = self.search_dataset(dataset_name)
        else:
            dataset_id = self.search_dataset(self.config["dataset"]["dataset_name"])
            
        if dataset_id is None:
            raise Exception(f"dataset with name '{dataset_name}' not found")
        
        # create job
        create_job_response=self.snsdk_client.create_job(
            project = project_id,
            job_name = job_name or self.config["job"]["job_name"],
            description = job_description or self.config["job"]["job_description"],
            job_type = job_type or self.config["job"]["job_type"],
            model_checkpoint = model_id,
            dataset = dataset_id,
            parallel_instances = parallel_instances or self.config["job"]["parallel_instances"],
            load_state = load_state or self.config["job"]["load_state"],
            sub_path = sub_path or self.config["job"]["sub_path"],
            rdu_arch = rdu_arch or self.config["sambastudio"]["rdu_arch"],
            hyperparams = json.dumps(hyperparams or self.config["job"]["hyperparams"]),
        )
        
        if create_job_response["status_code"] == 200:
            job_id = create_job_response["job_id"]
            logging.info(f"Job with name '{job_name}' created: '{create_job_response}'")
            return job_id
        else:
            logging.error(f"Failed to create job with name '{job_name or self.config['job']['job_name']}'. Details: {create_job_response}")
            raise Exception(f"Error message: {create_job_response}")

    def search_job(self, job_name: Optional[str] = None, project_name: Optional[str]=None):
        if job_name is None:
            job_name = self.config["job"]["job_name"]
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        search_job_response = self.snsdk_client.search_job(
            project=project_name,
            job_name=job_name
            )
        if search_job_response["status_code"] == 200:
             job_id = search_job_response["data"]["job_id"]
             logging.info(f"Job with name '{job_name}' in project '{project_name}' found with id '{job_id}'")
             return job_id
        else:
            logging.info(f"Job with name '{job_name}' in project '{project_name}' not found")
            return None
        
    def check_job_progress(
        self,
        project_name: Optional[str] = None,
        job_name: Optional[str] = None
        ):
        if job_name is None:
            job_name = self.config["job"]["job_name"]
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")
        check_job_progress_response = self.snsdk_client.job_info(
            project = project_name,
            job = job_id
            )
        if check_job_progress_response["status_code"] == 200:
            #TODO implement logic to filter out response, blocked by authorization error
            #job_progress = {}
            #logging.info(f"Job '{job_name}' with progress status: {job_progress}")
            return job_progress
        else:
            logging.error(f"Failed to check job progress. Details: {check_job_progress_response}")
            raise Exception(f"Error message: {check_job_progress_response}")
        
    def list_jobs(self, project_name: Optional[str]):
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None:
            logging.info(f"Project '{project_name}' not found listing all user jobs")
        list_jobs_response = self.snsdk_client.list_jobs(project_id=project_id)
        print(list_jobs_response)
        if list_jobs_response["status_code"] == 200:
            jobs=[]
            for job in list_jobs_response["jobs"]:
                jobs.append(
                    {k:v for k,v in job.items() if k in ["job_name", "job_id", "job_type", "project_id", "status"]}
                    )
            return jobs
        else:
            logging.error(f"Failed to list jobs. Details: {list_jobs_response['detail']}")
            raise Exception(f"Error message: {list_jobs_response['detail']}")

    def delete_job(self, project_name: Optional[str]=None, job_name: Optional[str]=None):
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        if job_name is None:
            job_name = self.config["job"]["job_name"]
        delete_job_response = self.snsdk_client.delete_job(
            project = project_name,
            job = job_name
        )
        if delete_job_response["status_code"] == 200:
            logging.info(f"Job with name '{job_name}' in project '{project_name}' deleted")
            #TODO check if working, blocked by authorization error
        else:
            logging.error(f"Failed to delete job with name '{job_name}' in project '{project_name}'. Details: {delete_job_response}")
            raise Exception(f"Error message: {delete_job_response}")

    """checkpoints"""

    def list_checkpoints():
        # check metrics: loss
        pass

    """endpoint"""

    def create_endpoint():
        pass

    def check_endpoint_progress():
        # check rdus available
        pass

    """generic stuff"""
    def list_models(self):
        list_models_response = self.snsdk_client.list_models()
        if list_models_response["status_code"] == 200:
            models={}
            for model in list_models_response["models"]:
                if set(["train","deploy"]).issubset(model.get("jobTypes")):
                    models[model.get("model_checkpoint_name")] = model.get("model_id")
            return models
        else:
            logging.error(f"Failed to list models. Details: {list_models_response['detail']}")
            raise Exception(f"Error message: {list_models_response['detail']}")
    
    def search_model(self, model_name: str):
        search_model_response=self.snsdk_client.search_model(model_name=model_name) 
        if search_model_response["status_code"]==200:
            model_id = search_model_response["data"]["model_id"]
            logging.info(f"Model with name '{model_name}' found with id {model_id}")
            return model_id
        else:
            logging.info(f"Project with name '{model_name}' not found")
            return None
    
    def search_trainable_model(self, model_name):
        models = self.list_models()
        model_id = models.get(model_name)
        if model_id is not None:
            logging.info(f"Model '{model_name}' with id '{model_id}' available for training and deployment found") 
            return model_id
        else:
            logging.info(f"Model '{model_name}' available for training and deployment not found")
            return None
            
    def search_dataset(self, dataset_name):
        search_dataset_response = self.snsdk_client.search_dataset(dataset_name=dataset_name) 
        if search_dataset_response["status_code"]==200:
            dataset_id = search_dataset_response["data"]["dataset_id"]
            logging.info(f"Dataset with name '{dataset_name}' found with id {dataset_id}")
            return dataset_id
        else:
            logging.info(f"Dataset with name '{dataset_name}' not found")
            return None