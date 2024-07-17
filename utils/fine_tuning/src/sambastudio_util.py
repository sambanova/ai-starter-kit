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
        self.project_id = None
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

    def search_project(self, project_name: Optional[str] = None):
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        search_project_response=self.snsdk_client.search_project(project_name=project_name)
        if search_project_response["status_code"]==200:
            project_id = search_project_response["data"]["project_id"]
            logging.info(f"Project with name '{project_name}' found with id {project_id}")
            return project_id
        else:
            logging.info(f"Project with name '{project_name}' not found")
            return None

    def create_project(self, project_name: Optional[str] = None):
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id=self.search_project(project_name)
        if project_id is None:
            create_project_response = self.snsdk_client.create_project(    
                project_name=project_name,
                description=self.config["project"]["project_description"]
                )
            if create_project_response["status_code"]==200:
                self.project_id = create_project_response["data"]["project_id"]
                logging.info(f"Project with name {project_name} created with id {self.project_id}")
            else:
                logging.error(f"Failed to create project with name '{project_name}'. Details: {create_project_response['detail']}")
                raise Exception(f"Error message: {create_project_response['detail']}")
        else:
            self.project_id=project_id
            logging.info(f"Project with name '{project_name}' already exists with id '{self.project_id}', using it")
        return self.project_id
    
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
            logging.error(f"Failed to delete project with name or id '{project_name}'. Details: {delete_project_response['detail']}")
            raise Exception(f"Error message: {delete_project_response['detail']}")
        print(delete_project_response)
        
    """Job"""

    def run_job(self, dataset_name):
        # runs a job
        # has to consider hyperparams for finetuning
        # if run outputs error, show it and stop e2e process
        pass

    def check_job_progress(self, job_id):
        # checks job progress
        # e2e process has to wait while this is working, showing progress
        pass

    def delete_job(self, job_id):
        pass

    def retrieve_results(self, job_id):
        # todo for batch inference
        pass

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

    def list_models():
        pass

    def list_tenants():
        pass
    
    def list_datasets():
        pass