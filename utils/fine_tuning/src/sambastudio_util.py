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
            host_url="https://sjc3-demo1.sambanova.net",
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

    def search_project(self, project_name: str = None):
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

    def create_project(
        self, project_name: Optional[str] = None, 
        project_description: Optional[str] = None
        ):
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        if project_description is None:
            project_description = self.config["project"]["project_description"]
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
    
    def list_projects(self, verbose: Optional[bool] = False):
        list_projects_response = self.snsdk_client.list_projects()
        if list_projects_response["status_code"] == 200:
            projects=[]
            for project in list_projects_response["data"].get("projects"):
                if verbose:
                    projects.append({k:v for k,v in project.items()})
                else:
                    projects.append({k:v for k,v in project.items() if k in ["project_name", "project_id","status","user_id"]})
            return projects
        else:
            logging.error(f"Failed to list projects. Details: {list_projects_response['detail']}")
            raise Exception(f"Error message: {list_projects_response['detail']}")
        
    def delete_project(self,  project_name: Optional[str] = None):
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name=project_name)
        if project_id is None:
            raise Exception(f"Project with name '{project_name}' not found")
        delete_project_response = self.snsdk_client.delete_project(project=project_id)
        if delete_project_response["status_code"] == 200:
            logging.info(f"Project with name '{project_name}' deleted")
        else:
            logging.error(f"Failed to delete project with name or id '{project_name}'. Details: {delete_project_response}")
            raise Exception(f"Error message: {delete_project_response}")
    
    """models"""
    def list_models(self, filter: Optional[list[str]] = [], verbose: Optional[bool] = False):
        #filter = ['train', 'batch_predict', 'deploy']
        list_models_response = self.snsdk_client.list_models()
        if list_models_response["status_code"] == 200:
            models=[]
            for model in list_models_response["models"]:
                if set(filter).issubset(model.get("jobTypes")):
                    if verbose:
                        models.append({k:v for k,v in model.items()})
                    else:
                        models.append({k:v for k,v in model.items() if k in ["model_checkpoint_name","model_id"]}) 
            return models
        else:
            logging.error(f"Failed to list models. Details: {list_models_response['detail']}")
            raise Exception(f"Error message: {list_models_response['detail']}")
    
    def search_model(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = self.config["job"]["model"]
        search_model_response=self.snsdk_client.search_model(model_name=model_name) 
        if search_model_response["status_code"]==200:
            model_id = search_model_response["data"]["model_id"]
            logging.info(f"Model with name '{model_name}' found with id {model_id}")
            return model_id
        else:
            logging.info(f"Project with name '{model_name}' not found")
            return None
    
    def search_trainable_model(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = self.config["job"]["model"]
        models = self.list_models(filter=['train', 'deploy'])
        model_id = [model["model_id"] for model in models if model["model_checkpoint_name"]==model_name]
        if len(model_id)>0:
            logging.info(f"Model '{model_name}' with id '{model_id[0]}' available for training and deployment found") 
            return model_id[0]
        else:
            logging.info(f"Model '{model_name}' available for training and deployment not found")
            return None
    
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
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")
        
        # check if model selected is found and is trainable
        if model is None:
            model = self.config["job"]["model"]
        model_id = self.search_trainable_model(model)   
        if model_id is None:
            raise Exception(f"model with name '{model}' not found")
            
        # check if dataset exist
        if dataset_name is None:
            dataset_name = self.config["dataset"]["dataset_name"]
        dataset_id = self.search_dataset(dataset_name)
        if dataset_id is None:
            raise Exception(f"dataset with name '{dataset_name}' not found")
            
        # create job
        create_job_response=self.snsdk_client.create_job(
            project = project_id,
            job_name = job_name or self.config.get("job",{}).get("job_name"),
            description = job_description or self.config.get("job",{}).get("job_description"),
            job_type = job_type or self.config.get("job",{}).get("job_type"),
            model_checkpoint = model_id,
            dataset = dataset_id,
            parallel_instances = parallel_instances or self.config.get("job",{}).get("parallel_instances"),
            load_state = load_state or self.config.get("job",{}).get("load_state"),
            sub_path = sub_path or self.config.get("job",{}).get("sub_path"),
            rdu_arch = rdu_arch or self.config.get("sambastudio",{}).get("rdu_arch"),
            hyperparams = json.dumps(hyperparams or self.config.get("job",{}).get("hyperparams")),
        )
        
        if create_job_response["status_code"] == 200:
            job_id = create_job_response["job_id"]
            logging.info(f"Job with name '{job_name or self.config.get('job',{}).get('job_name')}' created: '{create_job_response}'")
            return job_id
        else:
            logging.error(f"Failed to create job with name '{job_name or self.config.get('job',{}).get('job_name')}'. Details: {create_job_response}")
            raise Exception(f"Error message: {create_job_response}")

    def search_job(self, job_name: Optional[str] = None, project_name: Optional[str]=None):
        if job_name is None:
            job_name = self.config["job"]["job_name"]
            
        # check if project selected exists
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")
        
        # search for job 
        search_job_response = self.snsdk_client.search_job(
            project=project_id,
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
        # check if project selected exists
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")
        
        # check if job selected exists
        if job_name is None:
            job_name = self.config["job"]["job_name"]
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")
        
        # check job progress
        check_job_progress_response = self.snsdk_client.job_info(
            project = project_id,
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
        
    def list_jobs(self, project_name: Optional[str] = None, verbose: Optional[bool] = False):
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None:
            logging.info(f"Project '{project_name}' not found listing all user jobs")
        list_jobs_response = self.snsdk_client.list_jobs(project_id=project_id)
        if list_jobs_response["status_code"] == 200:
            jobs=[]
            for job in list_jobs_response["jobs"]:
                if verbose:
                    jobs.append({k:v for k,v in job.items()})
                else:
                    jobs.append(
                        {k:v for k,v in job.items() if k in ["job_name", "job_id", "job_type", "project_id", "status"]}
                        )
            return jobs
        else:
            logging.error(f"Failed to list jobs. Details: {list_jobs_response['detail']}")
            raise Exception(f"Error message: {list_jobs_response['detail']}")

    def delete_job(self, project_name: Optional[str]=None, job_name: Optional[str]=None):
        # check if project selected exists
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")  
            
        #check if job selected exists
        if job_name is None:
            job_name = self.config["job"]["job_name"]
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")  
            
        # delete job
        delete_job_response = self.snsdk_client.delete_job(
            project = project_id,
            job = job_id
        )
        if delete_job_response["status_code"] == 200:
            logging.info(f"Job with name '{job_name}' in project '{project_name}' deleted")
            #TODO check if working, blocked by authorization error
        else:
            logging.error(f"Failed to delete job with name '{job_name}' in project '{project_name}'. Details: {delete_job_response}")
            raise Exception(f"Error message: {delete_job_response}")

    """checkpoints"""

    def list_checkpoints(self, project_name: Optional[str]=None, job_name: Optional[str]=None, verbose: Optional[bool]=False):
        # check if project selected exists
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None: 
            raise Exception(f"Project '{project_name}' not found")
            
        # check if job selected exists
        if job_name is None:
            job_name = self.config["job"]["job_name"]
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")
        
        list_checkpoints_response=self.snsdk_client.list_checkpoints(
            project = project_id,
            job = job_id
        )
        if list_checkpoints_response["status_code"] == 200:
            checkpoints=[]
            #TODO implement logic to filter out response, blocked by authorization error
            #for checkpoint in list_checkpoints_response:
            #    if verbose:
            #        checkpoints.append({k:v for k,v in checkpoint.items()})
            #    else:
            #        checkpoints.append(
            #            {k:v for k,v in checkpoint.items() if k in ["checkpoint_id", "status"]}
            #        )
            return checkpoints
        else:
            logging.error(f"Failed to list checkpoints. Details: {list_checkpoints_response}")
            raise Exception(f"Error message: {list_checkpoints_response}")
        
    
    def promote_checkpoint(self, 
                           checkpoint_id: Optional[str] = None,
                           project_name: Optional[str] = None,
                           job_name: Optional[str] = None,
                           model_name: Optional[str] = None,
                           model_description: Optional[str] = None,
                           model_type: Optional[str] = None,
                           ):
        
        # check if project selected exists
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")  
            
        # check if job selected exists
        if job_name is None:
            job_name = self.config["job"]["job_name"]
        job_id = self.search_job(project_name=project_name, job_name=job_name)
        if job_id is None:
            raise Exception(f"Job with name '{job_name}' in project '{project_name}' not found")
        
        if checkpoint_id is None:
            checkpoint_id = self.config["model_checkpoint"]["model_checkpoint_id"]
            if not checkpoint_id:
                raise Exception("No model checkpoint_id provided")
            # TODO: check if checkpoint in list checkpoints list blocked because authorization error in lists checkpoints method
            # if checkpoint_id not in self.list_checkpoints(project_name=project_name, job_name=job_name):
            #     raise Exception(f"Checkpoint id '{checkpoint_id}' not found in job '{job_name}'")
        
        add_model_response = self.snsdk_client.add_model(
            project = project_name or self.config.get("project",{}).get("project_name"),
            job = job_name or self.config.get("job",{}).get("job_name"),
            model_checkpoint=checkpoint_id,
            model_checkpoint_name=model_name or self.config.get("model_checkpoint",{}).get("model_name"),
            description=model_description or self.config.get("model_checkpoint",{}).get("model_description"),
            checkpoint_type=model_type or self.config.get("model_checkpoint",{}).get("model_type")
        )
        
        if add_model_response["status_code"] == 200:
            logging.info(f"Model checkpoint '{checkpoint_id}' promoted to model '{model_name}'")
            #TODO test blocked because of authorization error 
        else:
            logging.error(f"Failed to promote checkpoint '{checkpoint_id}' to model. Details: {add_model_response}")
            raise Exception(f"Error message: {add_model_response}")
    
    def delete_checkpoint(self, checkpoint: str = None):
        #check if checkpoint exist
        if checkpoint is None:
            checkpoint = self.config["model_checkpoint"]["model_checkpoint_id"]
            if not checkpoint:
                raise Exception("No model checkpoint_id provided")
        delete_checkpoint_response = self.snsdk_client.delete_checkpoint(checkpoint=checkpoint)
        if delete_checkpoint_response["status_code"] == 200:
            logging.info(f"Model checkpoint '{checkpoint}' deleted")
        else:
            logging.error(f"Failed to delete checkpoint '{checkpoint}'. Details: {delete_checkpoint_response}")
            raise Exception(f"Error message: {delete_checkpoint_response}")

    """endpoint"""
    def list_endpoints(self, project_name: Optional[str] = None, verbose: Optional[bool] = None):
        #check if project exist
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None:
            logging.info(f"Project '{project_name}' not found listing all user endpoints")
            
        list_endpoints_response = self.snsdk_client.list_endpoints(project=project_id)
        if list_endpoints_response["status_code"] == 200:
            endpoints=[]
            for endpoint in list_endpoints_response["endpoints"]:
                if verbose:
                    endpoints.append({k:v for k,v in endpoint.items()})
                else:
                    endpoints.append(
                        {k:v for k,v in endpoint.items() if k in ["name", "id", "project_id",  "status", ]}
                        )
            return endpoints
        else:
            logging.error(f"Failed to list endpoints. Details: {list_endpoints_response['detail']}")
            raise Exception(f"Error message: {list_endpoints_response['detail']}")
        
    def create_endpoint(self,
                        project_name: Optional[str]=None,
                        endpoint_name: Optional[str]=None,
                        endpoint_description: Optional[str]=None,
                        model_name: Optional[str]=None,
                        instances: Optional[str]=None,
                        rdu_arch: Optional[str]=None,
                        hyperparams: Optional[str]=None,
                        ):
        # check if project selected exists
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")
            
        # check if model selected exists   
        if model_name is None:
            model_name = self.config["model_checkpoint"]["model_name"]
        model_id = self.search_model(model_name=model_name)
        if model_id is None:
            raise Exception(f"Model with name '{model_name}' not found")
        
        # check if endpoint selected exists
        if endpoint_name is None:
            endpoint_name = self.config["endpoint"]["endpoint_name"]
        endpoint_id = self.search_endpoint(project = project_id, endpoint_name = endpoint_name)
        if endpoint_id is not None:
            logging.info(f"Endpoint with name '{endpoint_name}' not created it already exist with id {endpoint_id}")
            return endpoint_id
        
        create_endpoint_response=self.snsdk_client.create_endpoint(
            project=project_id,
            endpoint_name=endpoint_name,
            description=endpoint_description or self.config.get("endpoint",{}).get("endpoint_description"),
            model_checkpoint=model_id,
            instances=instances or self.config.get("endpoint",{}).get("endpoint_instances"),
            rdu_arch = rdu_arch or self.config["sambastudio"]["rdu_arch"],
            hyperparams = json.dumps(hyperparams or self.config.get("endpoint",{}).get("hyperparams")),
        )
        
        if create_endpoint_response["status_code"] == 200:
            logging.info(f"Endpoint '{endpoint_name}' created")
            endpoint_id = create_endpoint_response["id"]
            return endpoint_id
            
        else:
            logging.error(f"Failed to create endpoint {endpoint_name}. Details: {create_endpoint_response}")
            raise Exception(f"Error message: {create_endpoint_response}") 
        

    def search_endpoint(self, project: Optional[str]=None, endpoint_name: Optional[str]=None):
        if project is None:
            project = self.config["project"]["project_name"]
        if endpoint_name is None:
            endpoint_name = self.config["endpoint"]["endpoint_name"]
        endpoint_info_response = self.snsdk_client.endpoint_info(
            project = project, 
            endpoint = endpoint_name
            )
         
        if endpoint_info_response["status_code"] == 200:
            endpoint_id = endpoint_info_response["id"]
            return endpoint_id
        elif endpoint_info_response["status_code"] == 404:
            logging.info(f"Endpoint with name '{endpoint_name}' not found in project '{project}'")
            return None
        else:
            logging.error(f"Failed to retrieve information for endpoint Details: {endpoint_info_response}")
            raise Exception(f"Error message: {endpoint_info_response}")         
        
        
    def get_endpoint_details(self, project_name: Optional[str]=None, endpoint_name: Optional[str]=None):
        # check if project selected exists
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")  
        
        if endpoint_name is None:
            endpoint_name = self.config["endpoint"]["endpoint_name"]
            
        endpoint_info_response = self.snsdk_client.endpoint_info(
            project = project_id, 
            endpoint = endpoint_name
            )
        
        if endpoint_info_response["status_code"] == 200:
            endpoint_url = endpoint_info_response["url"]
            endpoint_details={
                "status":endpoint_info_response["status"],
                "url": endpoint_url,
                "langchain wrapper env":{
                    "SAMBASTUDIO_BASE_URL": self.snsdk_client.host_url,
                    "SAMBASTUDIO_BASE_URI": "/".join(endpoint_url.split("/")[1:4]),
                    "SAMBASTUDIO_PROJECT_ID": endpoint_url.split("/")[-2],
                    "SAMBASTUDIO_ENDPOINT_ID": endpoint_url.split("/")[-1] ,
                    "SAMBASTUDIO_API_KEY": endpoint_info_response["api_key"],
                }
            }
            return endpoint_details
        else:
            logging.error(f"Failed to get details for endpoint '{endpoint_name}' in project '{project_name}'. Details: {endpoint_info_response}")
            raise Exception(f"Error message: {endpoint_info_response}")
        
    def delete_endpoint(self, project_name: Optional[str]=None, endpoint_name: Optional[str]=None):
        # check if project selected exists
        if project_name is None:
            project_name = self.config["project"]["project_name"]
        project_id = self.search_project(project_name)
        if project_id is None:
            raise Exception(f"Project '{project_name}' not found")  
        
        # check if endpoint selected exists
        if endpoint_name is None:
            endpoint_name = self.config["endpoint"]["endpoint_name"]
        endpoint_id = self.search_endpoint(project=project_id, endpoint_name = endpoint_name)
        if endpoint_id is None:
            raise Exception(f"Endpoint with name '{endpoint_name}' not found in project '{project_name}'")
        
        delete_endpoint_response = self.snsdk_client.delete_endpoint(
            project=project_id, 
            endpoint=endpoint_id
            )
        
        if delete_endpoint_response["status_code"] == 200:
            logging.info(f"Endpoint '{endpoint_name}' deleted in project '{project_name}'")
            
        else:
            logging.error(f"Failed to delete endpoint '{endpoint_name}' in project '{project_name}'. Details: {delete_endpoint_response}")
            raise Exception(f"Error message: {delete_endpoint_response}")
        
    """datasets"""
            
    def search_dataset(self, dataset_name):
        search_dataset_response = self.snsdk_client.search_dataset(dataset_name=dataset_name) 
        if search_dataset_response["status_code"]==200:
            dataset_id = search_dataset_response["data"]["dataset_id"]
            logging.info(f"Dataset with name '{dataset_name}' found with id {dataset_id}")
            return dataset_id
        else:
            logging.info(f"Dataset with name '{dataset_name}' not found")
            return None