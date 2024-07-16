class SnsdkWrapper:
    """init"""

    def __init__(self, config_path=None) -> None:
        # read configs
        #   - project name, description
        #   - job name, description, hyperparams
        #   - data set name, description
        #   - env variables
        #   - other input data if needed
        pass

    """Dataset"""

    def search_dataset(self, dataset_name):
        pass

    def delete_dataset(self, dataset_name):
        pass

    def create_dataset(self, path):
        # search first, if it already exists, return id, if not create it
        # to be generic, in theory it should be to upload files from a local folder
        # if creation outputs error, show it and stop e2e process
        pass

    def check_dataset_creation_progress(self, dataset_name):
        # check progress in case data is huge
        # e2e process has to wait while this is working, showing progress
        pass

    """Project"""

    def search_project(self, project_name):
        pass

    def create_project(self):
        # search first, if it already exists, return id, if not create it
        # if creation outputs error, show it and stop e2e process
        pass

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

    def list_projects():
        pass

    def list_datasets():
        pass