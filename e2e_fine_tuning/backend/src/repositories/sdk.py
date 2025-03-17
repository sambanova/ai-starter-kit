from typing import Any, Dict, List, Optional

from e2e_fine_tuning.backend.src.exceptions.sdk_repository_exceptions import (
    AppFetchError,
    DatasetCreateError,
    DatasetDeleteError,
    DatasetFetchError,
    ProjectCreateError,
    ProjectFetchError,
)
from utils.fine_tuning.src.snsdk_wrapper import SnsdkWrapper


class SnsdkWrapperRepository:
    def __init__(self, snsdk: SnsdkWrapper) -> None:
        self.__snsdk = snsdk

    def get_apps(self) -> List[Dict[str, str]]:
        try:
            available_apps = self.__snsdk.list_apps()
            return available_apps
        except Exception as e:
            raise AppFetchError(f'Error fetching available apps: {e}')

    def get_project(self, project_name: str) -> Optional[str]:
        project_id = self.__snsdk.search_project(project_name)
        return project_id

    def get_projects(self) -> List[Dict[str, Any]]:
        try:
            projects = self.__snsdk.list_projects()
            return projects
        except ProjectFetchError as e:
            raise ProjectFetchError(f'Error fetching projects: {e}')

    def create_project(self, project_name: str, project_description: str) -> str:
        try:
            project_id = self.__snsdk.create_project(project_name=project_name, project_description=project_description)
            return project_id
        except ProjectCreateError as e:
            raise ProjectCreateError(f'Error creating project: {e}')

    def get_datasets(self) -> List[Dict[str, str]]:
        try:
            datasets = self.__snsdk.list_datasets()
            return datasets
        except Exception as e:
            raise DatasetFetchError(f'Error fetching datasets: {e}')

    def get_dataset(self, dataset_name: str) -> Optional[Dict[str, str]]:
        dataset = self.__snsdk.search_dataset(dataset_name)
        return dataset

    def delete_dataset(self, dataset_name: str) -> bool:
        try:
            self.__snsdk.delete_dataset(dataset_name)
            return True
        except Exception as e:
            raise DatasetDeleteError(dataset_name, str(e))

    def create_dataset(self, dataset_name: str, dataset_config: Dict[str, Any]) -> str:
        try:
            dataset_id = self.__snsdk.create_dataset(
                dataset_path=dataset_config['dataset_path'],
                dataset_name=dataset_config['dataset_name'],
                dataset_description=dataset_config['dataset_description'],
                dataset_job_types=dataset_config['dataset_job_types'],
                dataset_source_type=dataset_config['dataset_source_type'],
                dataset_language=dataset_config['dataset_language'],
                dataset_url=dataset_config['dataset_url'],
                dataset_apps_availability=dataset_config['dataset_apps_availability'],
                dataset_filetype=dataset_config['dataset_filetype'],
                dataset_metadata=dataset_config['dataset_metadata'],
            )
            return dataset_id
        except Exception as e:
            raise DatasetCreateError(dataset_name, str(e))
