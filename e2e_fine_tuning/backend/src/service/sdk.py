import logging
from typing import Any, Dict, List, Optional

import e2e_fine_tuning.backend.src.schemas as schemas
from e2e_fine_tuning.backend.src.exceptions.sdk_repository_exceptions import (
    AppFetchError,
    DatasetCreateError,
    DatasetDeleteError,
    DatasetFetchError,
    ProjectCreateError,
    ProjectFetchError,
)
from e2e_fine_tuning.backend.src.repositories.sdk import SnsdkWrapperRepository

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SnsdkWrapperService:
    def __init__(self, repository: SnsdkWrapperRepository) -> None:
        self.repository = repository

    def get_apps(self) -> Optional[schemas.AvailableApps]:
        try:
            logger.info('Fetching available apps')
            available_apps = self.repository.get_apps()
            logger.info(f'Fetched {len(available_apps)} available apps')
            return schemas.AvailableApps(available_apps=available_apps)
        except AppFetchError as e:
            logger.error(f'Error fetching available apps: {e}')
            return None

    def get_app(self, model_family: str) -> Optional[List[str]]:
        available_apps = self.repository.get_apps()
        model_family_apps = [
            app['name'] for app in available_apps if model_family in app['name'].replace(' ', '').lower()
        ]
        return model_family_apps

    def get_projects(self) -> schemas.Projects:
        try:
            logger.info('Fetching projects')
            projects = self.repository.get_projects()
            logger.info(f'Fetched {len(projects)} datasets')
            return schemas.Projects(projects=projects)
        except ProjectFetchError as e:
            logger.error(f'Error fetching projects: {e}')
            return None

    def get_project(self, project_name: str) -> Optional[str]:
        logger.info(f'Fetching project with name: {project_name}')
        project_id = self.repository.get_project(project_name)
        if project_id is None:
            logger.warning(f"Project with name '{project_name}' not found")
            return None
        logger.info(f"Dataset '{project_name}' found")
        return schemas.Project(project_id=project_id)

    def create_project(self, project_name: str, project_description: str) -> Optional[str]:
        try:
            # todo check if it exists
            logger.info(f'Fetching project with name: {project_name}')
            project_id = self.repository.create_project(project_name, project_description)
            logger.info(f"Dataset '{project_name}' found")
            return schemas.Project(project_id=project_id)
        except ProjectCreateError as e:
            logger.error(f'Error creating project {project_name}: {e}')
            raise e

    def get_datasets(self) -> Optional[schemas.Datasets]:
        try:
            logger.info('Fetching datasets')
            datasets = self.repository.get_datasets()
            logger.info(f'Fetched {len(datasets)} datasets')
            return schemas.Datasets(datasets=datasets)
        except DatasetFetchError as e:
            logger.error(f'Error fetching datasets: {e}')
            return None

    def get_dataset(self, dataset_name: str) -> Optional[schemas.Dataset]:
        logger.info(f'Fetching dataset with name: {dataset_name}')
        dataset = self.repository.get_dataset(dataset_name)
        if dataset is None:
            logger.warning(f"Dataset with name '{dataset_name}' not found")
            return None
        logger.info(f"Dataset '{dataset_name}' found")
        return schemas.Dataset(dataset=dataset)

    def delete_dataset(self, dataset_name: str) -> bool:
        dataset = self.get_dataset(dataset_name)
        if dataset is None:
            logger.error(f"Dataset '{dataset_name}' not found.")
            return False
        try:
            logger.info(f'Deleting dataset with name: {dataset_name}')
            response = self.repository.delete_dataset(dataset_name)
            logger.info(f"Dataset '{dataset_name}' deleted successfully")
            return response
        except DatasetDeleteError as e:
            logger.error(f"Error deleting dataset '{dataset_name}': {e}")
            raise e

    def create_dataset(self, dataset_info: str, dataset_config: Dict[str, Any]) -> Optional[dict]:
        dataset = self.get_dataset(dataset_info['dataset_name_sambastudio'])
        if dataset is not None:
            logger.info(f"Dataset {dataset_info['dataset_name_sambastudio']} already exists.")
            raise DatasetFetchError(f"Dataset {dataset_info['dataset_name_sambastudio']} already exists.")

        try:
            new_dataset_id = self.repository.create_dataset(dataset_info['dataset_name_sambastudio'], dataset_config)
            new_dataset = self.get_dataset(dataset_info['dataset_name_sambastudio'])
            if new_dataset is None:
                raise DatasetFetchError(f'Dataset not found')
            logger.info(f'Dataset {new_dataset} created successfully')
            return new_dataset.model_dump() if new_dataset is not None else None
        except DatasetCreateError as e:
            logger.error(f"Error creating dataset {dataset_info['dataset_name_sambastudio']}: {e}")
            raise e
