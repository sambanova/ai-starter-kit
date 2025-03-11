import logging
from typing import Any, Dict, List, Optional

from e2e_fine_tuning.backend.src.exceptions.sdk_repository_exceptions import (
    AppFetchError,
    DatasetCreateError,
    DatasetDeleteError,
    DatasetFetchError,
)
from utils.fine_tuning.src.snsdk_wrapper import SnsdkWrapper

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SnsdkWrapperRepository:
    def __init__(self, snsdk: SnsdkWrapper) -> None:
        self.__snsdk = snsdk

    def get_apps(self) -> List[Dict[str, str]]:
        try:
            available_apps = self.__snsdk.list_apps()
            return available_apps
        except Exception as e:
            raise AppFetchError(f'Error fetching available apps: {e}')

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

    def create_dataset(self, dataset_name: str, dataset_config: Dict[str, Any]) -> Optional[dict]:
        try:
            self.__snsdk.create_dataset(
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
            return dataset_name
        except Exception as e:
            raise DatasetCreateError(dataset_name, str(e))
