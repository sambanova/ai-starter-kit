import logging
from typing import Optional

from celery import current_task
from celery.app import Celery
from fastapi import HTTPException, status

from e2e_fine_tuning.backend.src.config import settings
from e2e_fine_tuning.backend.src.dependencies import get_byoc_service, get_hf_handler, get_sdk_service
from e2e_fine_tuning.backend.src.exceptions.sdk_repository_exceptions import (
    DatasetFetchError,
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

redis_url = settings.redis_url
celery_app = Celery(__name__, broker=redis_url, backend=redis_url)

celery_app.conf.update(
    result_expires=3600,
)

hf_handler = get_hf_handler()
sdk_service = get_sdk_service()
byoc_service = get_byoc_service()


@celery_app.task(name='create_checkpoint_task')
def create_checkpoint_task(checkpoint_info: dict) -> str:
    logger.info(f"Downloading model:  {checkpoint_info['hf_model_name']}")
    current_task.update_state(state='PROGRESS', meta={'status': 'Downloading checkpoint'})
    response = hf_handler.download_hf_model(checkpoint_info['hf_model_name'])
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with name '{checkpoint_info['hf_checkpoint']}' not found in hugging_face or params invalid",
        )
    logger.info(f'Creating checkpoint')
    checkpoint = byoc_service.create_checkpoint_config(checkpoint_info)
    model_id = byoc_service.upload_checkpoint(checkpoint_info['messages'])
    return model_id


@celery_app.task(name='create_dataset_task')
def create_dataset_task(dataset_info: dict) -> Optional[dict]:
    logger.info(f"Creating dataset: {dataset_info['dataset_name_sambastudio']}")
    try:
        current_task.update_state(state='PROGRESS', meta={'status': 'Downloading dataset'})
        dataset_config = hf_handler.download_hf_dataset(dataset_info)
        if dataset_config is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with name '{dataset_info['hf_dataset']}' not found in hugging_face or params invalid",
            )

        apps = sdk_service.get_app(dataset_info['model_family'])
        if len(apps) == 0:
            logger.error(f"No apps found for model family: {dataset_info['model_family']}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"no apps found for model family: {dataset_info['model_family']}",
            )

        dataset_config['dataset_apps_availability'] = apps
        logger.info(f'Dataset apps availability: {len(apps)} found')

        current_task.update_state(state='PROGRESS', meta={'status': 'Creating dataset'})
        result = sdk_service.create_dataset(dataset_info, dataset_config)

        return result

    except HTTPException as e:
        raise e

    except DatasetFetchError as e:
        logger.error(f'Error creating dataset: {e}')
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset {dataset_info['dataset_name_sambastudio']} already exists.",
        )

    except Exception as e:
        logger.error(f'Error creating dataset: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='internal error occurred while creating dataset'
        )
