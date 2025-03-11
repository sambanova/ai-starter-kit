import os
import sys

import e2e_fine_tuning.backend.src.schemas as schemas
from e2e_fine_tuning.backend.src.config import settings
from e2e_fine_tuning.backend.src.exceptions.sdk_repository_exceptions import (
    DatasetFetchError,
)
from e2e_fine_tuning.backend.src.repositories.byoc import BYOCRepository
from e2e_fine_tuning.backend.src.repositories.sdk import SnsdkWrapperRepository
from e2e_fine_tuning.backend.src.routes import dataset
from e2e_fine_tuning.backend.src.service.hf import HuggingFaceHandler
from e2e_fine_tuning.backend.src.service.sdk import SnsdkWrapperService
from e2e_fine_tuning.backend.src.utils_functions import create_folder
from utils.byoc.src.snsdk_byoc_wrapper import BYOC

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(repo_dir)

import logging

from celery import current_task
from celery.app import Celery
from celery.result import AsyncResult
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi

from e2e_fine_tuning.backend.src.dependencies import get_byoc_service, get_sdk_service
from utils.fine_tuning.src.snsdk_wrapper import SnsdkWrapper

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

data_dir, target_dir = create_folder()
data_dir, target_dir_model = create_folder(target_dir='models')

sdk_service = get_sdk_service()
byoc_service = get_byoc_service()

hf_client = HfApi()
hf_handler = HuggingFaceHandler(hf_client, data_dir, target_dir_model)

redis_url = settings.redis_url
celery_app = Celery(__name__, broker=redis_url, backend=redis_url)

app.include_router(dataset.router)


@celery_app.task(name='create_checkpoint_task')
def create_checkpoint_task(checkpoint_info: dict):
    logger.info(f"Downloading model:  {checkpoint_info['hf_model_name']}")
    current_task.update_state(state='PROGRESS', meta={'status': 'Downloading checkpoint'})
    response = hf_handler.download_hf_model(checkpoint_info['hf_model_name'])
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with name '{checkpoint_info['hf_checkpoint']}' not found in hugging_face or params invalid",
        )
    logger.info(f"Creating checkpoint")
    checkpoint = byoc_service.create_checkpoint_config(checkpoint_info)
    model_id = byoc_service.upload_checkpoint(checkpoint_info['messages'])
    return model_id


@celery_app.task(name='create_dataset_task')
def create_dataset_task(dataset_info: dict):
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


@app.post('/dataset')
def create_dataset(dataset_info: schemas.DatasetCreate):
    task = create_dataset_task.delay(dataset_info.model_dump())
    return JSONResponse({'task_id': task.id})


@app.post('/checkpoint')
def create_checkpoint(checkpoint_info: schemas.CheckPointCreate):
    task = create_checkpoint_task.delay(checkpoint_info.model_dump())
    return JSONResponse({'task_id': task.id})


@app.get('/tasks/{task_id}')
def get_status(task_id):
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.status == 'SUCCESS':
        result = {'task_id': task_id, 'task_status': task_result.status, 'task_result': task_result.result}
    elif task_result.status == 'FAILURE':
        result = {'task_id': task_id, 'task_status': task_result.status, 'task_result': str(task_result.result)}
    else:
        result = {'task_id': task_id, 'task_status': task_result.status, 'task_result': 'Task is still in progress.'}

    return JSONResponse(result)


@app.get('/apps', response_model=schemas.AvailableApps)
def get_apps(sdk_service: SnsdkWrapperService = Depends(get_sdk_service)):
    available_apps = sdk_service.get_apps()
    if available_apps is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='no available apps found')
    return available_apps


# @app.get('/dataset', response_model=schemas.Datasets)
# def get_datasets():
#     datasets = sdk_service.get_datasets()
#     if datasets is None:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='no datasets found')
#     return datasets


# @app.get('/dataset/{dataset_name}', response_model=schemas.Dataset)
# def get_dataset(dataset_name: str):
#     dataset = sdk_service.get_dataset(dataset_name)
#     if dataset is None:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset with name '{dataset_name}' not found"
#         )
#     return dataset


# @app.delete('/dataset/{dataset_name}')
# def delete_dataset(dataset_name: str):
#     try:
#         response = sdk_service.delete_dataset(dataset_name)
#         if not response:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Dataset with name '{dataset_name}' not found",
#             )
#         return Response(status_code=status.HTTP_204_NO_CONTENT)
#     except HTTPException as e:
#         raise e
#     except Exception:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"internal error occurred: could not delete dataset with name: '{dataset_name}'",
#         )
