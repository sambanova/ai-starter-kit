import os
import sys

import e2e_fine_tuning.backend.src.schemas as schemas
from e2e_fine_tuning.backend.src.config import settings
from e2e_fine_tuning.backend.src.routes import checkpoint, dataset, project
from e2e_fine_tuning.backend.src.service.sdk import SnsdkWrapperService
from e2e_fine_tuning.backend.src.utils_functions import create_folder

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(repo_dir)

import logging

from celery.app import Celery
from celery.result import AsyncResult
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from e2e_fine_tuning.backend.src.dependencies import get_byoc_service, get_hf_handler, get_sdk_service

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
hf_handler = get_hf_handler()

redis_url = settings.redis_url
celery_app = Celery(__name__, broker=redis_url, backend=redis_url)

app.include_router(checkpoint.router)
app.include_router(dataset.router)
app.include_router(project.router)


@app.get('/tasks/{task_id}')
def get_status(task_id: str) -> JSONResponse:
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.status == 'SUCCESS':
        result = {'task_id': task_id, 'task_status': task_result.status, 'task_result': task_result.result}
    elif task_result.status == 'FAILURE':
        result = {'task_id': task_id, 'task_status': task_result.status, 'task_result': str(task_result.result)}
    else:
        result = {'task_id': task_id, 'task_status': task_result.status, 'task_result': 'Task is still in progress.'}

    return JSONResponse(result)


@app.get('/apps', response_model=schemas.AvailableApps)
def get_apps(sdk_service: SnsdkWrapperService = Depends(get_sdk_service)) -> schemas.AvailableApps:
    available_apps = sdk_service.get_apps()
    if available_apps is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No available apps found')
    return available_apps
