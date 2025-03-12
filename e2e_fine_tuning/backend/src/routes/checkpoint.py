from fastapi import APIRouter
from fastapi.responses import JSONResponse

import e2e_fine_tuning.backend.src.schemas as schemas
from e2e_fine_tuning.backend.src.celery_app_workers.celery_workers import create_checkpoint_task
from e2e_fine_tuning.backend.src.dependencies import get_byoc_service, get_hf_handler, get_sdk_service

router = APIRouter(prefix='/checkpoint', tags=['Checkpoint'])

sdk_service = get_sdk_service()
byoc_service = get_byoc_service()
hf_handler = get_hf_handler()


@router.post('/')
def create_checkpoint(checkpoint_info: schemas.CheckPointCreate) -> JSONResponse:
    task = create_checkpoint_task.delay(checkpoint_info.model_dump())
    return JSONResponse({'task_id': task.id})
