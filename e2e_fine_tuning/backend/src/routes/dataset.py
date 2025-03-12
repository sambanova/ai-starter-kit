from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse

import e2e_fine_tuning.backend.src.schemas as schemas
from e2e_fine_tuning.backend.src.celery_app_workers.celery_workers import create_dataset_task
from e2e_fine_tuning.backend.src.dependencies import get_byoc_service, get_hf_handler, get_sdk_service
from e2e_fine_tuning.backend.src.service.sdk import SnsdkWrapperService

router = APIRouter(prefix='/dataset', tags=['Dataset'])

sdk_service = get_sdk_service()
byoc_service = get_byoc_service()

hf_handler = get_hf_handler()


@router.get('/', response_model=schemas.Datasets)
def get_datasets(sdk_service: SnsdkWrapperService = Depends(get_sdk_service)) -> Optional[schemas.Datasets]:
    datasets = sdk_service.get_datasets()
    if datasets is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='no datasets found')
    return datasets


@router.get('/{dataset_name}', response_model=schemas.Dataset)
def get_dataset(
    dataset_name: str, sdk_service: SnsdkWrapperService = Depends(get_sdk_service)
) -> Optional[schemas.Dataset]:
    dataset = sdk_service.get_dataset(dataset_name)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset with name '{dataset_name}' not found"
        )
    return dataset


@router.post('/')
def create_dataset(dataset_info: schemas.DatasetCreate) -> JSONResponse:
    task = create_dataset_task.delay(dataset_info.model_dump())
    return JSONResponse({'task_id': task.id})


@router.delete('/{dataset_name}')
def delete_dataset(dataset_name: str, sdk_service: SnsdkWrapperService = Depends(get_sdk_service)) -> Response:
    try:
        response = sdk_service.delete_dataset(dataset_name)
        if not response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with name '{dataset_name}' not found",
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"internal error occurred: could not delete dataset with name: '{dataset_name}'",
        )
