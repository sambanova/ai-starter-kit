from fastapi import APIRouter, Depends, HTTPException, Response, status
from huggingface_hub import HfApi

import e2e_fine_tuning.backend.src.schemas as schemas
from e2e_fine_tuning.backend.src.dependencies import get_sdk_service
from e2e_fine_tuning.backend.src.service.hf import HuggingFaceHandler
from e2e_fine_tuning.backend.src.service.sdk import SnsdkWrapperService
from e2e_fine_tuning.backend.src.utils_functions import create_folder

# from e2e_fine_tuning.backend.src.celery_app_workers.dataset_worker import create_dataset_task


router = APIRouter(prefix='/dataset', tags=['Dataset'])

data_dir, target_dir = create_folder()
data_dir, target_dir_model = create_folder(target_dir='models')

hf_client = HfApi()
hf_handler = HuggingFaceHandler(hf_client, data_dir, target_dir_model)


@router.get('/', response_model=schemas.Datasets)
def get_datasets(sdk_service: SnsdkWrapperService = Depends(get_sdk_service)):
    datasets = sdk_service.get_datasets()
    if datasets is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='no datasets found')
    return datasets


@router.get('/{dataset_name}', response_model=schemas.Dataset)
def get_dataset(dataset_name: str, sdk_service: SnsdkWrapperService = Depends(get_sdk_service)):
    dataset = sdk_service.get_dataset(dataset_name)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset with name '{dataset_name}' not found"
        )
    return dataset


@router.delete('/{dataset_name}')
def delete_dataset(dataset_name: str, sdk_service: SnsdkWrapperService = Depends(get_sdk_service)):
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
