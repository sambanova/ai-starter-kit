from huggingface_hub import HfApi

from e2e_fine_tuning.backend.src.celery_app_workers.celery_app import celery_app
from e2e_fine_tuning.backend.src.service.hf import HuggingFaceHandler
from e2e_fine_tuning.backend.src.utils_functions import create_folder

# redis_url = settings.redis_url
# celery_app = Celery(__name__, broker=redis_url, backend=redis_url)

# celery_app.conf.update(
#     result_expires=3600,
# )

data_dir, target_dir = create_folder()
data_dir, target_dir_model = create_folder(target_dir='models')


hf_client = HfApi()
hf_handler = HuggingFaceHandler(hf_client, data_dir, target_dir_model)


@celery_app.task(name='create_checkpoint_task', pydantic=True)
def create_checkpoint_task(checkpoint_info: dict):
    response = hf_handler.download_hf_model(checkpoint_info['hf_model_name'])
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with name '{checkpoint_info['hf_checkpoint']}' not found in hugging_face or params invalid",
        )
    pass


@celery_app.task(name='create_dataset_task', pydantic=True)
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
