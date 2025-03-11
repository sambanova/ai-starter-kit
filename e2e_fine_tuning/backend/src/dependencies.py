from functools import lru_cache

from e2e_fine_tuning.backend.src.repositories.byoc import BYOCRepository
from e2e_fine_tuning.backend.src.service.byoc import BYOCService
from e2e_fine_tuning.backend.src.repositories.sdk import SnsdkWrapperRepository
from e2e_fine_tuning.backend.src.service.sdk import SnsdkWrapperService
from utils.byoc.src.snsdk_byoc_wrapper import BYOC
from utils.fine_tuning.src.snsdk_wrapper import SnsdkWrapper
from e2e_fine_tuning.backend.src.utils_functions import create_folder


data_dir, target_dir_model = create_folder(target_dir='models')
sambastudio_client = SnsdkWrapper()
sdk_repository = SnsdkWrapperRepository(sambastudio_client)
sdk_service = SnsdkWrapperService(sdk_repository)

byoc = BYOC()
byoc_repo = BYOCRepository(byoc, target_dir_model)
byoc_service = BYOCService(byoc_repo)


@lru_cache()
def get_sdk_service() -> SnsdkWrapperService:
    return sdk_service

@lru_cache()
def get_byoc_service() -> BYOCService:
    return byoc_service
