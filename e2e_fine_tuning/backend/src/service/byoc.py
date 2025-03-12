import logging
from typing import Any, Dict, List

from e2e_fine_tuning.backend.src.repositories.byoc import BYOCRepository

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class BYOCService:
    def __init__(self, repository: BYOCRepository) -> None:
        self.repository = repository

    def create_checkpoint_config(self, checkpoint_info: str) -> None:
        return self.repository.create_checkpoint_config(checkpoint_info)

    def get_suitable_apps(self) -> List[List[Dict[str, Any]]]:
        suitable_apps = self.repository.get_suitable_apps()
        return suitable_apps

    def upload_checkpoint(self, messages: List[Dict[str, str]]) -> str:
        logger.info('start set_chat_template')
        self.repository.set_chat_template()
        logger.info('check_chat_template')
        self.repository.check_chat_template(messages)
        logger.info('set_padding_token')
        self.repository.set_padding_token()
        logger.info('get_model_params')
        self.repository.get_model_params()
        logger.info('update_app')
        self.repository.update_app()
        logger.info('upload_checkpoint')
        model_id = self.repository.upload_checkpoint()
        logger.info(f'model uploaded {model_id}')
        return model_id
