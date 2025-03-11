import logging
import os
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import DatasetInfo, HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from requests.exceptions import HTTPError

from utils.fine_tuning.src import sambastudio_utils

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# hf_model = 'lightblue/suzume-llama-3-8B-multilingual'


class HuggingFaceHandler:
    def __init__(self, client: HfApi, data_dir: str, target_dir: str) -> None:
        self.__client = client
        self.data_dir = data_dir
        self.target_dir = target_dir

    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        try:
            dataset_info = self.__client.dataset_info(dataset_name)
            return dataset_info
        except HTTPError as e:
            logger.info(f'Error trying to get dataset info: {e}')
            return None

    def list_repo_files_from_model(self, hf_model_name: str) -> Optional[List[str]]:
        try:
            response = self.__client.list_repo_files(hf_model_name)
            return response
        except RepositoryNotFoundError as e:
            return None

    def download_hf_dataset(self, dataset_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            dataset = load_dataset(
                dataset_info['hf_dataset'],
                data_dir=self.data_dir,
                data_files=dataset_info['data_files'],
                split=dataset_info['dataset_split'],
            )
            logger.info(f"Dataset loaded: {dataset_info['hf_dataset']}")
        except DatasetNotFoundError:
            return None
        except ValueError as e:
            return None

        dataset = dataset.rename_columns({'question': 'prompt', 'answer': 'completion'}).select_columns(
            ['prompt', 'completion']
        )
        dataset.to_json(os.path.join(self.target_dir, f'{dataset_info["hf_dataset"].split("/")[-1]}.jsonl'))
        logger.info(f'Dataset saved as jsonl at {self.target_dir}')

        # gen repo pipeline
        hdf5_dataset_path = sambastudio_utils.gen_data_prep_pipeline(
            input_files=os.path.join(self.target_dir, f'{dataset_info["hf_dataset"].split("/")[-1]}.jsonl'),
            output_path=os.path.join(self.target_dir, f'fine_tuning-{dataset_info["hf_dataset"].split("/")[-1]}'),
            tokenizer=dataset_info['tokenizer'],  # use the tokenizer of the model to train with
            max_seq_length=dataset_info['max_seq_length'],
            shuffle=dataset_info['shuffle'],
            input_packing_config=dataset_info['input_packing_config'],
            prompt_keyword=dataset_info['prompt_keyword'],
            completion_keyword=dataset_info['completion_keyword'],
            num_training_splits=dataset_info['num_training_splits'],
            apply_chat_template=dataset_info['apply_chat_template'],
        )
        logger.info(f'Data preparation pipeline completed. Dataset path: {hdf5_dataset_path}')

        dataset = {
            'dataset_path': hdf5_dataset_path,
            'dataset_name': dataset_info['dataset_name_sambastudio'],
            'dataset_description': dataset_info['dataset_description'],
            'dataset_job_types': dataset_info['dataset_job_types'],
            'dataset_source_type': dataset_info['dataset_source_type'],
            'dataset_language': dataset_info['dataset_language'],
            'dataset_filetype': dataset_info['dataset_filetype'],
            'dataset_url': dataset_info['dataset_url'],
            'dataset_metadata': dataset_info['dataset_metadata'],
        }
        return dataset

    def download_hf_model(self, hf_model_name: str) -> Optional[bool]:
        repo_files = self.list_repo_files_from_model(hf_model_name)
        if repo_files is None:
            return False
        for file_name in repo_files:
            hf_hub_download(repo_id=hf_model_name, filename=file_name, cache_dir=self.target_dir)
        return True
