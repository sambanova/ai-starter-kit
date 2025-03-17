from typing import Any, Dict, List, Optional

from pydantic import BaseModel

test_messages = [
    {'role': 'system', 'content': 'This is a system prompt.'},
    {'role': 'user', 'content': 'This is a user prompt.'},
    {'role': 'assistant', 'content': 'This is a response from the assistant.'},
    {'role': 'user', 'content': 'This is an user follow up'},
]


class DatasetCreate(BaseModel):
    dataset_name_sambastudio: str = 'publichealth-testing'
    dataset_description: str = 'Q&A pages and FAQs'
    dataset_split: str = 'train'
    data_files: List[str] = ['english.csv', 'spanish.csv', 'french.csv', 'russian.csv', 'chinese.csv']
    dataset_job_types: List[str] = ['evaluation', 'train']
    dataset_source_type: str = 'localMachine'
    dataset_language: str = 'english'
    dataset_filetype: str = 'hdf5'
    dataset_url: str = 'https://huggingface.co/datasets/xhluca/publichealth-qa'
    dataset_metadata: Dict[str, Any] = {}
    hf_dataset: str = 'xhluca/publichealth-qa'
    model_family: str = 'llama3'
    tokenizer: str = 'lightblue/suzume-llama-3-8B-multilingual'
    max_seq_length: int = 8192
    shuffle: str = 'on_RAM'
    input_packing_config: str = 'single::truncate_right'
    prompt_keyword: str = 'prompt'
    completion_keyword: str = 'completion'
    num_training_splits: int = 8
    apply_chat_template: bool = False


class CheckPointCreate(BaseModel):
    hf_model_name: str = 'lightblue/suzume-llama-3-8B-multilingual'
    model_name_sambastudio: str = 'Suzume-Llama-3-8B-Multilingual'
    publisher: str = 'lightblue'
    description: str = 'Suzume 8B, a multilingual finetune of Llama 3'
    param_count: int = 8
    messages: Optional[List[Dict[str, str]]] = test_messages


class BaseResponse(BaseModel):
    id: str
    name: str


class DatasetResponse(BaseModel):
    id: str
    dataset_name: str


class Dataset(BaseModel):
    dataset: str


class Datasets(BaseModel):
    datasets: List[DatasetResponse]


class Status(BaseModel):
    status: str


class ProjectResponse(BaseModel):
    id: str
    name: str
    status: Status
    owner: str


class Projects(BaseModel):
    projects: List[ProjectResponse]


class Project(BaseModel):
    project_id: str


class ProjectCreate(BaseModel):
    project_name: str
    project_description: str


class AvailableApps(BaseModel):
    available_apps: List[BaseResponse]
