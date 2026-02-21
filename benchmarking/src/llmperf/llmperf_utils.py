import json
import time
from collections.abc import Iterable
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

NUM_RNG_ATTEMPTS = 10  # Unlikely to be used in practice: prevents eternal WHILE-loops
LVLM_IMAGE_PATHS = {
    'small': './imgs/vision_perf_eval-small.jpg',
    'medium': './imgs/vision_perf_eval-medium.jpg',
    'large': './imgs/vision_perf_eval-large.jpg',
}


class LLMPerfResults:
    """Class with LLM Performance results"""

    def __init__(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata['timestamp'] = self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Updates and flattens dictionary

        Returns:
            dict: transformed dictionary
        """
        data = {
            'name': self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self) -> str:
        """Transforms dictionary to json string

        Returns:
            str: json string
        """
        data = self.to_dict()
        return json.dumps(data)


# Functions imported from benchmarking_utils - kept here for backwards compatibility
# find_family_model_type and get_tokenizer are now imported from benchmarking_utils


def flatten(item: Union[Iterable[Union[str, Iterable[str]]], str]) -> Generator[str, None, None]:
    """Flattens an iterable"""
    for sub_item in item:
        if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
            yield from flatten(sub_item)
        else:
            yield sub_item


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flattens dictionary

    Args:
        d (dict): input dictionary
        parent_key (str, optional): parent key. Defaults to "".
        sep (str, optional): separator. Defaults to "_".

    Returns:
        dict: output flat dictionary
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
