import os
from typing import Tuple


def create_folder(data_dir: str = 'data', target_dir: str = 'datasets') -> Tuple[str, str]:
    target_dir = os.path.join('./e2e_fine_tuning', data_dir, target_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    return data_dir, target_dir
