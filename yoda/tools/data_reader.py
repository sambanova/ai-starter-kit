# data_reader.py
import os
import re
from typing import Any, Dict, List


def read_jsonl_data(file_path: str) -> List[Any]:
    assert file_path.endswith('jsonl')
    data = []
    with open(file_path) as reader:
        for obj in reader:
            data.append(eval(obj))
    return data


def read_txt_data(file_path: str) -> str:
    assert file_path.endswith('txt')
    with open(file_path) as reader:
        lines = reader.readlines()
    return ''.join(lines)


def format_text(text: str) -> Any:
    text = re.sub(r'\n\s+\n', '\n\n', text)
    text = re.sub(r'\n+', '\n', text)
    return text


def collect_articles(folders: List[str]) -> List[Dict[str, Any]]:
    """
    Collects articles from the given folders.

    Args:
        folders (list): List of folder paths.

    Returns:
        list: List of dictionaries containing article information.
            Each dictionary has the following keys:
            - "filename": Name of the file.
            - "filepath": Full path of the file.
            - "article": Formatted text of the article.
    """
    articles = []
    for folder in folders:
        files = os.listdir(folder)
        for filename in files:
            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(folder, filename)
            article = format_text(read_txt_data(filepath))
            articles.append({'filename': filename, 'filepath': filepath, 'article': article})
    return articles
