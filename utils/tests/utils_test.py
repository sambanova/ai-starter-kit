import base64
from typing import Any, Dict, Optional

import requests


def image_to_base64(image_path: str) -> str:
    """
    Convert an image to base64.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded image.
    """
    with open(image_path, 'rb') as img_file:
        base64_img = base64.b64encode(img_file.read()).decode('utf-8')
        return base64_img


def encode_to_base64(content: bytes) -> str:
    """Encode audio file to base64"""
    return base64.b64encode(content).decode('utf-8')


def load_encode_audio(path: str) -> str:
    with open(path, 'rb') as file:
        audio = file.read()
    b64_audio = encode_to_base64(content=audio)
    return b64_audio


def audio_requests(
    url: str,
    api_key: str,
    file_path: str,
    language: str = 'english',
    model: str = 'Qwen2-Audio-7B-Instruct',
    prompt: Optional[str] = None,
    temperature: float = 0.01,
) -> Dict[str, Any] | str:
    headers = {
        'Authorization': f'Bearer {api_key}',
    }
    with open(file_path, 'rb') as file:
        files: Dict[str, Any] = {
            'file': file,
        }
        data = {
            'model': model,
            'language': language,
            'response_format': 'json',
            'temperature': str(temperature),
        }
        if prompt:
            files['prompt'] = prompt

        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        return response.text
