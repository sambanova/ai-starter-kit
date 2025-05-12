import base64
from typing import Any, Dict, List, Optional
from openai import OpenAI

import requests
import json


def read_json_file(file_path: str) -> Any:
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary. If the file does not exist or the JSON is invalid,
          returns None.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        json.JSONDecodeError: If the JSON data is invalid.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None


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


def function_calling(messages: List[Dict[str, Any]],
                   client: OpenAI,
                   model: str,
                   tools: List[Dict[str, Any]] = None,
                   tool_choise: str = "auto",
                   parallel_tool_calls: bool = False,
                   response_format: Dict[str, Any] = None,
                   stream: bool = False,
                   ) -> Any:
    """
    
    """
    if tools is not None:
      tools_args={
          "tools": tools,
          "tool_choice": tool_choise,
          "parallel_tool_calls": parallel_tool_calls
          }
    else:
      tools_args={}

    results = []

    try:
        completion = client.chat.completions.create(
        model=model,
        messages = messages,
        stream = stream,
        response_format = response_format,
        **tools_args
        )
        if stream:
            for chunk in completion:
                results.append(chunk.choices)
        else:
            if completion and hasattr(completion, "error"):
                results = f"Error: {completion.error}"
            else:
                results = completion.choices[0].message
    except Exception as e:
        raise e

    return results
