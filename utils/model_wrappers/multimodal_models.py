"""Wrapper around Sambanova multimodal APIs."""
import os
import re

import base64
import binascii
import json
from pathlib import Path
from typing import Dict

import requests


class SambastudioMultimodal:
    """
    Sambanova Multimodal models wrapper.
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = 0.01,
        max_tokens_to_generate: int = 1024,
        top_p: float = 0.01,
        top_k: int = 1,
        stop: list = None,
        do_sample: bool = False,
    ) -> None:
        """
        Initialize the SambastudioMultimodal.

        :param str base_url:  Base URL of the deployed Sambastudio multimodal endpoint,
        :param str api_key: pi_key the deployed Sambastudio multimodal endpoint ,
        :param float temperature: model temperature,
        :param str model: model name,
        :param int max_tokens_to_generate: maximum number of tokens to generate,
        :param float top_p: model top k,
        :param int top_k: model top k,
        :param list stop: list of token to stop generation when stop token is found
        :param bool do_sample: whether to do sample for model generation
        """
        self.base_url = base_url
        if self.base_url is None:
            self.base_url = os.getenv('LVLM_BASE_URL')
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = os.getenv('LVLM_API_KEY')
        self.temperature = temperature
        self.model = model
        self.max_tokens_to_generate = max_tokens_to_generate
        self.top_p = top_p
        self.top_k = top_k
        self.stop = stop
        if stop is None:
            self.stop = []
        self.do_sample = do_sample
        self.http_session = requests.Session()

    def image_to_base64(self, image_path: str) -> str:
        """
        Converts an image file to a base64 encoded string.

        :param: str image_path: The path to the image file.
        :return: The base64 encoded string representation of the image.
        rtype: str
        """
        with open(image_path, 'rb') as image_file:
            image_binary = image_file.read()
            base64_image = base64.b64encode(image_binary).decode()
            return base64_image

    def _is_base64_encoded(self, image: str) -> bool:
        """
        Checks if a string is base64 encoded.

        :param: str image: The string to check.
        :return: True if the string is base64 encoded, False otherwise.
        :rtype: bool
        """
        try:
            base64.b64decode(self, image)
            return True
        except (binascii.Error, TypeError):
            return False

    def _is_file_path(self, image: str) -> bool:
        """
        Returns True if the path exists on the filesystem
        :param: str image: The string to check.
        :return: True if the string is base64 encoded, False otherwise.
        :rtype: bool
        """
        path = Path(image)
        return path.exists()
    
    def _is_url(self, image: str) -> bool:
        """
        Returns True if the string is an url
        
        :param: str image: The string to check.
        :return: True if the string is an url, False otherwise.
        :rtype: bool
        """
        regex = re.compile(
            r'^(https?://.*\.(?:png|jpg|jpeg|gif|bmp|webp|svg))$', re.IGNORECASE
        )

        return re.match(regex, url) is not None

    def _process_generic_api_response(self, response: Dict) -> str:
        """
        Processes the generic API response and returns the resulting string.

        :param dict response: The API response
        :return: The response text
        :rtype: str
        """
        try:
            generation = response['predictions'][0]['completion']
        except Exception as e:
            raise (
                "Error: The API response does not contain the 'predictions' key or the 'completion' value.",
                f'raw response: {response}',
            )
        return generation

    def _process_openai_api_response(self, response: Dict) -> str:
        """
        Processes the generic API response and returns the resulting string.

        :param dict response: The API response
        :return: The response text
        :rtype: str
        """
        try:
            generation = response['choices'][0]['message']['content']
        except Exception as e:
            raise (
                "Error: The API response does not contain the 'choices' key or the 'message' 'content' values.",
                f'raw response: {response}',
            )
        return generation

    def _call_generic_api(self, prompt: str, image_b64: str) -> Dict:
        """
        Calls the Sambastudio multimodal generic endpoint to generate a response.
        :param str prompt: Prompt for the model to generate a response
        :param str image: Image to be used with the model
        :return: The request json response
        :rtype: Dict
        """
        data = {
            'instances': [{'prompt': prompt, 'image_content': f'{image_b64}'}],
            'params': {
                'do_sample': {'type': 'bool', 'value': str(self.do_sample)},
                'max_tokens_to_generate': {'type': 'int', 'value': str(self.max_tokens_to_generate)},
                'temperature': {'type': 'float', 'value': str(self.temperature)},
                'top_k': {'type': 'int', 'value': str(self.top_k)},
                'top_logprobs': {'type': 'int', 'value': '0'},
                'top_p': {'type': 'float', 'value': str(self.top_p)},
            },
        }
        headers = {'Content-Type': 'application/json', 'key': self.api_key}
        response = requests.post(self.base_url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            raise RuntimeError(
                f'Sambastudio multimodal API call failed with status code {response.status_code}',
                f'Details: {response.text}'
            )
        else:
            return response.json()

    def _call_openai_api(self, prompt: str, image: str) -> Dict:
        """
        Calls the Sambastudio multimodal openai compatible endpoint to generate a response.
        :param str prompt: Prompt for the model to generate a response
        :param str image: Image to be used with the model
        :return: The request json response
        :rtype: Dict
        """
        
        data = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': f'{prompt}'},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image}'}},
                    ],
                }
            ],
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens_to_generate,
            'top_p': self.top_p,
            'stream': False,
            'stop': self.stop,
        }
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        response = requests.post(self.base_url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            raise RuntimeError(
                f'Sambastudio multimodal API call failed with status code {response.status_code}.',
                f'Details: {response.text}'
            )
        else:
            return response.json()

    def invoke(self, prompt: str, image: str) -> str:
        """
        Calls the Sambastudio multimodal endpoint to generate a response.

        :param str prompt: Prompt for the model to generate a response
        :param str image: Image to be used with the model absolute path or base64 image
        :return: The generated response
        :rtype: str
        """
        if self._is_base64_encoded(image):
            image = image
            is_url = False
        elif self._is_file_path(image):
            image = self.image_to_base64(image)
            is_url = False
        elif self._is_url(image):
            is_url = True
        else:
            raise ('image should be provided as an url, a path or as a base64 encoded image')

        # Call the appropriate API based on the host URL
        if 'openai' in self.base_url:
            response = self._call_openai_api(prompt, image)
            generation = self._process_openai_api_response(response)
        elif 'generic' in self.base_url:
            if is_url:
                raise("image should be provided as a path or as a base64 encoded image for generic endpoint")
            formatted_prompt = f"""A chat between a curious human and an artificial intelligence assistant.
            The assistant gives helpful, detailed, and polite answers to the humans question.\
            USER: <image>
            {prompt}
            ASSISTANT:"""
            response = self._call_generic_api(formatted_prompt, image)
            generation = self._process_generic_api_response(response)
        else:
            raise ValueError(
                f'Unsupported host URL: {self.base_url}', 'only Generic and open AI compatible APIs supported'
            )
        return generation
