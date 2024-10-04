"""Wrapper around Sambanova multimodal APIs."""

import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Union

import requests
import sseclient


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
        
    def url_to_b64(self, url:str) -> str:
        """
        Converts an image from a URL to a base64 encoded string.

        :param str url: The URL of the image.
        :return: The base64 encoded string representation of the image.
        :rtype: str
        """
        response = requests.get(url)
        try:
            if response.status_code == 200:
                image_binary = response.content
                base64_image = base64.b64encode(image_binary).decode()
                return base64_image
            else:
                raise ValueError(f"Unable to retrieve image from URL status code {response.status_code}")
        except Exception as e:
            raise ValueError(f"Can't encode image to b64 from provided url: {e}")


    def _is_base64_encoded(self, image: str) -> bool:
        """
        Checks if a string is base64 encoded.

        :param: str image: The string to check.
        :return: True if the string is base64 encoded, False otherwise.
        :rtype: bool
        """
        image = image.strip()

        if len(image) % 4 != 0:
            return False

        try:
            # Decode the base64 string
            base64_bytes = base64.b64decode(image, validate=True)

            # Check if it starts with common image file headers
            if base64_bytes.startswith(b'\xff\xd8\xff'):  # JPEG
                return True
            elif base64_bytes.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                return True
            elif base64_bytes.startswith(b'GIF87a') or base64_bytes.startswith(b'GIF89a'):  # GIF
                return True
            elif base64_bytes.startswith(b'BM'):  # BMP
                return True
            else:
                return False
        except Exception as e:
            # print(f"Exception: {e}")
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
        regex = re.compile(r'^(https?://.*\.(?:png|jpg|jpeg|gif|bmp|webp|svg))$', re.IGNORECASE)

        return re.match(regex, image) is not None

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
            raise ValueError(
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
            raise RuntimeError(
                "Error: The API response does not contain the 'choices' key or the 'message' 'content' values.",
                f'raw response: {response}',
            )
        return generation

    def _process_openai_api_response_stream(self, response: requests.Response) -> Generator[Dict, None, None]:
        """
        Processes the generic API response and yields the resulting string.

        :param response: The API response iterator
        :yield: The response text
        :rtype: str
        """
        client = sseclient.SSEClient(response)
        for event in client.events():
            chunk = {
                'event': event.event,
                'data': event.data,
                'status_code': response.status_code,
            }
            if chunk['event'] == 'error_event' or chunk['status_code'] != 200:
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code " f"{chunk['status_code']}." f"{chunk}."
                )
            try:
                # check if the response is a final event
                # in that case event data response is '[DONE]'
                if chunk['data'] != '[DONE]':
                    if isinstance(chunk['data'], str):
                        data = json.loads(chunk['data'])
                    else:
                        raise RuntimeError(
                            f"Sambanova /complete call failed with status code " f"{chunk['status_code']}." f"{chunk}."
                        )
                    if data.get('error'):
                        raise RuntimeError(
                            f"Sambanova /complete call failed with status code " f"{chunk['status_code']}." f"{chunk}."
                        )
                    if len(data['choices']) > 0:
                        yield data['choices'][0]['delta'].get('content', '')
            except Exception:
                raise Exception(f'Error getting content chunk raw streamed response: {chunk}')

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
                f'Details: {response.text}',
            )
        else:
            return response.json()

    def _call_openai_api(self, prompt: str, images: List) -> Dict:
        """
        Calls the Sambastudio multimodal openai compatible endpoint to generate a response.
        :param str prompt: Prompt for the model to generate a response
        :param list images: Images to be used with the model
        :return: The request json response
        :rtype: Dict
        """

        data = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': f'{prompt}'},
                    ],
                }
            ],
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens_to_generate,
            'top_p': self.top_p,
            'stream': False,
        }
        if len(self.stop) > 1:
            data['stop'] = self.stop
        for image in images:
            if not self._is_url(image):
                image = f'data:image/jpeg;base64,{image}'
            else:
                # temporal conversion until URL is supported directly by API
                image = f'data:image/jpeg;base64,{self.url_to_b64(image)}'
                
            data['messages'][0]['content'].append({'type': 'image_url', 'image_url': {'url': image}})

        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        response = requests.post(self.base_url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            raise RuntimeError(
                f'Sambastudio multimodal API call failed with status code {response.status_code}.',
                f'Details: {response.text}',
            )
        else:
            return response.json()

    def _call_openai_api_stream(self, prompt: str, images: List) -> Iterator:
        """
        Calls the Sambastudio multimodal openai compatible endpoint to stream a response.
        :param str prompt: Prompt for the model to stream a response
        :param list images: Images to be used with the model
        :return: The request json response
        :rtype: Dict
        """

        data = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': f'{prompt}'},
                    ],
                }
            ],
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens_to_generate,
            'top_p': self.top_p,
            'stream': True,
        }
        if len(self.stop) > 1:
            data['stop'] = self.stop
        for image in images:
            if not self._is_url(image):
                image = f'data:image/jpeg;base64,{image}'
            else:
                # temporal conversion until URL is supported directly by API
                image = f'data:image/jpeg;base64,{self.url_to_b64(image)}'
            data['messages'][0]['content'].append({'type': 'image_url', 'image_url': {'url': image}})

        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        response = requests.post(self.base_url, headers=headers, data=json.dumps(data), stream=True)
        if response.status_code != 200:
            raise RuntimeError(
                f'Sambastudio multimodal API call failed with status code {response.status_code}.',
                f'Details: {response.text}',
            )
        else:
            return response

    def _load_images(self, images: Union[str, List] = None) -> Optional[List]:
        """
        Loads the images into base64 format or url.

        :param Union[str, List] images: Image or images to be used with the model url, absolute path or base64 image
        :return: List of base64 encoded / URL images
        """
        if images is None:
            images = []
        if isinstance(images, str):
            images = [images]
        images_list = []
        for image in images:
            if self._is_base64_encoded(image):
                image = image
                images_list.append(image)
            elif self._is_file_path(image):
                images_list.append(self.image_to_base64(image))
            elif self._is_url(image):
                images_list.append(image)
            else:
                raise ValueError('images should be provided as an url, a path or as a base64 encoded image')
        return images_list

    def invoke(self, prompt: str = None, images: Union[str, List] = None) -> str:
        """
        Calls the Sambastudio multimodal endpoint to generate a response.

        :param str prompt: Prompt for the model to generate a response
        :param str, list images: Image or images to be used with the model url, absolute path or base64 image
        :return: The generated response
        :rtype: str
        """
        images_list = self._load_images(images)
        # Call the appropriate API based on the host URL
        if 'v1/chat/completions' in self.base_url:
            response = self._call_openai_api(prompt, images_list)
            generation = self._process_openai_api_response(response)
        elif 'generic' in self.base_url:
            if len(images_list) > 1:
                raise ValueError('only one image can be provided for generic endpoint')
            if self._is_url(images_list[0]):
                # temporal conversion until URL is supported directly by API
                image = f'data:image/jpeg;base64,{self.url_to_b64(images_list[0])}'
                #raise ValueError('image should be provided as a path or as a base64 encoded image for generic endpoint')
            else:
                image = images_list[0]
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

    def stream(self, prompt: str = None, images: Union[str, List] = None) -> Iterator:
        """
        Calls the Sambastudio multimodal endpoint to generate a response.

        :param str prompt: Prompt for the model to generate a response
        :param str, list images: Image or images to be used with the model url, absolute path or base64 image
        :return: The generated response
        :rtype: str
        """
        images_list = self._load_images(images)
        # Call the appropriate API based on the host URL
        if 'v1/chat/completions' in self.base_url:
            response = self._call_openai_api_stream(prompt, images_list)
            for chunk in self._process_openai_api_response_stream(response):
                yield chunk
        elif 'generic' in self.base_url:
            if len(images_list) > 1:
                raise ValueError('only one image can be provided for generic endpoint')
            if self._is_url(images_list[0]):
                raise ValueError('image should be provided as a path or as a base64 encoded image for generic endpoint')

            formatted_prompt = f"""A chat between a curious human and an artificial intelligence assistant.
            The assistant gives helpful, detailed, and polite answers to the humans question.\
            USER: <image>
            {prompt}
            ASSISTANT:"""
            raise NotImplementedError('Streaming method not implemented for multimodal generic endpoints')
        else:
            raise ValueError(
                f'Unsupported host URL: {self.base_url}', 'only Generic and open AI compatible APIs supported'
            )
