"""Wrapper around Sambanova APIs."""

import json
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

import requests
from langchain.callbacks.base import BaseCallbackHandler  # type: ignore
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env


class SVEndpointHandler:
    """
    SambaNova Systems Interface for Sambaverse endpoint.

    :param str host_url: Base URL of the DaaS API service
    """

    API_BASE_PATH = '/api/predict'

    def __init__(self, host_url: str):
        """
        Initialize the SVEndpointHandler.

        :param str host_url: Base URL of the DaaS API service
        """
        self.host_url = host_url
        self.http_session = requests.Session()

    @staticmethod
    def _process_response(response: requests.Response) -> Dict:
        """
        Processes the API response and returns the resulting dict.

        All resulting dicts, regardless of success or failure, will contain the
        `status_code` key with the API response status code.

        If the API returned an error, the resulting dict will contain the key
        `detail` with the error message.

        If the API call was successful, the resulting dict will contain the key
        `data` with the response data.

        :param requests.Response response: the response object to process
        :return: the response dict
        :type: dict
        """
        result: Dict[str, Any] = {}
        try:
            lines_result = response.text.strip().split('\n')
            text_result = lines_result[-1]
            if response.status_code == 200 and json.loads(text_result).get('error'):
                completion = ''
                for line in lines_result[:-1]:
                    completion += json.loads(line)['result']['responses'][0]['stream_token']
                text_result = lines_result[-2]
                result = json.loads(text_result)
                result['result']['responses'][0]['completion'] = completion
            else:
                result = json.loads(text_result)
        except Exception as e:
            result['detail'] = str(e)
        if 'status_code' not in result:
            result['status_code'] = response.status_code
        return result

    @staticmethod
    def _process_streaming_response(
        response: requests.Response,
    ) -> Generator[Dict, None, None]:
        """Process the streaming response"""
        try:
            for line in response.iter_lines():
                chunk = json.loads(line)
                if 'status_code' not in chunk:
                    chunk['status_code'] = response.status_code
                if chunk['status_code'] == 200 and chunk.get('error'):
                    chunk['result'] = {'responses': [{'stream_token': ''}]}
                    return chunk
                yield chunk
        except Exception as e:
            raise RuntimeError(f'Error processing streaming response: {e}')

    def _get_full_url(self) -> str:
        """
        Return the full API URL for a given path.
        :returns: the full API URL for the sub-path
        :type: str
        """
        return f'{self.host_url}{self.API_BASE_PATH}'

    def nlp_predict(
        self,
        key: str,
        sambaverse_model_name: Optional[str],
        input: Union[List[str], str],
        params: Optional[str] = '',
        stream: bool = False,
    ) -> Dict:
        """
        NLP predict using inline input string.

        :param str project: Project ID in which the endpoint exists
        :param str endpoint: Endpoint ID
        :param str key: API Key
        :param str input_str: Input string
        :param str params: Input params string
        :returns: Prediction results
        :type: dict
        """
        if params:
            data = {'instance': input, 'params': json.loads(params)}
        else:
            data = {'instance': input}
        response = self.http_session.post(
            self._get_full_url(),
            headers={
                'key': key,
                'Content-Type': 'application/json',
                'modelName': sambaverse_model_name,
            },
            json=data,
        )
        return SVEndpointHandler._process_response(response)

    def nlp_predict_stream(
        self,
        key: str,
        sambaverse_model_name: Optional[str],
        input: Union[List[str], str],
        params: Optional[str] = '',
    ) -> Iterator[Dict]:
        """
        NLP predict using inline input string.

        :param str project: Project ID in which the endpoint exists
        :param str endpoint: Endpoint ID
        :param str key: API Key
        :param str input_str: Input string
        :param str params: Input params string
        :returns: Prediction results
        :type: dict
        """
        if params:
            data = {'instance': input, 'params': json.loads(params)}
        else:
            data = {'instance': input}
        # Streaming output
        response = self.http_session.post(
            self._get_full_url(),
            headers={
                'key': key,
                'Content-Type': 'application/json',
                'modelName': sambaverse_model_name,
            },
            json=data,
            stream=True,
        )
        for chunk in SVEndpointHandler._process_streaming_response(response):
            yield chunk


class SsStreamingHandler(BaseCallbackHandler):
    """Wrapper around Base Callback Handler."""

    def __init__(self, queue):
        self.queue = queue

    """Callback handler for Sambanova streaming."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.queue.put(token)


class SambaverseEndpoint(LLM):
    """
    Sambaverse large language models.

    To use, you should have the environment variable ``SAMBAVERSE_API_KEY``
    set with your API key.

    get one in https://sambaverse.sambanova.ai
    read extra documentation in https://docs.sambanova.ai/sambaverse/latest/index.html


    Example:
    .. code-block:: python

        from langchain_community.llms.sambanova  import Sambaverse
        Sambaverse(
            sambaverse_url="https://sambaverse.sambanova.ai",
            sambaverse_api_key="your-sambaverse-api-key",
            sambaverse_model_name="Meta/llama-2-7b-chat-hf",
            streaming: = False
            model_kwargs={
                "select_expert": "llama-2-7b-chat-hf",
                "do_sample": False,
                "max_tokens_to_generate": 100,
                "temperature": 0.7,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "top_k": 50,
            },
        )
    """

    sambaverse_url: str = ''
    """Sambaverse url to use"""

    sambaverse_api_key: str = ''
    """sambaverse api key"""

    sambaverse_model_name: Optional[str] = None
    """sambaverse expert model to use"""

    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    streaming: Optional[bool] = False
    """Streaming flag to get streamed response."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values['sambaverse_url'] = get_from_dict_or_env(
            values,
            'sambaverse_url',
            'SAMBAVERSE_URL',
            default='https://sambaverse.sambanova.ai',
        )
        values['sambaverse_api_key'] = get_from_dict_or_env(values, 'sambaverse_api_key', 'SAMBAVERSE_API_KEY')
        values['sambaverse_model_name'] = get_from_dict_or_env(values, 'sambaverse_model_name', 'SAMBAVERSE_MODEL_NAME')
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{'model_kwargs': self.model_kwargs}}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'Sambaverse LLM'

    def _get_tuning_params(self, stop: Optional[List[str]]) -> str:
        """
        Get the tuning parameters to use when calling the LLM.

        Args:
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.

        Returns:
            The tuning parameters as a JSON string.
        """
        _model_kwargs = self.model_kwargs or {}
        _kwarg_stop_sequences = _model_kwargs.get('stop_sequences', [])
        _stop_sequences = stop or _kwarg_stop_sequences
        if not _kwarg_stop_sequences:
            _model_kwargs['stop_sequences'] = ','.join(f'"{x}"' for x in _stop_sequences)
        tuning_params_dict = {k: {'type': type(v).__name__, 'value': str(v)} for k, v in (_model_kwargs.items())}
        _model_kwargs['stop_sequences'] = _kwarg_stop_sequences
        tuning_params = json.dumps(tuning_params_dict)
        return tuning_params

    def _handle_nlp_predict(
        self,
        sdk: SVEndpointHandler,
        prompt: Union[List[str], str],
        tuning_params: str,
    ) -> str:
        """
        Perform an NLP prediction using the Sambaverse endpoint handler.

        Args:
            sdk: The SVEndpointHandler to use for the prediction.
            prompt: The prompt to use for the prediction.
            tuning_params: The tuning parameters to use for the prediction.

        Returns:
            The prediction result.

        Raises:
            ValueError: If the prediction fails.
        """
        response = sdk.nlp_predict(self.sambaverse_api_key, self.sambaverse_model_name, prompt, tuning_params)
        if response['status_code'] != 200:
            error = response.get('error')
            if error:
                optional_code = error.get('code')
                optional_details = error.get('details')
                optional_message = error.get('message')
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response['status_code']}.\n"
                    f"Message: {optional_message}\n"
                    f"Details: {optional_details}\n"
                    f"Code: {optional_code}\n"
                )
            else:
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code " f"{response['status_code']}." f"{response}."
                )
        return response['result']['responses'][0]['completion']

    def _handle_completion_requests(self, prompt: Union[List[str], str], stop: Optional[List[str]]) -> str:
        """
        Perform a prediction using the Sambaverse endpoint handler.

        Args:
            prompt: The prompt to use for the prediction.
            stop: stop sequences.

        Returns:
            The prediction result.

        Raises:
            ValueError: If the prediction fails.
        """
        ss_endpoint = SVEndpointHandler(self.sambaverse_url)
        tuning_params = self._get_tuning_params(stop)
        return self._handle_nlp_predict(ss_endpoint, prompt, tuning_params)

    def _handle_nlp_predict_stream(
        self, sdk: SVEndpointHandler, prompt: Union[List[str], str], tuning_params: str
    ) -> Iterator[GenerationChunk]:
        """
        Perform a streaming request to the LLM.

        Args:
            sdk: The SVEndpointHandler to use for the prediction.
            prompt: The prompt to use for the prediction.
            tuning_params: The tuning parameters to use for the prediction.

        Returns:
            An iterator of GenerationChunks.
        """
        for chunk in sdk.nlp_predict_stream(self.sambaverse_api_key, self.sambaverse_model_name, prompt, tuning_params):
            if chunk['status_code'] != 200:
                error = chunk.get('error')
                if error:
                    optional_code = error.get('code')
                    optional_details = error.get('details')
                    optional_message = error.get('message')
                    raise ValueError(
                        f"Sambanova /complete call failed with status code "
                        f"{chunk['status_code']}.\n"
                        f"Message: {optional_message}\n"
                        f"Details: {optional_details}\n"
                        f"Code: {optional_code}\n"
                    )
                else:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code " f"{chunk['status_code']}." f"{chunk}."
                    )
            text = chunk['result']['responses'][0]['stream_token']
            generated_chunk = GenerationChunk(text=text)
            yield generated_chunk

    def _stream(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the Sambaverse's LLM on the given prompt.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments. directly passed
                to the sambaverse model in API call.

        Returns:
            An iterator of GenerationChunks.
        """
        ss_endpoint = SVEndpointHandler(self.sambaverse_url)
        tuning_params = self._get_tuning_params(stop)
        try:
            if self.streaming:
                for chunk in self._handle_nlp_predict_stream(ss_endpoint, prompt, tuning_params):
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text)
                    yield chunk
            else:
                return
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f'Error raised by the inference endpoint: {e}') from e

    def _handle_stream_request(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> str:
        """
        Perform a streaming request to the LLM.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments. directly passed
                to the sambaverse model in API call.

        Returns:
            The model output as a string.
        """
        completion = ''
        for chunk in self._stream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs):
            completion += chunk.text
        return completion

    def _call(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments. directly passed
                to the sambaverse model in API call.

        Returns:
            The model output as a string.
        """
        try:
            if self.streaming:
                return self._handle_stream_request(prompt, stop, run_manager, kwargs)
            return self._handle_completion_requests(prompt, stop)
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f'Error raised by the inference endpoint: {e}') from e


class SSEndpointHandler:
    """
    SambaNova Systems Interface for SambaStudio model endpoints.

    :param str host_url: Base URL of the DaaS API service
    """

    def __init__(self, host_url: str, api_base_uri: str):
        """
        Initialize the SSEndpointHandler.

        :param str host_url: Base URL of the DaaS API service
        :param str api_base_uri: Base URI of the DaaS API service
        """
        self.host_url = host_url
        self.api_base_uri = api_base_uri
        self.http_session = requests.Session()

    def _process_response(self, response: requests.Response) -> Dict:
        """
        Processes the API response and returns the resulting dict.

        All resulting dicts, regardless of success or failure, will contain the
        `status_code` key with the API response status code.

        If the API returned an error, the resulting dict will contain the key
        `detail` with the error message.

        If the API call was successful, the resulting dict will contain the key
        `data` with the response data.

        :param requests.Response response: the response object to process
        :return: the response dict
        :type: dict
        """
        result: Dict[str, Any] = {}
        try:
            result = response.json()
        except Exception as e:
            result['detail'] = str(e)
        if 'status_code' not in result:
            result['status_code'] = response.status_code
        return result

    def _process_streaming_response(
        self,
        response: requests.Response,
    ) -> Generator[Dict, None, None]:
        """Process the streaming response"""
        if 'nlp' in self.api_base_uri:
            try:
                import sseclient
            except ImportError:
                raise ImportError(
                    'could not import sseclient library' 'Please install it with `pip install sseclient-py`.'
                )
            client = sseclient.SSEClient(response)
            close_conn = False
            for event in client.events():
                if event.event == 'error_event':
                    close_conn = True
                chunk = {
                    'event': event.event,
                    'data': event.data,
                    'status_code': response.status_code,
                }
                yield chunk
            if close_conn:
                client.close()
        elif 'generic' in self.api_base_uri:
            try:
                for line in response.iter_lines():
                    chunk = json.loads(line)
                    if 'status_code' not in chunk:
                        chunk['status_code'] = response.status_code
                    if chunk['status_code'] == 200 and chunk.get('error'):
                        chunk['result'] = {'responses': [{'stream_token': ''}]}
                    yield chunk
            except Exception as e:
                raise RuntimeError(f'Error processing streaming response: {e}')
        else:
            raise ValueError(f'handling of endpoint uri: {self.api_base_uri} not implemented')

    def _get_full_url(self, path: str) -> str:
        """
        Return the full API URL for a given path.

        :param str path: the sub-path
        :returns: the full API URL for the sub-path
        :type: str
        """
        return f'{self.host_url}/{self.api_base_uri}/{path}'

    def nlp_predict(
        self,
        project: str,
        endpoint: str,
        key: str,
        input: Union[List[str], str],
        params: Optional[str] = '',
        stream: bool = False,
    ) -> Dict:
        """
        NLP predict using inline input string.

        :param str project: Project ID in which the endpoint exists
        :param str endpoint: Endpoint ID
        :param str key: API Key
        :param str input_str: Input string
        :param str params: Input params string
        :returns: Prediction results
        :type: dict
        """
        if isinstance(input, str):
            input = [input]
        if 'nlp' in self.api_base_uri:
            if params:
                data = {'inputs': input, 'params': json.loads(params)}
            else:
                data = {'inputs': input}
        elif 'generic' in self.api_base_uri:
            if params:
                data = {'instances': input, 'params': json.loads(params)}
            else:
                data = {'instances': input}
        else:
            raise ValueError(f'handling of endpoint uri: {self.api_base_uri} not implemented')
        response = self.http_session.post(
            self._get_full_url(f'{project}/{endpoint}'),
            headers={'key': key},
            json=data,
        )
        return self._process_response(response)

    def nlp_predict_stream(
        self,
        project: str,
        endpoint: str,
        key: str,
        input: Union[List[str], str],
        params: Optional[str] = '',
    ) -> Iterator[Dict]:
        """
        NLP predict using inline input string.

        :param str project: Project ID in which the endpoint exists
        :param str endpoint: Endpoint ID
        :param str key: API Key
        :param str input_str: Input string
        :param str params: Input params string
        :returns: Prediction results
        :type: dict
        """
        if 'nlp' in self.api_base_uri:
            if isinstance(input, str):
                input = [input]
            if params:
                data = {'inputs': input, 'params': json.loads(params)}
            else:
                data = {'inputs': input}
        elif 'generic' in self.api_base_uri:
            if isinstance(input, list):
                input = input[0]
            if params:
                data = {'instance': input, 'params': json.loads(params)}
            else:
                data = {'instance': input}
        else:
            raise ValueError(f'handling of endpoint uri: {self.api_base_uri} not implemented')
        # Streaming output
        response = self.http_session.post(
            self._get_full_url(f'stream/{project}/{endpoint}'),
            headers={'key': key},
            json=data,
            stream=True,
        )
        for chunk in self._process_streaming_response(response):
            yield chunk


class SvStreamingHandler(BaseCallbackHandler):
    """Wrapper around Base Callback Handler."""

    def __init__(self, queue):
        self.queue = queue

    """Callback handler for Sambanova streaming."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.queue.put(token)


class SambaNovaEndpoint(LLM):
    """
    SambaStudio large language models.

    To use, you should have the environment variables
    ``BASE_URL`` set with your SambaStudio environment URL.
    ``BASE_URI`` set with your SambaStudio api base URI.
    ``PROJECT_ID`` set with your SambaStudio project ID.
    ``ENDPOINT_ID`` set with your SambaStudio endpoint ID.
    ``API_KEY``  set with your SambaStudio endpoint API key.

    https://sambanova.ai/products/enterprise-ai-platform-sambanova-suite

    read extra documentation in https://docs.sambanova.ai/sambastudio/latest/index.html

    Example:
    .. code-block:: python

        from langchain_community.llms.sambanova  import Sambaverse
        SambaStudio(
            base_url="your SambaStudio environment URL",
            base_uri="your-SambaStudio-base-URI",
            project_id=set with your SambaStudio project ID.,
            endpoint_id=set with your SambaStudio endpoint ID.,
            api_token= set with your SambaStudio endpoint API key.,
            streaming=false
            model_kwargs={
                "do_sample": False,
                "max_tokens_to_generate": 1000,
                "temperature": 0.7,
                "top_p": 1.0,
                "repetition_penalty": 1,
                "top_k": 50,
            },
        )
    """

    base_url: str = ''
    """Base url to use"""

    base_uri: str = ''
    """endpoint base uri"""

    project_id: str = ''
    """Project id on sambastudio for model"""

    endpoint_id: str = ''
    """endpoint id on sambastudio for model"""

    api_key: str = ''
    """sambastudio api key"""

    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    streaming: Optional[bool] = False
    """Streaming flag to get streamed response."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{'model_kwargs': self.model_kwargs}}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'Sambastudio LLM'

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['base_url'] = get_from_dict_or_env(values, 'base_url', 'BASE_URL')
        values['base_uri'] = get_from_dict_or_env(values, 'base_uri', 'BASE_URI', default='api/predict/nlp')
        values['project_id'] = get_from_dict_or_env(values, 'project_id', 'PROJECT_ID')
        values['endpoint_id'] = get_from_dict_or_env(values, 'endpoint_id', 'ENDPOINT_ID')
        values['api_key'] = get_from_dict_or_env(values, 'api_key', 'API_KEY')
        return values

    def _get_tuning_params(self, stop: Optional[List[str]]) -> str:
        """
        Get the tuning parameters to use when calling the LLM.

        Args:
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.

        Returns:
            The tuning parameters as a JSON string.
        """
        _model_kwargs = self.model_kwargs or {}
        _kwarg_stop_sequences = _model_kwargs.get('stop_sequences', [])
        _stop_sequences = stop or _kwarg_stop_sequences
        # if not _kwarg_stop_sequences:
        # _model_kwargs["stop_sequences"] = ",".join(
        #    f'"{x}"' for x in _stop_sequences
        # )
        tuning_params_dict = {k: {'type': type(v).__name__, 'value': str(v)} for k, v in (_model_kwargs.items())}
        # _model_kwargs["stop_sequences"] = _kwarg_stop_sequences
        tuning_params = json.dumps(tuning_params_dict)
        return tuning_params

    def _handle_nlp_predict(self, sdk: SSEndpointHandler, prompt: Union[List[str], str], tuning_params: str) -> str:
        """
        Perform an NLP prediction using the SambaStudio endpoint handler.

        Args:
            sdk: The SSEndpointHandler to use for the prediction.
            prompt: The prompt to use for the prediction.
            tuning_params: The tuning parameters to use for the prediction.

        Returns:
            The prediction result.

        Raises:
            ValueError: If the prediction fails.
        """
        response = sdk.nlp_predict(
            self.project_id,
            self.endpoint_id,
            self.api_key,
            prompt,
            tuning_params,
        )
        if response['status_code'] != 200:
            optional_detail = response.get('detail')
            if optional_detail:
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response['status_code']}.\n Details: {optional_detail}"
                )
            else:
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response['status_code']}.\n response {response}"
                )
        if 'nlp' in self.base_uri:
            return response['data'][0]['completion']
        elif 'generic' in self.base_uri:
            return response['predictions'][0]['completion']
        else:
            raise ValueError(f'handling of endpoint uri: {self.base_uri} not implemented')

    def _handle_completion_requests(self, prompt: Union[List[str], str], stop: Optional[List[str]]) -> str:
        """
        Perform a prediction using the SambaStudio endpoint handler.

        Args:
            prompt: The prompt to use for the prediction.
            stop: stop sequences.

        Returns:
            The prediction result.

        Raises:
            ValueError: If the prediction fails.
        """
        ss_endpoint = SSEndpointHandler(self.base_url, self.base_uri)
        tuning_params = self._get_tuning_params(stop)
        return self._handle_nlp_predict(ss_endpoint, prompt, tuning_params)

    def _handle_nlp_predict_stream(
        self, sdk: SSEndpointHandler, prompt: Union[List[str], str], tuning_params: str
    ) -> Iterator[GenerationChunk]:
        """
        Perform a streaming request to the LLM.

        Args:
            sdk: The SVEndpointHandler to use for the prediction.
            prompt: The prompt to use for the prediction.
            tuning_params: The tuning parameters to use for the prediction.

        Returns:
            An iterator of GenerationChunks.
        """
        for chunk in sdk.nlp_predict_stream(self.project_id, self.endpoint_id, self.api_key, prompt, tuning_params):
            if chunk['status_code'] != 200:
                error = chunk.get('error')
                if error:
                    optional_code = error.get('code')
                    optional_details = error.get('details')
                    optional_message = error.get('message')
                    raise ValueError(
                        f"Sambanova /complete call failed with status code "
                        f"{chunk['status_code']}.\n"
                        f"Message: {optional_message}\n"
                        f"Details: {optional_details}\n"
                        f"Code: {optional_code}\n"
                    )
                else:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code " f"{chunk['status_code']}." f"{chunk}."
                    )
            if 'nlp' in self.base_uri:
                text = json.loads(chunk['data'])['stream_token']
            elif 'generic' in self.base_uri:
                text = chunk['result']['responses'][0]['stream_token']
            else:
                raise ValueError(f'handling of endpoint uri: {self.base_uri} not implemented')
            generated_chunk = GenerationChunk(text=text)
            yield generated_chunk

    def _stream(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call out to Sambanova's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        ss_endpoint = SSEndpointHandler(self.base_url, self.base_uri)
        tuning_params = self._get_tuning_params(stop)
        try:
            if self.streaming:
                for chunk in self._handle_nlp_predict_stream(ss_endpoint, prompt, tuning_params):
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text)
                    yield chunk
            else:
                return
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f'Error raised by the inference endpoint: {e}') from e

    def _handle_stream_request(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> str:
        """
        Perform a streaming request to the LLM.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments. directly passed
                to the sambaverse model in API call.

        Returns:
            The model output as a string.
        """
        completion = ''
        for chunk in self._stream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs):
            completion += chunk.text
        return completion

    def _call(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Sambanova's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        if stop is not None:
            raise Exception('stop not implemented')
        try:
            if self.streaming:
                return self._handle_stream_request(prompt, stop, run_manager, kwargs)
            return self._handle_completion_requests(prompt, stop)
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f'Error raised by the inference endpoint: {e}') from e


class SambaStudioFastCoE(LLM):
    """
    SambaStudio large language models.

    To use, you should have the environment variables
    ``FAST_COE_URL`` set with your SambaStudio environment URL.
    ``FAST_COE_API_KEY``  set with your SambaStudio endpoint API key.

    https://sambanova.ai/products/enterprise-ai-platform-sambanova-suite

    read extra documentation in https://docs.sambanova.ai/sambastudio/latest/index.html

    Example:
    .. code-block:: python

        from langchain_community.llms.sambanova  import Sambaverse
        SambaStudio(
            fast_coe_url="your fast CoE endpoint URL",
            api_token= set with your fast CoE endpoint API key.,
            max_tokens = mas number of tokens to generate
            stop_tokens = list of stop tokens
            model = model name
        )
    """

    fast_coe_url: str = ''
    """Url to use"""

    fast_coe_api_key: str = ''
    """fastCoE api key"""

    max_tokens: int = 1024
    """max tokens to generate"""

    stop_tokens: list = ['<|eot_id|>']
    """Stop tokens"""

    model: str = 'llama3-8b'
    """LLM model expert to use"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {'model': self.model, 'max_tokens': self.max_tokens, 'stop': self.stop_tokens}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'Sambastudio Fast CoE'

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['fast_coe_url'] = get_from_dict_or_env(values, 'fast_coe_url', 'FAST_COE_URL')
        values['fast_coe_api_key'] = get_from_dict_or_env(values, 'fast_coe_api_key', 'FAST_COE_API_KEY')
        return values

    def _handle_nlp_predict_stream(
        self,
        prompt: Union[List[str], str],
        stop: List[str],
    ) -> Iterator[GenerationChunk]:
        """
        Perform a streaming request to the LLM.

        Args:
            prompt: The prompt to use for the prediction.
            stop: list of stop tokens

        Returns:
            An iterator of GenerationChunks.
        """
        try:
            import sseclient
        except ImportError:
            raise ImportError('could not import sseclient library' 'Please install it with `pip install sseclient-py`.')
        try:
            formatted_prompt = json.loads(prompt)
        except:
            formatted_prompt = [{'role': 'user', 'content': prompt}]

        
        http_session = requests.Session()
        if not stop:
            stop = self.stop_tokens
        data = {'inputs': formatted_prompt, 'max_tokens': self.max_tokens, 'stop': stop, 'model': self.model}
        # Streaming output
        response = http_session.post(
            self.fast_coe_url,
            headers={'Authorization': f'Basic {self.fast_coe_api_key}', 'Content-Type': 'application/json'},
            json=data,
            stream=True,
        )

        client = sseclient.SSEClient(response)
        close_conn = False
        for event in client.events():
            if event.event == 'error_event':
                close_conn = True
            chunk = {
                'event': event.event,
                'data': event.data,
                'status_code': response.status_code,
            }

            if chunk['status_code'] == 200 and chunk.get('error'):
                chunk['result'] = {'responses': [{'stream_token': ''}]}
            if chunk['status_code'] != 200:
                error = chunk.get('is_error')
                optional_details = chunk.get('completion')
                if error:
                    optional_details = error.get('details')
                    raise ValueError(
                        f"Sambanova /complete call failed with status code "
                        f"{chunk['status_code']}.\n"
                        f"Details: {optional_details}\n"
                    )
                else:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code " f"{chunk['status_code']}." f"{chunk}."
                    )
            text = json.loads(chunk['data'])['stream_token']
            generated_chunk = GenerationChunk(text=text)
            yield generated_chunk

    def _stream(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call out to Sambanova's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        try:
            for chunk in self._handle_nlp_predict_stream(prompt, stop):
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
                yield chunk
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f'Error raised by the inference endpoint: {e}') from e

    def _handle_stream_request(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> str:
        """
        Perform a streaming request to the LLM.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments. directly passed
                to the sambaverse model in API call.

        Returns:
            The model output as a string.
        """
        completion = ''
        for chunk in self._stream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs):
            completion += chunk.text
        return completion

    def _call(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Sambanova's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        try:
            return self._handle_stream_request(prompt, stop, run_manager, kwargs)
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f'Error raised by the inference endpoint: {e}') from e

class SambaNovaEmbeddingModel(BaseModel, Embeddings):
    """SambaNova embedding models.

    To use, you should have the environment variables
    ``EMBED_BASE_URL``, ``EMBED_BASE_URI``
    ``EMBED_PROJECT_ID``, ``EMBED_ENDPOINT_ID``,
    ``EMBED_API_KEY``
    set with your personal sambastudio variable or pass it as a named parameter
    to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import SambaStudioEmbeddings

            embeddings = SambaStudioEmbeddings(embed_base_url=base_url,
                                          embed_base_uri=base_uri,
                                          embed_project_id=project_id,
                                          embed_endpoint_id=endpoint_id,
                                          embed_api_key=api_key,
                                          batch_size=32)
            (or)

            embeddings = SambaStudioEmbeddings(batch_size=32)

            (or)

            # CoE example
            embeddings = SambaStudioEmbeddings(
                batch_size=1,
                model_kwargs={
                    'select_expert':'e5-mistral-7b-instruct'
                }
            )
    """

    embed_base_url: str = ''
    """Base url to use"""

    embed_base_uri: str = ''
    """endpoint base uri"""

    embed_project_id: str = ''
    """Project id on sambastudio for model"""

    embed_endpoint_id: str = ''
    """endpoint id on sambastudio for model"""

    embed_api_key: str = ''
    """sambastudio api key"""

    model_kwargs: dict = {}
    """Key word arguments to pass to the model."""

    batch_size: int = 32
    """Batch size for the embedding models"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['embed_base_url'] = get_from_dict_or_env(values, 'embed_base_url', 'EMBED_BASE_URL')
        values['embed_base_uri'] = get_from_dict_or_env(
            values,
            'embed_base_uri',
            'EMBED_BASE_URI',
            default='api/predict/generic',
        )
        values['embed_project_id'] = get_from_dict_or_env(values, 'embed_project_id', 'EMBED_PROJECT_ID')
        values['embed_endpoint_id'] = get_from_dict_or_env(values, 'embed_endpoint_id', 'EMBED_ENDPOINT_ID')
        values['embed_api_key'] = get_from_dict_or_env(values, 'embed_api_key', 'EMBED_API_KEY')
        return values

    def _get_tuning_params(self) -> str:
        """
        Get the tuning parameters to use when calling the model

        Returns:
            The tuning parameters as a JSON string.
        """
        tuning_params_dict = {k: {'type': type(v).__name__, 'value': str(v)} for k, v in (self.model_kwargs.items())}
        tuning_params = json.dumps(tuning_params_dict)
        return tuning_params

    def _get_full_url(self, path: str) -> str:
        """
        Return the full API URL for a given path.

        :param str path: the sub-path
        :returns: the full API URL for the sub-path
        :type: str
        """
        return f'{self.embed_base_url}/{self.embed_base_uri}/{path}'  # noqa: E501

    def _iterate_over_batches(self, texts: List[str], batch_size: int) -> Generator:
        """Generator for creating batches in the embed documents method
        Args:
            texts (List[str]): list of strings to embed
            batch_size (int, optional): batch size to be used for the embedding model.
            Will depend on the RDU endpoint used.
        Yields:
            List[str]: list (batch) of strings of size batch size
        """
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Returns a list of embeddings for the given sentences.
        Args:
            texts (`List[str]`): List of texts to encode
            batch_size (`int`): Batch size for the encoding
        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings
            for the given sentences
        """
        if batch_size is None:
            batch_size = self.batch_size
        http_session = requests.Session()
        url = self._get_full_url(f'{self.embed_project_id}/{self.embed_endpoint_id}')
        params = json.loads(self._get_tuning_params())
        embeddings = []

        if 'nlp' in self.embed_base_uri:
            for batch in self._iterate_over_batches(texts, batch_size):
                data = {'inputs': batch, 'params': params}
                response = http_session.post(
                    url,
                    headers={'key': self.embed_api_key},
                    json=data,
                )
                try:
                    embedding = response.json()['data']
                    embeddings.extend(embedding)
                except KeyError:
                    raise KeyError(
                        "'data' not found in endpoint response",
                        response.json(),
                    )

        elif 'generic' in self.embed_base_uri:
            for batch in self._iterate_over_batches(texts, batch_size):
                data = {'instances': batch, 'params': params}
                response = http_session.post(
                    url,
                    headers={'key': self.embed_api_key},
                    json=data,
                )
                try:
                    if params.get('select_expert'):
                        embedding = response.json()['predictions'][0]
                    else:
                        embedding = response.json()['predictions']
                    embeddings.extend(embedding)
                except KeyError:
                    raise KeyError(
                        "'predictions' not found in endpoint response",
                        response.json(),
                    )

        else:
            raise ValueError(f'handling of endpoint uri: {self.embed_base_uri} not implemented')

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings
            for the given sentences
        """
        http_session = requests.Session()
        url = self._get_full_url(f'{self.embed_project_id}/{self.embed_endpoint_id}')
        params = json.loads(self._get_tuning_params())

        if 'nlp' in self.embed_base_uri:
            data = {'inputs': [text], 'params': params}
            response = http_session.post(
                url,
                headers={'key': self.embed_api_key},
                json=data,
            )
            try:
                embedding = response.json()['data'][0]
            except KeyError:
                raise KeyError(
                    "'data' not found in endpoint response",
                    response.json(),
                )

        elif 'generic' in self.embed_base_uri:
            data = {'instances': [text], 'params': params}
            response = http_session.post(
                url,
                headers={'key': self.embed_api_key},
                json=data,
            )
            try:
                if params.get('select_expert'):
                    embedding = response.json()['predictions'][0][0]
                else:
                    embedding = response.json()['predictions'][0]
            except KeyError:
                raise KeyError(
                    "'predictions' not found in endpoint response",
                    response.json(),
                )

        else:
            raise ValueError(
                f'handling of endpoint uri: {self.embed_base_uri} not implemented'  # noqa: E501
            )

        return embedding
