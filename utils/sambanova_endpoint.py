"""Wrapper around Sambanova APIs."""

from typing import Any, Dict, List, Optional, Union, Iterator
import json
import requests  # type: ignore
import sseclient  # type: ignore

from pydantic import Extra, root_validator  # type: ignore
from langchain.schema.output import GenerationChunk  # type: ignore
from langchain.callbacks.manager import CallbackManagerForLLMRun  # type: ignore
from langchain.llms.base import LLM  # type: ignore
from langchain.utils import get_from_dict_or_env  # type: ignore
from langchain.callbacks.base import BaseCallbackHandler  # type: ignore
from langchain_core.embeddings import Embeddings


class SSEndpointHandler:
    """
    SambaNova Systems Interface for SS model endpoints.

    :param str host_url: Base URL of the DaaS API service
    :param str access_key: Secret key with which to authenticate
    """

    API_BASE_PATH = "/api"

    def __init__(self, host_url: str):
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
        :rtype: dict
        """
        result = {}
        try:
            result = response.json()
        except Exception as e:
            result["detail"] = str(e)
        if "status_code" not in result:
            result["status_code"] = response.status_code
        return result

    @staticmethod
    def _process_streaming_response(response: requests.Response) -> Dict:
        """Process the streaming response"""
        client = sseclient.SSEClient(response)
        close_conn = False
        for event in client.events():
            if event.event == "error_event":
                close_conn = True
            text = json.dumps({"event": event.event, "data": event.data})
            chunk = GenerationChunk(text=text)
            yield chunk
        if close_conn:
            client.close()

    def _get_full_url(self, path: str) -> str:
        """
        Return the full API URL for a given path.

        :param str path: the sub-path
        :returns: the full API URL for the sub-path
        :rtype: str
        """
        return f"{self.host_url}{self.API_BASE_PATH}{path}"

    def nlp_predict(
        self,
        project: str,
        endpoint: str,
        key: str,
        input: Union[List[str], str],
        params: str = None,
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
        :rtype: dict
        """
        if type(input) is str:
            input = [input]
        if params:
            data = {"inputs": input, "params": json.loads(params)}
        else:
            data = {"inputs": input}
        response = self.http_session.post(
            self._get_full_url(f"/predict/nlp/{project}/{endpoint}"),
            headers={"key": key},
            json=data,
        )
        return SSEndpointHandler._process_response(response)

    def nlp_predict_stream(
        self,
        project: str,
        endpoint: str,
        key: str,
        input: Union[List[str], str],
        params: str = None,
    ) -> Iterator[GenerationChunk]:
        """
        NLP predict using inline input string.

        :param str project: Project ID in which the endpoint exists
        :param str endpoint: Endpoint ID
        :param str key: API Key
        :param str input_str: Input string
        :param str params: Input params string
        :returns: Prediction results
        :rtype: dict
        """
        if type(input) is str:
            input = [input]
        if params:
            data = {"inputs": input, "params": json.loads(params)}
        else:
            data = {"inputs": input}
        # Streaming output
        response = self.http_session.post(
            self._get_full_url(f"/predict/nlp/stream/{project}/{endpoint}"),
            headers={"key": key},
            json=data,
            stream=True,
        )
        for chunk in SSEndpointHandler._process_streaming_response(response):
            yield chunk


class SsStreamingHandler(BaseCallbackHandler):
    """Wrapper around Base Callback Handler."""

    def __init__(self, queue):
        self.queue = queue

    """Callback handler for Sambanova streaming."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.queue.put(token)


class SambaNovaEndpoint(LLM):
    """Wrapper around Sambanova large language models.

    To use, you should have the environment variable ``Sambanova_API_KEY``
    set with your API key.

    Example:
    .. code-block:: python

        from langchain.llms.sambanova_endpoint  import SambaNovaEndpoint
        SambaNovaEndpoint(
            base_url="sambastudio host url",
            project_id="project_id",
            endpoint_id="endpoint_id",
            api_token="api_token",
            streaming=true

            model_kwargs={
                "do_sample": False,
                "max_tokens_to_generate": 256,
                "temperature": 0.7,
                "top_p": 1.0,
                "repetition_penalty": 1,
                "stop_sequences":"",
                "top_k": 50,
            },
        )
    """

    base_url: Optional[str] = None
    """Base url to use"""

    project_id: Optional[str] = None
    """Project id on sambastudio for model"""

    endpoint_id: Optional[str] = None
    """endpoint id on sambastudio for model"""

    api_key: Optional[str] = None
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

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["base_url"] = get_from_dict_or_env(values, "base_url", "BASE_URL")
        values["project_id"] = get_from_dict_or_env(values, "project_id", "PROJECT_ID")
        values["endpoint_id"] = get_from_dict_or_env(
            values, "endpoint_id", "ENDPOINT_ID"
        )
        values["api_key"] = get_from_dict_or_env(values, "api_key", "API_KEY")
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_kwargs": self.model_kwargs}}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "Sambastudio LLM"

    def _handle_nlp_predict(self, sdk, prompt, tuning_params) -> str:
        response = sdk.nlp_predict(
            self.project_id, self.endpoint_id, self.api_key, prompt, tuning_params
        )
        if response["status_code"] != 200:
            optional_detail = response["detail"]
            raise ValueError(
                f"Sambanova /complete call failed with status code "
                f"{response['status_code']}. Details: {optional_detail}"
            )
        return response["data"][0]["completion"]

    def _handle_nlp_predict_stream(
        self, sdk, prompt, tuning_params
    ) -> Iterator[GenerationChunk]:
        for chunk in sdk.nlp_predict_stream(
            self.project_id, self.endpoint_id, self.api_key, prompt, tuning_params
        ):
            yield chunk

    def _get_tuning_params(self, stop):
        _model_kwargs = self.model_kwargs or {}
        _stop_sequences = _model_kwargs.get("stop_sequences", [])
        _stop_sequences = stop or _stop_sequences
        # _model_kwargs['stop_sequences'] = ','.join(
        #     f"'{x}'" for x in _stop_sequences)
        tuning_params = {
            k: {"type": type(v).__name__, "value": str(v)}
            for k, v in (_model_kwargs.items())
        }
        tuning_params = json.dumps(tuning_params)
        return tuning_params

    def _handle_stream_request(self, prompt, stop, run_manager, kwargs):
        completion = ""
        for chunk in self._stream(
            prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
        ):
            completion += chunk.text
        return completion

    def _handle_completion_requests(self, prompt, stop):
        ss_endpoint = SSEndpointHandler(self.base_url)
        tuning_params = self._get_tuning_params(stop)
        return self._handle_nlp_predict(ss_endpoint, prompt, tuning_params)

    def _call(
        self,
        prompt: str,
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
            if self.streaming:
                return self._handle_stream_request(prompt, stop, run_manager, kwargs)
            return self._handle_completion_requests(prompt, stop)
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f"Error raised by the inference endpoint: {e}") from e

    def _stream(
        self,
        prompt: str,
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
        ss_endpoint = SSEndpointHandler(self.base_url)
        tuning_params = self._get_tuning_params(stop)
        try:
            if self.streaming:
                for chunk in self._handle_nlp_predict_stream(
                    ss_endpoint, prompt, tuning_params
                ):
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text)
            else:
                return
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f"Error raised by the inference endpoint: {e}") from e
        
        
class SVEndpointHandler:
    """
    SambaNova Systems Interface for Sambaverse endpoint.

    :param str host_url: Base URL of the DaaS API service
    """

    API_BASE_PATH = "/api/predict"

    def __init__(self, host_url: str):
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
        :rtype: dict
        """
        result = {}
        try:
            result = response.text.strip().split('\n')[-1]
            result = {"data": json.loads(result.split("data: ")[-1])}
        except Exception as e:
            result["detail"] = str(e)
        if "status_code" not in result:
            result["status_code"] = response.status_code
        return result

    @staticmethod
    def _process_streaming_response(response: requests.Response) -> Dict:
        """Process the streaming response"""
        client = sseclient.SSEClient(response)
        close_conn = False
        for event in client.events():
            if event.event == "error_event":
                close_conn = True
            text = json.dumps({"event": event.event, "data": event.data})
            chunk = GenerationChunk(text=text)
            yield chunk
        if close_conn:
            client.close()

    def _get_full_url(self) -> str:
        """
        Return the full API URL for a given path.
        :returns: the full API URL for the sub-path
        :rtype: str
        """
        return f"{self.host_url}{self.API_BASE_PATH}"

    def nlp_predict(
        self,
        key: str,
        sambaverse_model_name: str,
        input: Union[List[str], str],
        params: str = None,
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
        :rtype: dict
        """
        if type(input) is str:
            input = [input]
        parsed_input = []
        for element in input:
            parsed_element={
                "conversation_id": "sambaverse-conversation-id",
                "messages": [
                    {
                        "message_id": 0,
                        "role": "user",
                        "content": element,
                    }
                ],
            }
            parsed_input.append(json.dumps(parsed_element))
        if params:
            data = {"inputs": parsed_input, "params": json.loads(params)}
        else:
            data = {"inputs": parsed_input}
        response = self.http_session.post(
            self._get_full_url(),
            headers={
                "key": key, 
                "Content-Type":"application/json",
                "modelName":sambaverse_model_name
                },
            json=data,
        )
        return SVEndpointHandler._process_response(response)

    def nlp_predict_stream(
        self,
        key: str,
        sambaverse_model_name: str,
        input: Union[List[str], str],
        params: str = None,
    ) -> Iterator[GenerationChunk]:
        """
        NLP predict using inline input string.

        :param str project: Project ID in which the endpoint exists
        :param str endpoint: Endpoint ID
        :param str key: API Key
        :param str input_str: Input string
        :param str params: Input params string
        :returns: Prediction results
        :rtype: dict
        """
        if type(input) is str:
            input = [input] 
        parsed_input = []
        for element in input:
            parsed_element={
                "conversation_id": "sambaverse-conversation-id",
                "messages": [
                    {
                        "message_id": 0,
                        "role": "user",
                        "content": element,
                    }
                ],
            }
            parsed_input.append(json.dumps(parsed_element))
        if params:
            data = {"inputs": parsed_input, "params": json.loads(params)}
        else:
            data = {"inputs": parsed_input}
        # Streaming output
        response = self.http_session.post(
            self._get_full_url(),
            headers={
                "key": key, 
                "Content-Type":"application/json",
                "modelName":sambaverse_model_name
                },
            json=data,
            stream=True
        )
        for chunk in SVEndpointHandler._process_streaming_response(response):
            yield chunk


class SvStreamingHandler(BaseCallbackHandler):
    """Wrapper around Base Callback Handler."""

    def __init__(self, queue):
        self.queue = queue

    """Callback handler for Sambanova streaming."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.queue.put(token)


class SambaverseEndpoint(LLM):
    """Wrapper around Sambaverse large language models.

    To use, you should have the environment variable ``SAMBAVERSE_API_KEY``
    set with your API key.

    Example:
    .. code-block:: python

        from langchain.llms.sambaverse_endpoint  import SambaVerseEndpoint
        SambaNovaEndpoint(
            sambaverse_url="sambastudio host url",
            project_id="project_id",
            endpoint_id="endpoint_id",
            api_token="api_token",
            streaming=true

            model_kwargs={
                "do_sample": False,
                "max_tokens_to_generate": 256,
                "temperature": 0.7,
                "top_p": 1.0,
                "repetition_penalty": 1,
                "top_k": 50,
            },
        )
    """

    sambaverse_url: Optional[str] = "https://sambaverse.sambanova.ai"
    """Sambaverse url to use"""
    
    sambaverse_api_key: Optional[str] = None
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
        """Validate that api key and python package exists in environment."""
        values["sambaverse_url"] = get_from_dict_or_env(values, "sambaverse_url", "SAMBAVERSE_URL")
        values["sambaverse_api_key"] = get_from_dict_or_env(values, "sambaverse_api_key", "SAMBAVERSE_API_KEY")
        values["sambaverse_model_name"] = get_from_dict_or_env(values, "sambaverse_model_name", "SAMBAVERSE_MODEL_NAME")
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_kwargs": self.model_kwargs}}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "Sambaverse LLM"

    def _handle_nlp_predict(self, sdk, prompt, tuning_params) -> str:
        response = sdk.nlp_predict(
           self.sambaverse_api_key, self.sambaverse_model_name ,prompt, tuning_params
        )
        if response["status_code"] != 200:
            optional_details = response["details"]
            optional_message = response["message"]
            raise ValueError(
                f"Sambanova /complete call failed with status code "
                f"{response['status_code']}. Details: {optional_details}"
                f"{response['status_code']}. Message: {optional_message}"
            )
        return response["data"]["completion"]

    def _handle_nlp_predict_stream(
        self, sdk, prompt, tuning_params
    ) -> Iterator[GenerationChunk]:
        for chunk in sdk.nlp_predict_stream(
            self.sambaverse_api_key, self.sambaverse_model_name ,prompt, tuning_params
        ):
            yield chunk

    def _get_tuning_params(self, stop):
        _model_kwargs = self.model_kwargs or {}
        _stop_sequences = _model_kwargs.get("stop_sequences", [])
        _stop_sequences = stop or _stop_sequences
        # _model_kwargs['stop_sequences'] = ','.join(
        #     f"'{x}'" for x in _stop_sequences)
        tuning_params = {
            k: {"type": type(v).__name__, "value": str(v)}
            for k, v in (_model_kwargs.items())
        }
        tuning_params = json.dumps(tuning_params)
        return tuning_params

    def _handle_stream_request(self, prompt, stop, run_manager, kwargs):
        completion = ""
        for chunk in self._stream(
            prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
        ):
            completion += chunk.text
        return completion

    def _handle_completion_requests(self, prompt, stop):
        ss_endpoint = SVEndpointHandler(self.sambaverse_url)
        tuning_params = self._get_tuning_params(stop)
        return self._handle_nlp_predict(ss_endpoint, prompt, tuning_params)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Sambaverse's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        try:
            if self.streaming:
                return self._handle_stream_request(prompt, stop, run_manager, kwargs)
            return self._handle_completion_requests(prompt, stop)
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f"Error raised by the inference endpoint: {e}") from e

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call out to Sambaverse's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        ss_endpoint = SVEndpointHandler(self.sambaverse_url)
        tuning_params = self._get_tuning_params(stop)
        try:
            if self.streaming:
                for chunk in self._handle_nlp_predict_stream(
                    ss_endpoint, prompt, tuning_params
                ):
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text)
            else:
                return
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f"Error raised by the inference endpoint: {e}") from e
        
class SambaNovaEmbeddingModel(Embeddings):
    '''def __init__(self, url_domain = None, project_id = None, endpoint_id = None, key = None):
        self.url_domain = get_from_dict_or_env 
        self.project_id = project_id
        self.endpoint_id = endpoint_id
        self.key = key
    ''' 
    def __init__(self, 
                embed_base_url: Optional[str] = None,
                embed_project_id: Optional[str] = None,
                embed_endpoint_id: Optional[str] = None,
                embed_api_key: Optional[str] = None) -> None:
        self.API_BASE_PATH = "/api/predict/nlp/"
        values = {"embed_base_url": embed_base_url,
                  "embed_project_id": embed_project_id,
                  "embed_endpoint_id": embed_endpoint_id,
                  "embed_api_key": embed_api_key}
        self.embed_base_url = get_from_dict_or_env(values, "embed_base_url", "EMBED_BASE_URL")
        self.embed_project_id = get_from_dict_or_env(values, "embed_project_id", "EMBED_PROJECT_ID")
        self.embed_endpoint_id = get_from_dict_or_env(values, "embed_endpoint_id", "EMBED_ENDPOINT_ID") 
        self.embed_api_key = get_from_dict_or_env(values, "embed_api_key", "EMBED_API_KEY")

    def _get_full_url(self, path: str) -> str:
        """
        Return the full API URL for a given path.

        :param str path: the sub-path
        :returns: the full API URL for the sub-path
        :rtype: str
        """
        return f"{self.embed_base_url}{self.API_BASE_PATH}{path}"
    
    def _iterate_over_batches(self, texts: List[str], batch_size: int = 32):
        """Generator for creating batches in the embed documents method
        Args:
            texts (List[str]): list of strings to embedd
            batch_size (int, optional): batch size to be used for the embedding model.  Will depend on the RDU endpoint used. Defaults to 32.
        Yields:
            List[str]: list (batch) of strings of size batch size
        """

        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]
    
    def embed_documents(self, texts, batch_size=32, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        http_session = requests.Session()
        url = self._get_full_url(f"{self.embed_project_id}/{self.embed_endpoint_id}")

        embeddings = []
        
        for batch in self._iterate_over_batches(texts, batch_size):
            data = {"inputs": batch}
            response =http_session.post(
                 url,
                 headers={"key": self.embed_api_key},
                 json=data,
                 )
            print(response.json)
            embedding = response.json()["data"]
            embeddings.extend(embedding)

        return embeddings
    
    def embed_query(self, text, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        http_session = requests.Session()
        url = self._get_full_url(f"{self.embed_project_id}/{self.embed_endpoint_id}")
        
        data = {"inputs": [text]}
        
        response =http_session.post(
            url,
            headers={"key": self.embed_api_key},
            json=data,
            )
        embedding = response.json()["data"][0]
        
        return embedding
    