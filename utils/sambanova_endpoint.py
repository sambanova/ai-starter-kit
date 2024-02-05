"""Wrapper around Sambanova APIs."""
import json
from typing import Any, Dict, List, Optional, Union

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from pydantic import Extra, root_validator


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
        if type(input) == str:
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

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

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
        sdk = SSEndpointHandler(self.base_url)
        _model_kwargs = self.model_kwargs or {}
        # _stop_sequences = _model_kwargs.get("stop_sequences", [])
        # _stop_sequences = stop or _stop_sequences
        # _model_kwargs["stop_sequences"] = ",".join(f"'{x}'" for x in _stop_sequences)
        tuning_params = {
            k: {"type": type(v).__name__, "value": str(v)}
            for k, v in _model_kwargs.items()
        }
        tuning_params = json.dumps(tuning_params)
        try:
            response = sdk.nlp_predict(
                self.project_id, self.endpoint_id, self.api_key, prompt, tuning_params
            )
            if response["status_code"] != 200:
                optional_detail = response["detail"]
                raise ValueError(
                    f"Sambanova /complete call failed with status code {response['status_code']}."
                    f" Details: {optional_detail}"
                )
            text = response["data"][0]["completion"]
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f"Error raised by the inference endpoint: {e}") from e
        return text
