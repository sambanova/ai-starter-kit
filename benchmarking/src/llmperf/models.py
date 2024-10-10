from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel


class RequestConfig(BaseModel):
    """The configuration for a request to the LLM API.

    Args:
        model: The model to use.
        prompt_tuple: A tuple containing the prompt to provide to the LLM API along with the tokenized prompt length.
        sampling_params: Additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions
        llm_api: The name of the LLM API to send the request to.
        mode: API mode (stream or batch)
        num_concurrent_requests: number of concurrent requests
        metadata: Additional metadata to attach to the request for logging or validation purposes.
    """

    request_idx: int
    model: str
    prompt_tuple: Tuple[str, int]
    sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None
    api_variables: Dict[str, str] = {}
    is_stream_mode: Optional[bool] = None
    num_concurrent_requests: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMResponse(BaseModel):
    """The response object created from a response from one of the SambaStudio LLM APIs

    Args:
        metrics: Dictionary containing the throughput metrics from the endpoint
        response_text: The generated text from the LLM
        request_config: The associated request config
    """

    metrics: Dict[str, Any]
    response_text: str
    request_config: RequestConfig
