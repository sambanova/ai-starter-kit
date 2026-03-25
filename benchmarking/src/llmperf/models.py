from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class RequestConfig(BaseModel):
    """The configuration for a request to the LLM API.

    Args:
        request_idx: The index of the request
        model: The model to use for the request
        prompt_tuple: A tuple containing the prompt to provide to the LLM API along with the tokenized prompt length.
        messages: Optional list of chat messages in OpenAI format (role/content dicts). When set, takes precedence
            over prompt_tuple for the API call body. prompt_tuple is still used for token-count tracking.
        tools: Optional list of tool definitions in OpenAI function-calling format. Only used when messages is set.
        image: Optional image to include in the request
        sampling_params: Optional additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions
        llm_api: Optional name of the LLM API to send the request to.
        use_debugging_mode: Optional flag to enable debugging mode
        api_variables: Dictionary of additional variables for the API
        is_stream_mode: Optional flag to enable streaming mode
        num_concurrent_requests: Optional number of concurrent requests
        metadata: Optional additional metadata to attach to the request for logging or validation purposes.

    """

    request_idx: int
    model: str
    prompt_tuple: Tuple[Dict[str, Any], int]
    messages: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    image: Optional[str] = None
    sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None
    use_debugging_mode: Optional[bool] = None
    api_variables: Dict[str, str] = {}
    is_stream_mode: Optional[bool] = None
    num_concurrent_requests: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMResponse(BaseModel):
    """The response object created from a response from one of the LLM APIs

    Args:
        metrics: Dictionary containing the throughput metrics from the endpoint
        response_text: The generated text from the LLM
        request_config: The associated request config
    """

    metrics: Dict[str, Any]
    response_text: str
    request_config: RequestConfig
