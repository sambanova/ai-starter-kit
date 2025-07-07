from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class SambaStudioOpenAIResponseMetadata(BaseModel):
    finish_reason: Optional[str]
    usage: Optional[Dict[str, Any]]
    model_name: str
    system_fingerprint: str
    created: int


class SambaStudioOpenAIResponse(BaseModel):
    content: str
    id: str
    response_metadata: SambaStudioOpenAIResponseMetadata


class SambaStudioGenericV2Item(BaseModel):
    id: str
    value: Dict[str, Any]


class SambaStudioGenericV2Response(BaseModel):
    items: List[SambaStudioGenericV2Item]


class SambaStudioGenericV1Response(BaseModel):
    predictions: List[Dict[str, Any]]


class Delta(BaseModel):
    content: Optional[str]
    role: Optional[str] = None


class SNCloudStreamingChoice(BaseModel):
    delta: Delta
    finish_reason: Optional[str]
    index: int
    logprobs: Optional[Dict[str, Any]]


class ToolCall(BaseModel):
    function: dict
    id: str
    type: str


class ToolMessage(BaseModel):
    content: Optional[str]
    role: str
    tool_calls: List[ToolCall]


class BaseMessage(BaseModel):
    content: str
    role: Literal['assistant']


class SNCloudBaseChoice(BaseModel):
    finish_reason: str
    index: int
    logprobs: str | None
    message: BaseMessage | ToolMessage


class SNCloudBaseUsage(BaseModel):
    acceptance_rate: float
    completion_tokens: int
    completion_tokens_after_first_per_sec: float
    completion_tokens_after_first_per_sec_first_ten: float
    completion_tokens_per_sec: float
    end_time: datetime
    is_last_response: bool
    prompt_tokens: int
    start_time: datetime
    time_to_first_token: float
    total_latency: float
    total_tokens: int
    total_tokens_per_sec: float


class SNCloudChatCompletionChunk(BaseModel):
    choices: List[SNCloudBaseChoice | SNCloudStreamingChoice]
    created: int
    id: str
    model: str
    object: str
    system_fingerprint: str
    usage: Optional[SNCloudBaseUsage] = None


class SNCloudBaseResponse(BaseModel):
    choices: List[SNCloudBaseChoice | SNCloudStreamingChoice]
    created: int
    id: str
    model: str
    object: str
    system_fingerprint: str
    usage: Optional[SNCloudBaseUsage] = None


class EmbeddingsBaseModel(BaseModel):
    type: str
    batch_size: int
    bundle: bool
    model: str


class LLMBaseModel(BaseModel):
    type: str
    model: str
    max_tokens: int
    temperature: float
    streaming: bool
    bundle: Optional[bool] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None
    select_expert: Optional[str] = None
