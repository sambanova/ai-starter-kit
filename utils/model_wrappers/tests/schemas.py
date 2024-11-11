from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


class SambaStudioOpenAIResponseMetadata(BaseModel):
    finish_reason: str
    usage: Optional[Dict[str, Any]]
    model_name: str
    system_fingerprint: str
    created: str


class SambaStudioOpenAIResponse(BaseModel):
    content: str
    id: str
    response_metadata: SambaStudioOpenAIResponseMetadata


class SambaStudioGenericV2Item(BaseModel):
    id: str
    value: Dict[str, Any]


class SambaStudioGenericV2Response(BaseModel):
    items: list[SambaStudioGenericV2Item]


class SambaStudioGenericV1Response(BaseModel):
    predictions: list[Dict[str, Any]]


class Message(BaseModel):
    content: str
    role: Literal['assistant']


class Choice(BaseModel):
    finish_reason: str
    index: int
    logprobs: str | None
    message: Message


class Usage(BaseModel):
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


class SNCloudResponse(BaseModel):
    choices: list[Choice]
    created: int
    id: str
    model: str
    object: str
    system_fingerprint: str
    usage: Usage


class EmbeddingsBaseModel(BaseModel):
    type: str
    batch_size: int
    coe: bool
    select_expert: str


class LLMBaseModel(BaseModel):
    type: str
    model: str
    max_tokens: int
    temperature: float
    streaming: bool
    coe: Optional[bool] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None
    select_expert: Optional[str] = None
