from typing import Optional

from pydantic import BaseModel


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
