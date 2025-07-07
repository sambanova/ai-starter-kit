from typing import Literal

from pydantic import BaseModel


class SNCloudSchema(BaseModel):
    type: str
    model: str
    temperature: float
    max_tokens: int


class EmbeddingsSchema(BaseModel):
    type: Literal['sncloud' ,'cpu', 'sambastudio'] = 'sncloud'
    batch_size: int = 1
    bundle: bool = True
    model: str


class VectorDBSchema(BaseModel):
    db_type: Literal['chroma'] = 'chroma'
    collection_name: str = 'demo'
