from typing import Literal

from pydantic import BaseModel


class SNCloudSchema(BaseModel):
    type: str
    model: str
    temperature: float
    max_tokens: int


class EmbeddingsSchema(BaseModel):
    model: str


class VectorDBSchema(BaseModel):
    db_type: Literal['chroma'] = 'chroma'
    collection_name: str = 'demo'
