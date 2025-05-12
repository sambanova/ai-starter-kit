from typing import List

from pydantic import BaseModel


class DataExtraction(BaseModel):
    section: str
    products: List[str]


class ContactForm(BaseModel):
    name: str
    phone_number: float


class Step(BaseModel):
    explanation: str
    output: str


class Solution(BaseModel):
    final_answer: str
    steps: List[Step]
