from pydantic import BaseModel
from typing import List


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