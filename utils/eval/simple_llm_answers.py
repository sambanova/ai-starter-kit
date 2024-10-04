from typing import Any, Dict

from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate


class SimpleLLMAnswers:
    """Simple LLM answer generation without a vector DB"""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
        self.prompt = PromptTemplate(template='Question: {question}\nAnswer:', input_variables=['question'])

    def generate(self, query: str) -> Dict[str, Any]:
        """Generate an answer for the given query"""
        response = self.llm(self.prompt.format(question=query))
        return {'answer': response}
