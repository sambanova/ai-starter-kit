from typing import Any, Dict

from langchain_core.messages import AIMessage


def custom_parser(ai_message: AIMessage) -> Dict[str, Any]:
    return {'content': ai_message.content, 'metadata': ai_message.response_metadata}


def calculate_cost(input_tokens: int, output_tokens: int, input_token_cost: float, ouput_token_cost: float) -> float:
    if input_tokens is None or output_tokens is None:
        return 0.0
    total_cost = (input_tokens * (input_token_cost / 1000000)) + (output_tokens * (ouput_token_cost / 1000000))
    return total_cost
