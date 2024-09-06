CONVERSATIONAL_RESPONSE_PROMPT_TEMPLATE = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are an expert in financial markets.
    Format instructions: {format_instructions}
    Here is the user's request:

    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>

    {user_query}

    Here is the response object:
    {response_string}

    Please rephrase and return the response object in a conversational but formal style.
    Simplify and round numbers for easy readability.

    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
"""
