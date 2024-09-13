QA_RETRIEVAL_PROMPT_TEMPLATE = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Answer any use questions based solely on the context below:

    <context>

    {context}

    </context>

    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>

    {input}

    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
"""
