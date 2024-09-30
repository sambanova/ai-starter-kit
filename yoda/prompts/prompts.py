# Prompt prefix and postfix for llama chat model
LLAMA_CHAT_PROMPT_PREFIX = '[INST] '
LLAMA_CHAT_PROMPT_POSTFIX = ' [\INST]'

QA_GEN_TEMPLATE = """

    Based on the previous context article, please generate 10 question-answer pairs in the format of:
    <human>:{question}\n<bot>:{answer}

    Phrase the question so that it does NOT refer to specific context. \
    For instance, do NOT put phrases like "given provided context" or \
    "in this work" in the question, because if the question is asked elsewhere \
    it wouldn't be provided specific context. \
    Replace these terms with specific details.

    Separate the question-answer pairs with '###'.

    Generate the question-answer pairs below:
    """

RAG_prompt_template = """
[INST] Here is some relevant context that might assist in answering the SambaNova-related question.\n\n'
```
{context}
```
Answer the following SambaNova-related question: {question}\n\nAnswer: [/INST]"""
