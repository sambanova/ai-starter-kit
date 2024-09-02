FUNCTION_CALLING_PROMPT_TEMPLATE = """
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>

  When you receive a tool call response, use the output to format an answer to the orginal user question.

  You are a helpful assistant with tool calling capabilities.

  Format instructions: {format_instructions}

  <|eot_id|>

  <|start_header_id|>user<|end_header_id|>

  Given the following functions/tools, please respond with a JSON for a function call
  with its proper arguments that best answers the given prompt.

  Respond in the JSON format

  ```json
  [{{
  "name": function name,
    "parameters": dictionary of argument name and its value
  }}]
  ```

  Do not use variables.

  Available tools:
  {tools}

  Please only select one tool.

  Question: {user_query}

  <|eot_id|>

  |start_header_id|>assistant<|end_header_id|>
"""
