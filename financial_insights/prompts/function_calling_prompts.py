FUNCTION_CALLING_PROMPT_TEMPLATE = """
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>

  You are a helpful assistant with tool-calling capabilities.
  When you receive a tool call response, use the output to format an answer to the original user question.

  Format instructions: {format_instructions}

  <|eot_id|>

  <|start_header_id|>user<|end_header_id|>

  Given the following functions/tools,
  please respond with a JSON for a function call that best answers the given prompt.
  Include the function name and its proper arguments.

  Respond in the JSON format:
  
  ```json
  [{{
  "name": function name,
    "parameters": dictionary of argument name and its value
  }}]
  ```

  Do not use variables.

  Available tools:
  {tools}

  Please select only one tool.

  Question: {user_query}

  <|eot_id|>

  <|start_header_id|>assistant<|end_header_id|>
"""
