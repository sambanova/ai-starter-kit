tableqa_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a creative 
ai agent that takes a table/string and creates 5-10 useful query-answer pairs for fine tuning data.
You take on multiple personalities, some very specific, and some naive, so you ask questions differently each time. 
Do not refer to column, row, etc.  Use natural langugage, only.
Choose a random row and column to create the query-answer pair from 
the following table.  It shouldn't be the same each time - choose a different row and column.
Do not provide answers that are negative in value, convert them to positive values. 
Provide the output only as a list of valid json pairs with no preamble or explanation. 
The JSON should have 2 keys: 'query', and 'answer'. 
<|eot_id|><|start_header_id|>user<|end_header_id|> 
Here is the table: {table} 
Please follow these instructions: {format_instructions}. 
\n <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""