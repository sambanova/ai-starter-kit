JUDGE_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>        
You are an evaluator who generates JSON<|eot_id|><|start_header_id|>user<|end_header_id|>"

Given the evaluation steps, return a JSON with only one key: a `score` key that can take a value from 0 to 1.

Evaluation Steps:
1. You should return a score of 0.0 if the completion contradicts the expected completion.
2. You should return a score of 1.0 if the completion matches the expected completion.
3. You should not reduce the score if the completion and expected completion have minor rephrasing but the content
 is correct.
4. You should return a lower score than 1.0 if the completion and expected completion is not entirely correct,
 ignoring grammar.

query: {query}

generated answer: {generated_answer} 

expected answer: {expected_answer}

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" key and make sure value of
"score" is a valid integer.


Example JSON:
{{
    "score": 0
}}
**

JSON:
<|eot_id|><|start_header_id|>machine-readable JSON<|end_header_id|>\n
"""
