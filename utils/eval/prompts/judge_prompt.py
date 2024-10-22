JUDGE_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>        
You are an evaluator who generates JSON<|eot_id|><|start_header_id|>user<|end_header_id|>"

Given the evaluation steps, return a JSON with the "score" and "reason" key.

Evaluation Steps:
1. You should return a score of 0.0 if the completion contradicts the expected completion.
2. You should return a score of 1.0 if the completion matches the expected completion.
3. You should not reduce the score if the completion and expected completion have minor rephrasing or if the completion
 larger than the expected completion but the content is correct.
4. You should return a lower score than 1.0 if the completion and expected completion is not entirely correct,
 ignoring grammar.
5. When providing a reason for the score, explain the merits (if any) and demerits (if any) of the completion in
 relation to the expected completion. 

query: {query}

generated answer: {generated_answer} 

expected answer: {expected_answer}

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" key and make sure value of
"score" is a valid integer.


Example JSON:
{{
    "score": 0,
    "reason": "The text does not follow the evaluation steps provided."
}}
**

JSON:
<|eot_id|><|start_header_id|>machine-readable JSON<|end_header_id|>\n
"""
