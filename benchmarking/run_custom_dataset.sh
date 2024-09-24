#!/bin/bash
# run_custom_dataset.sh

python src/evaluator.py \
--mode custom \
--model-name "llama3-405b" \
--results-dir "./data/results/llmperf" \
--num-concurrent-requests 1 \
--timeout 600 \
--input-file-path "/Users/rodrigom/Desktop/aisk/benchmarking/custom_benchmark_test_data.jsonl" \
--save-llm-responses False \
--sampling-params '{"max_tokens": 256}' \
--llm-api sncloud

# Notes:
# 1. Replace <CUSTOM DATASET PATH HERE> with the path to your custom dataset.
#
# 2. For CoE Models, make sure to include the prefix "COE/" before the expert name.
#   For example:
#      --model-name "COE/llama-2-7b-chat-hf"
#          OR
#      --model-name "COE/llama-2-13b-chat-hf"
#          OR
#      --model-name "COE/Mistral-7B-Instruct-v0.2"
#          OR
#      --model-name "COE/Meta-Llama-3-8B-Instruct"
#
# 3. For Non-CoE models, use the model name directly and remember to update and source the `.env` file for a new endpoint.
#   For example:
#      --model-name "llama-2-7b-chat-hf"
#          OR
#      --model-name "llama-2-13b-chat-hf"
#          OR
#      --model-name "Mistral-7B-Instruct-v0.2"
#          OR
#      --model-name "Meta-Llama-3-8B-Instruct"
#
# 4. For SambaNovaCloud endpoints, change the llm-api parameter to "snloud" and use the model name directly.
#   For example:
#      --model-name "llama3-8b"
#
# 5. You can modify the number of maximum tokens depending on the LLM endpoint type.
#   If it's "sambastudio": add or modify in the `--sampling-params` dictionary the `max_tokens_to_generate` key and a value
#   If it's "snloud": add or modify in the `--sampling-params` dictionary the `max_tokens` key and a value   
#