#!/bin/bash
# run_custom_dataset.sh

python src/evaluator.py \
--mode custom \
--model-name "COE/Meta-Llama-3-8B-Instruct" \
--results-dir "./data/results/llmperf" \
--num-workers 1 \
--timeout 600 \
--input-file-path "<CUSTOM DATASET PATH HERE>" \
--llm-api "sambastudio"

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
# 4. For FastAPI endpoints, change the llm-api parameter to "fastapi" and use the model name directly.
#   For example:
#      --model-name "llama3-8b"