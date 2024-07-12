#!/bin/bash
# run_synthetic_dataset.sh

python src/evaluator.py \
--mode synthetic \
--model-name "COE/Meta-Llama-3-8B-Instruct" \
--results-dir "./data/results/llmperf" \
--num-workers 1 \
--timeout 600 \
--num-input-tokens 1000 \
--num-output-tokens 1000 \
--num-requests 32 \

# Notes:
# 1. For CoE Models, make sure to include the prefix "COE/" before the expert name.
#   For example:
#      --model-name "COE/llama-2-7b-chat-hf"
#          OR
#      --model-name "COE/llama-2-13b-chat-hf"
#          OR
#      --model-name "COE/Mistral-7B-Instruct-v0.2"
#          OR
#      --model-name "COE/Meta-Llama-3-8B-Instruct"
#
# 2. For Non-CoE models, use the model name directly and remember to update and source the `.env` file for a new endpoint.
#   For example:
#      --model-name "llama-2-7b-chat-hf"
#          OR
#      --model-name "llama-2-13b-chat-hf"
#          OR
#      --model-name "Mistral-7B-Instruct-v0.2"
#          OR
#      --model-name "Meta-Llama-3-8B-Instruct"