#!/bin/bash
# run_synthetic_dataset.sh

python src/evaluator.py \
--mode synthetic \
--model-names "Meta-Llama-3.1-70B-Instruct" \
--results-dir "./data/results/ankur_softbank" \
--num-concurrent-requests 1 \
--timeout 600 \
--num-input-tokens 100 \
--num-output-tokens 100 \
--num-requests 1 \
--llm-api sncloud

# Notes:
# 1. For Bundle Models, make sure to include the prefix "Bundle/" before each expert name.
#   For example:
#      --model-names "Bundle/llama-2-7b-chat-hf"
#          OR
#      --model-names "Bundle/llama-2-7b-chat-hf Bundle/llama-2-13b-chat-hf"
#          OR
#      --model-names "Bundle/llama-2-7b-chat-hf Bundle/Mistral-7B-Instruct-v0.2"
#          OR
#      --model-names "Bundle/Meta-Llama-3-8B-Instruct"
#
# 2. For Non-Bundle models, use the model name directly and remember to update and source the `.env` file for a new endpoint.
#   For example:
#      --model-names "llama-2-7b-chat-hf"
#          OR
#      --model-names "llama-2-13b-chat-hf"
#          OR
#      --model-names "Mistral-7B-Instruct-v0.2"
#          OR
#      --model-names "Meta-Llama-3-8B-Instruct"
#
# 3. For SambaNovaCloud endpoints, change the llm-api parameter to "sncloud" and use the model name directly.
#   For example:
#      --model-names "Meta-Llama-3.1-8B-Instruct"
#          OR
#      --model-names "Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-70B-Instruct"
