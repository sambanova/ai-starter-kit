#!/bin/bash
# run_custom_dataset.sh

ulimit -n 4096
python src/evaluator.py \
--mode custom \
--model-name "Meta-Llama-3.3-70B-Instruct" \
--results-dir "./data/results" \
--num-concurrent-requests 1 \
--timeout 600 \
--input-file-path "<AISK_REPOSITORY_PATH>/prompts/custom_prompt_example.jsonl" \
--num-warmup-requests 0 \
--save-llm-responses False \
--sampling-params '{"max_tokens_to_generate": 256}' \
--use-debugging-mode False \
--llm-api sncloud

# Notes:
# Here are some examples of how to run the script with different models and API endpoints.
#
# 1. SambaNova Cloud 
#
#   1.1 Instruct models

# python src/evaluator.py \
# --mode custom \
# --model-name "Meta-Llama-3.3-70B-Instruct" \
# --results-dir "./data/results" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --input-file-path "<CUSTOM DATASET PATH HERE>" \
# --num-warmup-requests 0 \
# --save-llm-responses False \
# --sampling-params '{"max_tokens_to_generate": 256}' \
# --use-debugging-mode False \
# --llm-api sncloud

#   1.2 Multimodal models 

# python src/evaluator.py \
# --mode custom \
# --model-name "gemma-4-31B-it" \
# --results-dir "./data/results" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --input-file-path "<CUSTOM DATASET PATH HERE>" \
# --num-warmup-requests 0 \
# --save-llm-responses False \
# --sampling-params '{"max_tokens_to_generate": 256}' \
# --use-debugging-mode False \
# --llm-api sncloud
