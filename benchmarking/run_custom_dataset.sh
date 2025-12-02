#!/bin/bash
# run_custom_dataset.sh
# Meta-Llama-3.3-70B-Instruct
# gpt-oss-120b

ulimit -n 4096
python src/evaluator.py \
--mode custom \
--model-name "gpt-oss-120b" \
--results-dir "./data/bundle_tests/audio/custom_prompts" \
--num-concurrent-requests 1 \
--timeout 600 \
--input-file-path "../benchmarking/prompts/bundle_tests/audio/audio.jsonl" \
--save-llm-responses True \
--sampling-params '{"max_tokens_to_generate": 200}' \
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
# --results-dir "./data/results/llmperf" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --input-file-path "<CUSTOM DATASET PATH HERE>" \
# --save-llm-responses False \
# --sampling-params '{"max_tokens_to_generate": 256}' \
# --use-debugging-mode False \
# --llm-api sncloud

#   1.2 Multimodal models 

# python src/evaluator.py \
# --mode custom \
# --model-name "Llama-3.2-11B-Vision-Instruct" \
# --results-dir "./data/results/llmperf" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --input-file-path "<CUSTOM DATASET PATH HERE>" \
# --save-llm-responses False \
# --sampling-params '{"max_tokens_to_generate": 256}' \
# --use-debugging-mode False \
# --llm-api sncloud
