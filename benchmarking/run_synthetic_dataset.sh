#!/bin/bash
# run_synthetic_dataset.sh

ulimit -n 4096
python src/evaluator.py \
--mode synthetic \
--model-names "Meta-Llama-3.1-8B-Instruct" \
--results-dir "./data/results/llmperf" \
--num-concurrent-requests 1 \
--timeout 600 \
--num-input-tokens 100 \
--num-output-tokens 100 \
--multimodal-image-size na \
--num-requests 1 \
--use-multiple-prompts False \
--save-llm-responses False \
--use-debugging-mode False \
--llm-api sncloud


# Notes:
# Here are some examples of how to run the script with different models and API endpoints.
#
# 1. SambaNova Cloud 
#
#   1.1 Instruct models

# python src/evaluator.py \
# --mode synthetic \
# --model-names "Meta-Llama-3.3-70B-Instruct" \
# --results-dir "./data/results/llmperf" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --num-input-tokens 1000 \
# --num-output-tokens 1000 \
# --multimodal-image-size na \
# --num-requests 16 \
# --save-llm-responses False \
# --use-debugging-mode False \
# --llm-api sncloud

#   1.2 Multimodal models 

# python src/evaluator.py \
# --mode synthetic \
# --model-names "Llama-3.2-11B-Vision-Instruct" \
# --results-dir "./data/results/llmperf" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --num-input-tokens 1000 \
# --num-output-tokens 1000 \
# --multimodal-image-size medium \
# --num-requests 16 \
# --save-llm-responses False \
# --use-debugging-mode False \
# --llm-api sncloud