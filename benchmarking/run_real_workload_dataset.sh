#!/bin/bash
# run_real_workload_dataset.sh

ulimit -n 4096
python src/evaluator.py \
--mode real_workload \
--model-name "Meta-Llama-3.3-70B-Instruct" \
--results-dir "./data/results/llmperf" \
--qps 1 \
--qps-distribution "constant" \
--timeout 600 \
--num-input-tokens 1000 \
--num-output-tokens 1000 \
--multimodal-image-size na \
--num-requests 16 \
--use-debugging-mode False \
--llm-api sncloud


# Notes:
# Here are some examples of how to run the script with different models and API endpoints.
#
# 1. SambaNova Cloud 
#
#   1.1 Instruct models

# python src/evaluator.py \
# --mode real_workload \
# --model-names "Meta-Llama-3.3-70B-Instruct" \
# --results-dir "./data/results/llmperf" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --num-input-tokens 1000 \
# --num-output-tokens 1000 \
# --multimodal-image-size na \
# --num-requests 16 \
# --use-debugging-mode False \
# --llm-api sncloud

#   1.2 Multimodal models 

# python src/evaluator.py \
# --mode real_workload \
# --model-names "Llama-3.2-11B-Vision-Instruct" \
# --results-dir "./data/results/llmperf" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --num-input-tokens 1000 \
# --num-output-tokens 1000 \
# --multimodal-image-size medium \
# --num-requests 16 \
# --use-debugging-mode False \
# --llm-api sncloud
