#!/bin/bash
# run_synthetic_dataset.sh

ulimit -n 4096
python src/evaluator.py \
--mode synthetic \
--model-names "Meta-Llama-3.3-70B-Instruct" \
--results-dir "./data/results" \
--num-concurrent-requests 1 \
--timeout 600 \
--num-input-tokens 1000 \
--num-output-tokens 1000 \
--multimodal-image-size na \
--num-requests 16 \
--num-warmup-requests 0 \
--use-multiple-prompts False \
--save-llm-responses False \
--use-debugging-mode False \
--llm-api sncloud \
--benchmark-mode kit \
--use-wandb False \
--wandb-project sambanova-benchmarking


# Notes:
# Here are some examples of how to run the script with different models and API endpoints.
#
# 1. SambaNova Cloud 
#
#   1.1 Instruct models

# python src/evaluator.py \
# --mode synthetic \
# --model-names "Meta-Llama-3.3-70B-Instruct" \
# --results-dir "./data/results" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --num-input-tokens 1000 \
# --num-output-tokens 1000 \
# --multimodal-image-size na \
# --num-requests 16 \
# --num-warmup-requests 0 \
# --save-llm-responses False \
# --use-debugging-mode False \
# --llm-api sncloud
# --benchmark-mode both # will run kit and vllm benchmarks and compare them

#   1.2 Multimodal models 

# python src/evaluator.py \
# --mode synthetic \
# --model-names "gemma-4-31B-it" \
# --results-dir "./data/results" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --num-input-tokens 1000 \
# --num-output-tokens 1000 \
# --multimodal-image-size medium \
# --num-requests 16 \
# --num-warmup-requests 0 \
# --save-llm-responses False \
# --use-debugging-mode False \
# --llm-api sncloud
# --benchmark-mode kit