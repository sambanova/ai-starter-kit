#!/bin/bash
# run_synthetic_dataset.sh

ulimit -n 4096
python src/evaluator.py \
--mode synthetic \
--model-names "gpt-oss-120b DeepSeek-R1-0528 DeepSeek-R1-Distill-Llama-70B DeepSeek-V3.1 DeepSeek-V3.1-Terminus Qwen3-32B" \
--results-dir "./data/results/resoning_support" \
--num-concurrent-requests 1 \
--timeout 600 \
--num-input-tokens 500 \
--num-output-tokens 300 \
--multimodal-image-size na \
--num-requests 1 \
--use-multiple-prompts False \
--save-llm-responses True \
--use-debugging-mode False \
--llm-api sncloud


# Notes:
# Here are some examples of how to run the script with different models and API endpoints.
#
# 1. SambaStudio 
#   1.1 Instruct models

# python src/evaluator.py \
# --mode synthetic \
# --model-names "Bundle/Meta-Llama-3-70B-Instruct-4096" \
# --results-dir "./data/results/llmperf" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --num-input-tokens 1000 \
# --num-output-tokens 1000 \
# --multimodal-image-size na \
# --num-requests 16 \
# --save-llm-responses False \
# --use-debugging-mode False \
# --llm-api sambastudio

#   1.2 Multimodal models (remember to use OpenAI compatible URL in .env SAMBASTUDIO_URL variable)

# python src/evaluator.py \
# --mode synthetic \
# --model-names "Meta-Llama-3.2-11B-Vision-Instruct" \
# --results-dir "./data/results/llmperf" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --num-input-tokens 1000 \
# --num-output-tokens 1000 \
# --multimodal-image-size medium \
# --num-requests 16 \
# --save-llm-responses False \
# --use-debugging-mode False \
# --llm-api sambastudio

# 2. SambaNova Cloud 
#
#   2.1 Instruct models

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

#   2.2 Multimodal models (remember to use OpenAI compatible URL in .env SAMBASTUDIO_URL variable)

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