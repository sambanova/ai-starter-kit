#!/bin/bash
# run_custom_dataset.sh
# Meta-Llama-3.3-70B-Instruct
# gpt-oss-120b

ulimit -n 4096
python src/evaluator.py \
--mode custom \
--model-name "Meta-Llama-3.1-8B-Instruct" \
--results-dir "./data/bundle_tests/3d1_8b/custom_prompts" \
--num-concurrent-requests 1 \
--timeout 600 \
--input-file-path "../benchmarking/prompts/bundle_tests/3d1_8b/3s1_8b_bundle.jsonl" \
--save-llm-responses True \
--sampling-params '{"max_tokens_to_generate": 100}' \
--use-debugging-mode False \
--llm-api sncloud

# Notes:
# Here are some examples of how to run the script with different models and API endpoints.
#
# 1. SambaStudio 
#   1.1 Instruct models

# python src/evaluator.py \
# --mode custom \
# --model-name "Bundle/Meta-Llama-3-70B-Instruct-4096" \
# --results-dir "./data/results/llmperf" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --input-file-path "<AISK_REPOSITORY_PATH>/benchmarking/prompts/custom_prompt_example.jsonl" \
# --save-llm-responses False \
# --sampling-params '{"max_tokens_to_generate": 256}' \
# --use-debugging-mode False \
# --llm-api sambastudio

#   1.2 Multimodal models (remember to use OpenAI compatible URL in .env SAMBASTUDIO_URL variable)

# python src/evaluator.py \
# --mode custom \
# --model-name "Meta-Llama-3.2-11B-Vision-Instruct" \
# --results-dir "./data/results/llmperf" \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --input-file-path "<CUSTOM DATASET PATH HERE>" \
# --save-llm-responses False \
# --sampling-params '{"max_tokens_to_generate": 256}' \
# --use-debugging-mode False \
# --llm-api sambastudio

# 2. SambaNova Cloud 
#
#   2.1 Instruct models

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

#   2.2 Multimodal models (remember to use OpenAI compatible URL in .env SAMBASTUDIO_URL variable)

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
