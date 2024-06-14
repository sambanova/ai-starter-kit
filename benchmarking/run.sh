#!/bin/bash
# run.sh

# COE Turbo model
# --model "COE/llama-2-7b-chat-hf" \
# --model "COE/llama-2-13b-chat-hf" \
# --model "COE/Mistral-7B-Instruct-v0.2" \
# --model "COE/Meta-Llama-3-8B-Instruct" \
python src/token_benchmark_ray.py \
--model "COE/Mistral-7B-Instruct-v0.2" \
--mean-input-tokens 50 \
--stddev-input-tokens 10 \
--mean-output-tokens 50 \
--stddev-output-tokens 10 \
--max-num-completed-requests 10 \
--num-concurrent-requests 1 \
--timeout 600 \
--results-dir "./data/results/llmperf" \
--additional-sampling-params '{}'

# # Non-COE model: Remember to source .env for new endpoint
# python src/token_benchmark_ray.py \
# --model "llama-2-7b-chat" \
# --mean-input-tokens 150 \
# --stddev-input-tokens 10 \
# --mean-output-tokens 150 \
# --stddev-output-tokens 10 \
# --max-num-completed-requests 10 \
# --num-concurrent-requests 1 \
# --timeout 600 \
# --results-dir "./data/results/llmperf" \
# --additional-sampling-params '{}'