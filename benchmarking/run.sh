#!/bin/bash
# run.sh

# COE Turbo model
# --model "COE/llama-2-7b-chat-hf" \
# --model "COE/llama-2-13b-chat-hf" \
# --model "COE/Mistral-7B-Instruct-v0.2" \
# --model "COE/Meta-Llama-3-8B-Instruct" \
python src/token_benchmark.py \
--model "COE/Meta-Llama-3-8B-Instruct" \
--mean-input-tokens 1000 \
--mean-output-tokens 1000 \
--max-num-completed-requests 32 \
--num-concurrent-workers 1 \
--timeout 600 \
--results-dir "./data/results/llmperf" \
--additional-sampling-params '{}'

# # Non-COE model: Remember to source .env for new endpoint
# python src/token_benchmark.py \
# --model "llama-2-7b-chat" \
# --mean-input-tokens 1000 \
# --mean-output-tokens 1000 \
# --max-num-completed-requests 32 \
# --num-concurrent-workers 1 \
# --timeout 600 \
# --results-dir "./data/results/llmperf" \
# --additional-sampling-params '{}'