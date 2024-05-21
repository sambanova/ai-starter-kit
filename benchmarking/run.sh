# COE model
python src/token_benchmark_ray.py \
--model "COE/llama-2-7b-chat-hf" \
--mean-input-tokens 150 \
--stddev-input-tokens 10 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 32 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "./data/results/llmperf" \
--llm-api sambanova \
--additional-sampling-params '{}'

# # Non-COE model: Remember to source .env for new endpoint
# python src/token_benchmark_ray.py \
# --model "llama-2-7b-chat-hf" \
# --mean-input-tokens 150 \
# --stddev-input-tokens 10 \
# --mean-output-tokens 150 \
# --stddev-output-tokens 10 \
# --max-num-completed-requests 32 \
# --timeout 600 \
# --num-concurrent-requests 1 \
# --results-dir "./data/results/llmperf" \
# --llm-api sambanova \
# --additional-sampling-params '{}'