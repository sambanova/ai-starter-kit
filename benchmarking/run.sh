# COE Turbo model
# --model "COE/llama-2-7b-chat-hf" \
# --model "COE/Llama-2-13B-chat-hf" \
# --model "COE/Mistral-7B-Instruct-V0.2" \
# --model "COE/Meta-Llama-3-8B-Instruct" \
python src/token_benchmark_ray.py \
--model "COE/Meta-Llama-3-8B-Instruct" \
--mean-input-tokens 1000 \
--stddev-input-tokens 10 \
--mean-output-tokens 1000 \
--stddev-output-tokens 10 \
--max-num-completed-requests 32 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "./data/results/llmperf" \
--llm-api sambastudio \
--mode stream \
--additional-sampling-params '{}'

# # Non-COE model: Remember to source .env for new endpoint
# python src/token_benchmark_ray.py \
# --model "llama-2-7b-chat" \
# --mean-input-tokens 150 \
# --stddev-input-tokens 10 \
# --mean-output-tokens 150 \
# --stddev-output-tokens 10 \
# --max-num-completed-requests 10 \
# --timeout 600 \
# --num-concurrent-requests 1 \
# --results-dir "./data/results/llmperf" \
# --llm-api sambastudio \
# --additional-sampling-params '{}'