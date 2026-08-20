#!/bin/bash
# quickstart_custom.sh
# Custom Performance Evaluation: benchmarks the Kit against YOUR OWN dataset instead of
# Kit-generated prompts. Always runs `--mode custom` explicitly. Edit the variables below,
# then run: sh quickstart_custom.sh

ulimit -n 4096  # raise open-file-descriptor limit for higher-concurrency runs

# --- Common parameters ---
MODEL_NAMES="Meta-Llama-3.3-70B-Instruct"          # Model name(s), space-separated for multiple
RESULTS_DIR="./data/results"                       # Where result files get written
LLM_API=sncloud                                    # API type (only 'sncloud' supported)
TIMEOUT=600                                        # Seconds before the run times out
NUM_WARMUP_REQUESTS=0                              # Throwaway requests before the measured run; 0 disables
USE_DEBUGGING_MODE=False                           # True/False; more detail per request, adds latency
SAMPLING_PARAMS='{"max_tokens_to_generate": 256}'  # Must include max_tokens_to_generate (no NUM_OUTPUT_TOKENS fallback in this mode)

# --- Custom-mode parameters ---
INPUT_FILE_PATH="../../prompts/custom_prompt_example.jsonl"  # Your dataset (.jsonl, one {"prompt": ...} per line); required
NUM_CONCURRENT_REQUESTS=16                         # Max requests in flight at once (closed-loop concurrency)
SAVE_LLM_RESPONSES=False                           # True/False; also save each response's text to a file

# Note: no NUM_REQUESTS/NUM_INPUT_TOKENS/NUM_OUTPUT_TOKENS/QPS/MULTIMODAL_IMAGE_SIZE/
# USE_MULTIPLE_PROMPTS here -- your dataset file determines those directly.

python src/evaluator.py \
    --mode custom \
    --model-names "$MODEL_NAMES" \
    --results-dir "$RESULTS_DIR" \
    --llm-api "$LLM_API" \
    --timeout "$TIMEOUT" \
    --num-warmup-requests "$NUM_WARMUP_REQUESTS" \
    --use-debugging-mode "$USE_DEBUGGING_MODE" \
    --sampling-params "$SAMPLING_PARAMS" \
    --input-file-path "$INPUT_FILE_PATH" \
    --num-concurrent-requests "$NUM_CONCURRENT_REQUESTS" \
    --save-llm-responses "$SAVE_LLM_RESPONSES"
