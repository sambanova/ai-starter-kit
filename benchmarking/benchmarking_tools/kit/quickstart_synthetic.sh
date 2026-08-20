#!/bin/bash
# quickstart_synthetic.sh
# Synthetic Performance Evaluation: the Kit generates N synthetic prompts and fires them all as
# a single concurrency-capped burst -- measures throughput/latency at a fixed batch size.
# Always runs `--mode synthetic` explicitly. Edit the variables below, then run: sh quickstart_synthetic.sh

ulimit -n 4096  # raise open-file-descriptor limit for higher-concurrency runs

# --- Common parameters ---
MODEL_NAMES="Meta-Llama-3.3-70B-Instruct"  # Model name(s), space-separated for multiple
RESULTS_DIR="./data/results"               # Where result files get written
LLM_API=sncloud                            # API type (only 'sncloud' supported)
TIMEOUT=600                                # Seconds before the run times out
NUM_WARMUP_REQUESTS=0                      # Throwaway requests before the measured run; 0 disables
USE_DEBUGGING_MODE=False                   # True/False; more detail per request, adds latency
SAMPLING_PARAMS='{}'                       # Extra sampling params JSON, e.g. '{"temperature": 0.7}'

# --- Synthetic-mode parameters ---
NUM_REQUESTS=10                            # Total number of requests to send
NUM_CONCURRENT_REQUESTS=10                 # Max requests in flight at once; = NUM_REQUESTS fires one wave
NUM_INPUT_TOKENS=1000                      # Exact input token count per prompt (<=2000 recommended)
NUM_OUTPUT_TOKENS=1000                     # Max output tokens per request (<=2000 recommended)
MULTIMODAL_IMAGE_SIZE=na                   # 'na' | 'small' | 'medium' | 'large'; only for multimodal models
USE_MULTIPLE_PROMPTS=True                 # True/False; cycle random prompt templates instead of one
SAVE_LLM_RESPONSES=True                   # True/False; also save each response's text to a file

python src/evaluator.py \
    --mode synthetic \
    --model-names "$MODEL_NAMES" \
    --results-dir "$RESULTS_DIR" \
    --llm-api "$LLM_API" \
    --timeout "$TIMEOUT" \
    --num-warmup-requests "$NUM_WARMUP_REQUESTS" \
    --use-debugging-mode "$USE_DEBUGGING_MODE" \
    --sampling-params "$SAMPLING_PARAMS" \
    --num-requests "$NUM_REQUESTS" \
    --num-concurrent-requests "$NUM_CONCURRENT_REQUESTS" \
    --num-input-tokens "$NUM_INPUT_TOKENS" \
    --num-output-tokens "$NUM_OUTPUT_TOKENS" \
    --multimodal-image-size "$MULTIMODAL_IMAGE_SIZE" \
    --use-multiple-prompts "$USE_MULTIPLE_PROMPTS" \
    --save-llm-responses "$SAVE_LLM_RESPONSES"
