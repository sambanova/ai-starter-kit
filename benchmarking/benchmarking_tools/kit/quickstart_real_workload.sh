#!/bin/bash
# quickstart_real_workload.sh
# Real Workload Performance Evaluation: the Kit generates synthetic prompts and paces requests
# over time at a target QPS instead of firing a concurrency-capped burst -- an open-loop
# schedule, closer to real production traffic. Always runs `--mode real_workload` explicitly.
# Edit the variables below, then run: sh quickstart_real_workload.sh

ulimit -n 4096  # raise open-file-descriptor limit for higher-concurrency runs

# --- Common parameters ---
MODEL_NAMES="Meta-Llama-3.3-70B-Instruct"  # Model name(s), space-separated for multiple
RESULTS_DIR="./data/results"               # Where result files get written
LLM_API=sncloud                            # API type (only 'sncloud' supported)
TIMEOUT=3                                # Seconds before the run times out
NUM_WARMUP_REQUESTS=5                      # Throwaway requests before the measured run (fired together, not QPS-paced); 0 disables
USE_DEBUGGING_MODE=False                   # True/False; more detail per request, adds latency
SAMPLING_PARAMS='{"temperature": 0.7}'                       # Extra sampling params JSON, e.g. '{"temperature": 0.7}'

# --- Real-workload-mode parameters ---
NUM_REQUESTS=16                            # Total number of requests to send over the run
QPS=1                                      # Target queries per second (<10 recommended to avoid rate limits)
QPS_DISTRIBUTION=constant                  # 'constant' | 'uniform' | 'exponential' (Poisson) inter-arrival pacing
NUM_INPUT_TOKENS=2000                      # Exact input token count per prompt (<=2000 recommended)
NUM_OUTPUT_TOKENS=1500                     # Max output tokens per request (<=2000 recommended)
MULTIMODAL_IMAGE_SIZE=na                   # 'na' | 'small' | 'medium' | 'large'; required, only for multimodal models
USE_MULTIPLE_PROMPTS=False                 # True/False; cycle random prompt templates instead of one
SAVE_LLM_RESPONSES=False                   # True/False; also save each response's text to a file

# Note: no NUM_CONCURRENT_REQUESTS here -- QPS pacing is open-loop, there's no concurrency cap to set.

python src/evaluator.py \
    --mode real_workload \
    --model-names "$MODEL_NAMES" \
    --results-dir "$RESULTS_DIR" \
    --llm-api "$LLM_API" \
    --timeout "$TIMEOUT" \
    --num-warmup-requests "$NUM_WARMUP_REQUESTS" \
    --use-debugging-mode "$USE_DEBUGGING_MODE" \
    --sampling-params "$SAMPLING_PARAMS" \
    --num-requests "$NUM_REQUESTS" \
    --qps "$QPS" \
    --qps-distribution "$QPS_DISTRIBUTION" \
    --num-input-tokens "$NUM_INPUT_TOKENS" \
    --num-output-tokens "$NUM_OUTPUT_TOKENS" \
    --multimodal-image-size "$MULTIMODAL_IMAGE_SIZE" \
    --use-multiple-prompts "$USE_MULTIPLE_PROMPTS" \
    --save-llm-responses "$SAVE_LLM_RESPONSES"
