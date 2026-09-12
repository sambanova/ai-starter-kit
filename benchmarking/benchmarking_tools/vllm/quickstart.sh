#!/usr/bin/env bash
# quickstart.sh -- runs vLLM's own `vllm bench serve` CLI directly against a SambaNova endpoint
# (no Python wrapper): generate a dataset, run vLLM's benchmark, standardize its output into the
# Kit's convention. See ./README.md for custom datasets and the full vLLM flag reference.
# Edit the variables below, then run: sh quickstart.sh

set -euo pipefail

# ================================================================================================
# Parameters -- edit these
# ================================================================================================

MODEL_NAME="Meta-Llama-3.3-70B-Instruct"  # Actual API model name (--served_model_name)

# --- Load shape ---
NUM_REQUESTS=16                  # Total number of requests to send
NUM_CONCURRENT_REQUESTS=16       # Max requests in flight at once (closed-loop concurrency ceiling)
NUM_WARMUP_REQUESTS=0            # Throwaway requests before the measured run; 0 disables
QPS="inf"                        # Requests/sec; "inf" fires a closed-loop burst, a number paces via a Poisson process

# --- Prompt shape (synthetic dataset only) ---
NUM_INPUT_TOKENS=1000            # Exact input token count per generated prompt
NUM_OUTPUT_TOKENS=1000           # Max output tokens per request

# --- Paths ---
RESULT_DIR="./data/results/vllm"                            # Where vLLM writes its own result files
DATASET_PATH="$RESULT_DIR/vllm_synthetic_dataset.jsonl"     # Generated dataset, written inside RESULT_DIR
# To use your own prompts instead, comment out Step 1 below and point this at a
# {"prompt": "..."} .jsonl file instead, e.g.:
#   DATASET_PATH="../../prompts/custom_prompt_example.jsonl"

# ================================================================================================
# Everything below runs automatically -- no need to edit
# ================================================================================================

ulimit -n 4096  # raise open-file-descriptor limit for higher-concurrency runs

source ../kit/src/load_env.sh  # loads SAMBANOVA_API_BASE/KEY from .env unless already exported

# vLLM concatenates base-url + endpoint with no separator, so this must end in "/" or the request
# silently retries for 600s instead of erroring (confirmed live) -- enforced below.
SAMBANOVA_API_BASE="${SAMBANOVA_API_BASE:-https://api.sambanova.ai/v1}"
case "$SAMBANOVA_API_BASE" in
    */) ;;
    *) SAMBANOVA_API_BASE="${SAMBANOVA_API_BASE}/" ;;
esac
export OPENAI_API_KEY="${SAMBANOVA_API_KEY:-}"  # vLLM reads its API key from this env var, not SAMBANOVA_API_KEY

# HF tokenizer id vLLM uses locally (--model), auto-resolved from MODEL_NAME so it stays in sync
# with the Kit's own model registry. Override by exporting TOKENIZER_MODEL_NAME yourself.
TOKENIZER_MODEL_NAME="${TOKENIZER_MODEL_NAME:-$(python ../kit/src/resolve_tokenizer_name.py --model-name "$MODEL_NAME")}"

# --- Step 1: generate a synthetic dataset (skip and edit DATASET_PATH above to bring your own) ---
python ../kit/src/generate_dataset.py \
    --model-name "$MODEL_NAME" \
    --num-requests "$NUM_REQUESTS" \
    --num-input-tokens "$NUM_INPUT_TOKENS" \
    --output-path "$DATASET_PATH"

# --- Step 2: run vLLM's own benchmark CLI against that dataset ---
mkdir -p "$RESULT_DIR"
BEFORE_FILES="$(find "$RESULT_DIR" -maxdepth 1 -name '*.json' | sort)"

VLLM_ARGS=(
    bench serve
    --backend openai-chat
    --base-url "$SAMBANOVA_API_BASE"
    --dataset-name custom
    --dataset-path "$DATASET_PATH"
    --endpoint chat/completions
    --model "$TOKENIZER_MODEL_NAME"
    --served_model_name "$MODEL_NAME"
    --custom-output-len "$NUM_OUTPUT_TOKENS"
    --num-prompts "$NUM_REQUESTS"
    --max-concurrency "$NUM_CONCURRENT_REQUESTS"
    --request-rate "$QPS"
    --burstiness 1
    --save-detailed
    --save-result
    --result-dir "$RESULT_DIR"
)
if [ "$NUM_WARMUP_REQUESTS" -gt 0 ]; then
    VLLM_ARGS+=(--num-warmups "$NUM_WARMUP_REQUESTS")
fi

# Need a flag not listed above? Add it directly to VLLM_ARGS -- this repo doesn't wrap or
# validate vLLM's CLI, so any native flag works (see `vllm bench serve --help`).
vllm "${VLLM_ARGS[@]}"

# --- Step 3: standardize vLLM's output into the Kit's _individual_responses.json/_summary.json ---
AFTER_FILES="$(find "$RESULT_DIR" -maxdepth 1 -name '*.json' | sort)"
NEW_FILES="$(comm -13 <(printf '%s\n' "$BEFORE_FILES") <(printf '%s\n' "$AFTER_FILES"))"
RAW_RESULT_PATH="${NEW_FILES%%$'\n'*}"
if [ -n "$RAW_RESULT_PATH" ]; then
    python ../kit/src/convert_vllm_output.py \
        --raw-result-path "$RAW_RESULT_PATH" \
        --model-name "$MODEL_NAME" \
        --num-input-tokens "$NUM_INPUT_TOKENS" \
        --num-output-tokens "$NUM_OUTPUT_TOKENS"
else
    echo "Warning: could not locate vLLM's native result JSON in $RESULT_DIR to standardize." >&2
fi
