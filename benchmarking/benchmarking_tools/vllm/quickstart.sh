#!/usr/bin/env bash
# quickstart.sh
# Runs vLLM's own `vllm bench serve` CLI directly against a SambaNova endpoint -- no Python
# wrapper. Step 1 generates a synthetic prompt dataset via ../kit/src/generate_dataset.py; step 2
# runs vLLM's native benchmark against it (closed-loop, NUM_CONCURRENT_REQUESTS burst); step 3
# converts vLLM's native output into the Kit's own _individual_responses.json/_summary.json
# convention (../kit/src/convert_vllm_output.py), written alongside it in the same RESULT_DIR.
# See ./README.md for: your own dataset, rate-paced (QPS) runs, and the full vLLM flag reference.
# Edit the variables below, then run: sh quickstart.sh

set -euo pipefail

# ================================================================================================
# Parameters -- edit these
# ================================================================================================

# --- Model ---
MODEL_NAME="Meta-Llama-3.3-70B-Instruct"  # Actual API model name (--served_model_name)

# --- Load shape ---
NUM_REQUESTS=16                  # Total number of requests to send
NUM_CONCURRENT_REQUESTS=16       # Max requests in flight at once (closed-loop concurrency)
NUM_WARMUP_REQUESTS=0            # Throwaway requests before the measured run; 0 disables

# --- Prompt shape (synthetic dataset only) ---
NUM_INPUT_TOKENS=1000            # Exact input token count per generated prompt
NUM_OUTPUT_TOKENS=1000           # Max output tokens per request

# --- Paths ---
RESULT_DIR="./data/results/vllm"                            # Where vLLM writes its own result files
DATASET_PATH="$RESULT_DIR/vllm_synthetic_dataset.jsonl"     # Generated dataset, written inside RESULT_DIR

# ================================================================================================
# Everything below runs automatically -- no need to edit
# ================================================================================================

ulimit -n 4096  # raise open-file-descriptor limit for higher-concurrency runs

# --- Load repo-root .env, if present, so SAMBANOVA_API_BASE/SAMBANOVA_API_KEY don't need to be
# exported manually every time. Precedence: an already-exported shell env var wins over .env,
# which wins over the hardcoded default below. Uses python-dotenv (already a dependency) rather
# than `source .env` since this repo's .env has "KEY = value" spacing that plain bash can't parse.
REPO_ROOT="$(cd ../../.. && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then
    eval "$(python3 -c '
import shlex
from dotenv import dotenv_values
values = dotenv_values("'"$REPO_ROOT"'/.env")
for key in ("SAMBANOVA_API_BASE", "SAMBANOVA_API_KEY"):
    val = values.get(key) or ""
    print("DOTENV_" + key + "=" + shlex.quote(val))
')"
fi
: "${SAMBANOVA_API_BASE:=${DOTENV_SAMBANOVA_API_BASE:-}}"
: "${SAMBANOVA_API_KEY:=${DOTENV_SAMBANOVA_API_KEY:-}}"

# --- Endpoint ---
SAMBANOVA_API_BASE="${SAMBANOVA_API_BASE:-https://api.sambanova.ai/v1}"
# vLLM concatenates base-url + endpoint with no separator, so this must end in "/" or the request
# URL is malformed (confirmed: it silently retries for 600s instead of erroring) -- enforced below.
case "$SAMBANOVA_API_BASE" in
    */) ;;
    *) SAMBANOVA_API_BASE="${SAMBANOVA_API_BASE}/" ;;
esac
export OPENAI_API_KEY="${SAMBANOVA_API_KEY:-}"  # vLLM reads its API key from this env var, not SAMBANOVA_API_KEY

# --- Tokenizer ---
# HF tokenizer id vLLM uses locally (--model), resolved from MODEL_NAME via the Kit's own model
# registry (../kit/src/resolve_tokenizer_name.py) so it's never hand-typed or out of sync with
# what the Kit itself would use. Override by exporting TOKENIZER_MODEL_NAME yourself if needed.
TOKENIZER_MODEL_NAME="${TOKENIZER_MODEL_NAME:-$(python ../kit/src/resolve_tokenizer_name.py --model-name "$MODEL_NAME")}"

# --------------------------------------------------------------------------------------------
# Step 1: generate a synthetic dataset. To bring your own instead, comment this out and point
# DATASET_PATH above at your own .jsonl file (see ./README.md's "Custom dataset" section).
# --------------------------------------------------------------------------------------------
python ../kit/src/generate_dataset.py \
    --model-name "$MODEL_NAME" \
    --num-requests "$NUM_REQUESTS" \
    --num-input-tokens "$NUM_INPUT_TOKENS" \
    --output-path "$DATASET_PATH"

# --------------------------------------------------------------------------------------------
# Step 2: run vLLM's own benchmark CLI against that dataset. For a rate-paced (QPS) run instead
# of a concurrency burst, see ./README.md's "Real workload (rate-paced)" section.
# --------------------------------------------------------------------------------------------
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
    --save-detailed
    --save-result
    --result-dir "$RESULT_DIR"
)
if [ "$NUM_WARMUP_REQUESTS" -gt 0 ]; then
    VLLM_ARGS+=(--num-warmups "$NUM_WARMUP_REQUESTS")
fi

vllm "${VLLM_ARGS[@]}"

# --------------------------------------------------------------------------------------------
# Step 3: standardize vLLM's own output into the Kit's _individual_responses.json/_summary.json
# convention, written alongside it in RESULT_DIR (see ./README.md's "Output structure vs. the
# Kit" section for what this changes and why).
# --------------------------------------------------------------------------------------------
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
