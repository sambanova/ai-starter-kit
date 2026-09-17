#!/usr/bin/env bash
# quickstart.sh -- runs NVIDIA's aiperf CLI directly against a SambaNova endpoint (no Python
# wrapper): generate a dataset, run aiperf's benchmark, standardize its output into the Kit's
# convention. See ./README.md for custom datasets and the AgentX MVP scenario (agentx_quickstart.sh).
# Edit the variables below, then run: sh quickstart.sh

set -euo pipefail

# ================================================================================================
# Parameters -- edit these
# ================================================================================================

MODEL_NAME="Meta-Llama-3.3-70B-Instruct"  # Model name as exposed by the endpoint

# --- Load shape ---
NUM_REQUESTS=10                   # Total number of requests to send
NUM_CONCURRENT_REQUESTS=10        # Max requests in flight at once (concurrency ceiling)
NUM_WARMUP_REQUESTS=0             # Throwaway requests before the measured run; 0 disables
QPS="inf"                         # Requests/sec; "inf" fires a closed-loop burst, a number paces via a Poisson process

# --- Prompt shape ---
NUM_INPUT_TOKENS=550              # Input tokens per request (used to generate the dataset)
NUM_OUTPUT_TOKENS=150             # Max output tokens per request

# --- Paths ---
ARTIFACT_DIR="./data/aiperf_results"                          # Where aiperf writes everything, including the standardized output
DATASET_PATH="$ARTIFACT_DIR/aiperf_synthetic_dataset.jsonl"   # Generated dataset, written inside ARTIFACT_DIR
# To use your own prompts instead, comment out Step 1 below and point DATASET_PATH at a
# {"text": "..."} .jsonl file (this repo's other tools use {"prompt": ...} -- convert first, e.g.:
#   jq -c '{text: .prompt}' ../../prompts/custom_prompt_example.jsonl > aiperf_dataset.jsonl

# ================================================================================================
# Everything below runs automatically -- no need to edit
# ================================================================================================

source ../kit/src/load_env.sh  # loads SAMBANOVA_API_BASE/KEY from .env unless already exported

SAMBANOVA_API_BASE="${SAMBANOVA_API_BASE:-https://api.sambanova.ai/v1}"
# aiperf reads no env var for auth on its own -- every request 401s without an explicit --api-key.
SAMBANOVA_API_KEY="${SAMBANOVA_API_KEY:?SAMBANOVA_API_KEY must be set (export it, or set it in .env) -- aiperf requires --api-key explicitly.}"

# HF tokenizer id aiperf uses locally (--tokenizer), auto-resolved from MODEL_NAME. Override by
# exporting TOKENIZER_MODEL_NAME yourself.
TOKENIZER_MODEL_NAME="${TOKENIZER_MODEL_NAME:-$(python ../kit/src/resolve_tokenizer_name.py --model-name "$MODEL_NAME")}"

mkdir -p "$ARTIFACT_DIR"

# --- Step 1: generate a synthetic dataset (skip and edit DATASET_PATH above to bring your own) ---
python ../kit/src/generate_dataset.py \
    --model-name "$MODEL_NAME" \
    --num-requests "$NUM_REQUESTS" \
    --num-input-tokens "$NUM_INPUT_TOKENS" \
    --output-path "$DATASET_PATH" \
    --prompt-key text

# --- Step 2: run aiperf's own benchmark CLI against that dataset ---
AIPERF_ARGS=(
    profile
    --model "$MODEL_NAME"
    --url "$SAMBANOVA_API_BASE"
    --api-key "$SAMBANOVA_API_KEY"
    --artifact-dir "$ARTIFACT_DIR"
    --endpoint-type chat
    --streaming
    --tokenizer "$TOKENIZER_MODEL_NAME"
    --input-file "$DATASET_PATH"
    --custom-dataset-type single_turn
    --request-count "$NUM_REQUESTS"
    --concurrency "$NUM_CONCURRENT_REQUESTS"
    --request-rate "$QPS"
    --arrival-pattern poisson
    --output-tokens-mean "$NUM_OUTPUT_TOKENS"
)
if [ "$NUM_WARMUP_REQUESTS" -gt 0 ]; then
    AIPERF_ARGS+=(--warmup-request-count "$NUM_WARMUP_REQUESTS")
fi

# Need a flag not listed above? Add it directly to AIPERF_ARGS -- this repo doesn't wrap or
# validate aiperf's CLI, so any native flag works (see `aiperf profile --help`).
aiperf "${AIPERF_ARGS[@]}"

# --- Step 3: standardize aiperf's output into the Kit's aiperf_individual_responses.json/
# aiperf_summary.json (the --concurrency/--request-rate aiperf actually ran with are read back
# from its own aggregate file, so they aren't passed again here) ---
python ../kit/src/convert_aiperf_output.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --model-name "$MODEL_NAME" \
    --num-input-tokens "$NUM_INPUT_TOKENS" \
    --num-output-tokens "$NUM_OUTPUT_TOKENS"

echo "Done. aiperf artifacts (profile_export.jsonl, profile_export_aiperf.json) are under $ARTIFACT_DIR"
