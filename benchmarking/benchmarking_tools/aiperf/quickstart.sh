#!/usr/bin/env bash
# quickstart.sh
# Runs NVIDIA's aiperf CLI directly against a SambaNova endpoint -- no Python wrapper. MODE picks
# between a fixed-request-count burst against a generated dataset ("synthetic") or a rate-paced
# run using aiperf's own token generation ("real_workload"). The final step converts aiperf's
# native output into the Kit's own _individual_responses.json/_summary.json convention, written
# alongside it in the same ARTIFACT_DIR. See ./README.md for a "custom" (bring-your-own-dataset)
# variant and the AgentX MVP scenario (agentx_quickstart.sh).
# Edit the variables below, then run: sh quickstart.sh

set -euo pipefail

# ================================================================================================
# Parameters -- edit these
# ================================================================================================

MODEL_NAME="Meta-Llama-3.3-70B-Instruct"  # Model name as exposed by the endpoint

MODE="synthetic"                  # 'synthetic' (fixed-count burst) | 'real_workload' (rate-paced)

# --- Load shape ---
NUM_REQUESTS=10                   # Total number of requests to send
NUM_CONCURRENT_REQUESTS=10        # Max requests in flight at once (synthetic mode only)

# --- Prompt shape ---
NUM_INPUT_TOKENS=550              # Input tokens per request (used to generate the dataset)
NUM_OUTPUT_TOKENS=150             # Max output tokens per request

# --- real_workload-only parameters ---
QPS=0.5                           # Target requests per second (real_workload mode only)
QPS_DISTRIBUTION="constant"       # 'constant' | 'exponential' inter-arrival pacing (real_workload mode only)

# --- Paths ---
ARTIFACT_DIR="./data/aiperf_results"                     # Where aiperf writes everything, including the standardized output
DATASET_PATH="$ARTIFACT_DIR/aiperf_synthetic_dataset.jsonl"  # Generated dataset (synthetic mode only), written inside ARTIFACT_DIR

# ================================================================================================
# Everything below runs automatically -- no need to edit
# ================================================================================================

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

SAMBANOVA_API_BASE="${SAMBANOVA_API_BASE:-https://api.sambanova.ai/v1}"
# aiperf does NOT read any environment variable for auth on its own -- every request 401s
# ("You didn't provide an API key") without an explicit --api-key.
SAMBANOVA_API_KEY="${SAMBANOVA_API_KEY:?SAMBANOVA_API_KEY must be set (export it, or set it in .env) -- aiperf requires --api-key explicitly.}"

# HF tokenizer id aiperf uses locally (--tokenizer), resolved from MODEL_NAME via the Kit's own
# model registry (../kit/src/resolve_tokenizer_name.py). Override by exporting
# TOKENIZER_MODEL_NAME yourself if needed.
TOKENIZER_MODEL_NAME="${TOKENIZER_MODEL_NAME:-$(python ../kit/src/resolve_tokenizer_name.py --model-name "$MODEL_NAME")}"

mkdir -p "$ARTIFACT_DIR"

# --------------------------------------------------------------------------------------------
# synthetic mode: generate a Kit-style dataset with generate_dataset.py, then hand it to aiperf
# as a custom dataset via --input-file.
# --------------------------------------------------------------------------------------------
if [[ "$MODE" == "synthetic" ]]; then
  echo "Generating synthetic dataset with generate_dataset.py (--prompt-key text for aiperf) ..."
  python ../kit/src/generate_dataset.py \
    --model-name "$MODEL_NAME" \
    --num-requests "$NUM_REQUESTS" \
    --num-input-tokens "$NUM_INPUT_TOKENS" \
    --output-path "$DATASET_PATH" \
    --prompt-key text

  echo "Running aiperf profile (custom dataset, concurrency-burst) ..."
  aiperf profile \
    --model "$MODEL_NAME" \
    --url "$SAMBANOVA_API_BASE" \
    --api-key "$SAMBANOVA_API_KEY" \
    --artifact-dir "$ARTIFACT_DIR" \
    --endpoint-type chat \
    --streaming \
    --tokenizer "$TOKENIZER_MODEL_NAME" \
    --input-file "$DATASET_PATH" \
    --custom-dataset-type single_turn \
    --request-count "$NUM_REQUESTS" \
    --concurrency "$NUM_CONCURRENT_REQUESTS" \
    --arrival-pattern concurrency_burst \
    --output-tokens-mean "$NUM_OUTPUT_TOKENS"

# --------------------------------------------------------------------------------------------
# real_workload mode: aiperf's own synthetic token generation, rate-paced instead of
# concurrency-capped. No dataset file is generated or needed for this mode.
# --------------------------------------------------------------------------------------------
elif [[ "$MODE" == "real_workload" ]]; then
  # constant -> --arrival-pattern constant ; exponential -> --arrival-pattern poisson
  if [[ "$QPS_DISTRIBUTION" == "exponential" ]]; then
    ARRIVAL_PATTERN="poisson"
  else
    ARRIVAL_PATTERN="constant"
  fi

  echo "Running aiperf profile (rate-paced, aiperf-native synthetic tokens) ..."
  aiperf profile \
    --model "$MODEL_NAME" \
    --url "$SAMBANOVA_API_BASE" \
    --api-key "$SAMBANOVA_API_KEY" \
    --artifact-dir "$ARTIFACT_DIR" \
    --endpoint-type chat \
    --streaming \
    --tokenizer "$TOKENIZER_MODEL_NAME" \
    --synthetic-input-tokens-mean "$NUM_INPUT_TOKENS" \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "$NUM_OUTPUT_TOKENS" \
    --output-tokens-stddev 0 \
    --request-count "$NUM_REQUESTS" \
    --request-rate "$QPS" \
    --arrival-pattern "$ARRIVAL_PATTERN" \
    --concurrency 1000   # high ceiling so it never caps the rate-paced open-loop firing (placeholder)

else
  echo "Unknown MODE: $MODE (expected 'synthetic' or 'real_workload')" >&2
  exit 1
fi

# --------------------------------------------------------------------------------------------
# Standardize aiperf's own output into the Kit's aiperf_individual_responses.json/
# aiperf_summary.json convention, written alongside it in ARTIFACT_DIR (see ./README.md's
# "Output structure vs. the Kit" section for what this changes and why).
# --------------------------------------------------------------------------------------------
CONVERT_ARGS=(
    --artifact-dir "$ARTIFACT_DIR"
    --model-name "$MODEL_NAME"
    --workload-mode "$MODE"
    --num-input-tokens "$NUM_INPUT_TOKENS"
    --num-output-tokens "$NUM_OUTPUT_TOKENS"
)
if [[ "$MODE" == "synthetic" ]]; then
    CONVERT_ARGS+=(--num-concurrent-requests "$NUM_CONCURRENT_REQUESTS")
else
    CONVERT_ARGS+=(--qps "$QPS")
fi
python ../kit/src/convert_aiperf_output.py "${CONVERT_ARGS[@]}"

echo "Done. aiperf artifacts (profile_export.jsonl, profile_export_aiperf.json) are under $ARTIFACT_DIR"
