#!/usr/bin/env bash
# agentx_quickstart.sh -- runs aiperf's InferenceX AgentX MVP scenario (--scenario
# inferencex-agentx-mvp): a duration-based replay of a public agentic-coding trace corpus, not a
# fixed request-count benchmark. Partially verified -- see ./README.md's "AgentX MVP" section for
# what's confirmed, what isn't, and why several flags below are locked/forbidden by the scenario.
# Edit the variables below, then run: sh agentx_quickstart.sh

set -euo pipefail

# ================================================================================================
# Parameters -- edit these
# ================================================================================================

MODEL_NAME="MiniMax-M3"  # Model name as exposed by the endpoint (--model)

MAX_CONTEXT_LENGTH=131072       # REQUIRED -- the endpoint/model's max context length (--max-context-length)
NUM_CONCURRENT_REQUESTS=8       # --concurrency
PUBLIC_DATASET="semianalysis-cc-traces-weka-with-subagents"  # --public-dataset (the trace corpus; hyphens required)

# Benchmark duration in seconds (--benchmark-duration). Must be >= 900 (scenario's enforced
# minimum) for a submission-valid run; going lower requires --unsafe-override -- see ./README.md.
BENCHMARK_DURATION=1800

ARTIFACT_DIR="./data/aiperf_agentx_results"  # Where aiperf writes everything, including the standardized output

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

# Full AgentX MVP command. --streaming, --extra-inputs, --cache-bust, and
# --system-idle-gap-cap-seconds are LOCKED by this scenario -- required for a submission-valid
# run, not just defaults (see ./README.md for why). --trajectory-start-min/max-ratio 0.0/1.0
# replay full trajectories (scenario defaults; safe to change for a slice instead). You can add
# more of aiperf's own flags here too, as long as they don't conflict with the forbidden list below.
aiperf profile --scenario inferencex-agentx-mvp \
  --model "$MODEL_NAME" \
  --url "$SAMBANOVA_API_BASE" \
  --api-key "$SAMBANOVA_API_KEY" \
  --artifact-dir "$ARTIFACT_DIR" \
  --concurrency "$NUM_CONCURRENT_REQUESTS" \
  --public-dataset "$PUBLIC_DATASET" \
  --max-context-length "$MAX_CONTEXT_LENGTH" \
  --tokenizer "$TOKENIZER_MODEL_NAME" \
  --benchmark-duration "$BENCHMARK_DURATION" \
  --trajectory-start-min-ratio 0.0 \
  --trajectory-start-max-ratio 1.0 \
  --use-server-token-count \
  --streaming \
  --extra-inputs ignore_eos:true \
  --cache-bust first-turn-prefix \
  --system-idle-gap-cap-seconds 10

# Forbidden for this scenario (conflict with trace-based replay semantics -- see ./README.md):
# --synthesis-max-isl  --trace-idle-gap-cap-seconds  --inter-turn-delay-cap-seconds
# --fixed-schedule  --request-rate  --ignore-trace-delays

# Standardize aiperf's output into the Kit's aiperf_individual_responses.json/aiperf_summary.json.
python ../kit/src/convert_aiperf_output.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --model-name "$MODEL_NAME" \
    --workload-mode agentic_coding \
    --num-concurrent-requests "$NUM_CONCURRENT_REQUESTS"

echo "Done. aiperf artifacts (profile_export.jsonl, profile_export_aiperf.json) are under $ARTIFACT_DIR"
echo "(or under $ARTIFACT_DIR/aggregate/ if --num-profile-runs > 1 was passed)"
