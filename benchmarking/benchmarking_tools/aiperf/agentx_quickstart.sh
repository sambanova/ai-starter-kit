#!/usr/bin/env bash
# Quickstart for NVIDIA aiperf's InferenceX AgentX MVP scenario (--scenario inferencex-agentx-mvp),
# run directly against a SambaNova endpoint.
#
# This is NOT a token-count/request-count benchmark. It's a duration-based replay of a public
# multi-turn Claude-Code agentic-coding trace corpus (semianalysis-cc-traces-weka-with-subagents),
# pacing requests according to the recorded trace timings/turns rather than a fixed concurrency
# or QPS. The final step converts aiperf's native output into the Kit's own
# _individual_responses.json/_summary.json convention (../kit/src/convert_aiperf_output.py),
# written alongside it in the same ARTIFACT_DIR.
#
# PARTIALLY VERIFIED (aiperf 0.12.0, `aiperf profile --help` plus live runs against a real
# SambaNova endpoint): every flag name below is confirmed real, and several concrete errors from
# the original DOCS-ONLY draft were caught this way and fixed here:
#   - Requires --api-key explicitly (aiperf reads no env var for auth on its own) -- every
#     request 401s without it.
#   - --public-dataset's valid choices use HYPHENS: semianalysis-cc-traces-weka-with-subagents
#     (the draft had underscores, which --help confirms is not a valid choice).
#   - --cache-bust's valid choices also use HYPHENS: first-turn-prefix (not first_turn_prefix).
#   - --use-server-token-count / --apply-chat-template are plain boolean flags (no `true`/`false`
#     value) -- passing a value causes an "Unused Tokens" error. --apply-chat-template is simply
#     omitted below since its default (False) is already what this scenario wants.
#   - The 900s minimum-duration floor IS enforced by aiperf itself (confirmed via the exact
#     runtime warning: "requires duration >= 900s to reach steady state and trigger KV
#     offloading"), and --unsafe-override does convert it to a non-blocking warning as documented.
#
# STILL NOT FULLY VERIFIED: a complete, successful end-to-end run of this scenario was not
# achieved in the environment that wrote this script. Short --unsafe-override smoke tests
# (--benchmark-duration 30-120s with --num-dataset-entries 5) consistently failed with "Terminal
# warmup failure" / "No profile results to export" -- the scenario's internal agentic-replay
# warmup ramp appears calibrated to the full recorded trace spread (logs showed a ~1804s/30min
# ramp) regardless of --benchmark-duration, so a short override may not be enough to reach a
# successful request. Budget for something closer to the real 900s+ duration (and consider a
# larger --num-dataset-entries) to get a genuine successful run, rather than expecting a fast
# dry run to complete cleanly.
#
# See ../aiperf/README.md for full documentation.
# Edit the variables below, then run: sh agentx_quickstart.sh

set -euo pipefail

# ================================================================================================
# Parameters -- edit these
# ================================================================================================

MODEL_NAME="Meta-Llama-3.3-70B-Instruct"  # Model name as exposed by the endpoint (--model)

MAX_CONTEXT_LENGTH=131072       # REQUIRED -- the endpoint/model's max context length (--max-context-length)
NUM_CONCURRENT_REQUESTS=8       # --concurrency (default 8 per prior research)
PUBLIC_DATASET="semianalysis-cc-traces-weka-with-subagents"  # --public-dataset (the trace corpus)

# Duration of the benchmark in seconds (--benchmark-duration). Default 1800s (30 min).
# MUST be >= 900s (the scenario's min_benchmark_duration floor) for a submission-valid run.
# Going below 900s requires --unsafe-override (see smoke-test variant below) and produces a run
# that is explicitly NOT submission-valid -- use it only to sanity-check your setup quickly.
BENCHMARK_DURATION=1800

ARTIFACT_DIR="./data/aiperf_agentx_results"  # Where aiperf writes everything, including the standardized output

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
# Full AgentX MVP command
# --------------------------------------------------------------------------------------------

# --trajectory-start-min-ratio / --trajectory-start-max-ratio: scenario DEFAULTS (0.0 / 1.0) --
# they can be overridden if you want to replay only a slice of each trajectory, but leave them as
# 0.0/1.0 to replay full trajectories.
#
# The four flags below (--streaming, --extra-inputs, --cache-bust,
# --system-idle-gap-cap-seconds) are LOCKED by the inferencex-agentx-mvp scenario itself -- they
# are always set to exactly these values and cannot be changed or omitted for a submission-valid
# run:
#   --streaming                       forces token-by-token streaming responses (required by the
#                                      agentic trace replay -- disabling it breaks TTFT/ITL timing)
#   --extra-inputs ignore_eos:true      forces the model to generate the full recorded output
#                                       length per turn instead of stopping early at EOS
#   --cache-bust first-turn-prefix     busts the prefix cache on each first turn so prefix-cache
#                                       reuse doesn't mask true first-turn latency
#   --system-idle-gap-cap-seconds 10   caps the idle gaps the scenario injects between turns to 10s
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

# Forbidden flags for this scenario (never pass these -- they conflict with trace-based replay
# semantics and would be rejected or silently break the run):
#   --synthesis-max-isl  --trace-idle-gap-cap-seconds  --inter-turn-delay-cap-seconds
#   --fixed-schedule  --request-rate  --ignore-trace-delays

# --------------------------------------------------------------------------------------------
# Smoke-test variant (commented out) -- FAST DRY RUN ONLY, NOT SUBMISSION-VALID.
#
# --unsafe-override bypasses the 900s minimum-duration floor; --benchmark-duration 60 and
# --num-dataset-entries 20 shorten the run drastically so you can sanity-check that the command,
# credentials, and endpoint all work before committing to a full-length run. A run using this
# variant does NOT satisfy the scenario's real requirements and must never be reported/submitted
# as an official AgentX MVP result.
# --------------------------------------------------------------------------------------------
#
# aiperf profile --scenario inferencex-agentx-mvp \
#   --model "$MODEL_NAME" \
#   --url "$SAMBANOVA_API_BASE" \
#   --api-key "$SAMBANOVA_API_KEY" \
#   --artifact-dir "$ARTIFACT_DIR" \
#   --concurrency "$NUM_CONCURRENT_REQUESTS" \
#   --public-dataset "$PUBLIC_DATASET" \
#   --max-context-length "$MAX_CONTEXT_LENGTH" \
#   --tokenizer "$TOKENIZER_MODEL_NAME" \
#   --unsafe-override \
#   --benchmark-duration 60 \
#   --num-dataset-entries 20 \
#   --trajectory-start-min-ratio 0.0 \
#   --trajectory-start-max-ratio 1.0 \
#   --use-server-token-count \
#   --streaming \
#   --extra-inputs ignore_eos:true \
#   --cache-bust first-turn-prefix \
#   --system-idle-gap-cap-seconds 10

# --------------------------------------------------------------------------------------------
# Standardize aiperf's own output into the Kit's aiperf_individual_responses.json/
# aiperf_summary.json convention, written alongside it in ARTIFACT_DIR (see ./README.md's
# "Output structure vs. the Kit" section for what this changes and why).
# --------------------------------------------------------------------------------------------
python ../kit/src/convert_aiperf_output.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --model-name "$MODEL_NAME" \
    --workload-mode agentic_coding \
    --num-concurrent-requests "$NUM_CONCURRENT_REQUESTS"

echo "Done. aiperf artifacts (profile_export.jsonl, profile_export_aiperf.json) are under $ARTIFACT_DIR"
echo "(or under $ARTIFACT_DIR/aggregate/ if --num-profile-runs > 1 was passed)"
