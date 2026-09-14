# aiperf

<!-- TOC -->

- [Overview](#overview)
- [Verification status](#verification-status)
- [Prerequisites](#prerequisites)
- [Quickstart (synthetic dataset)](#quickstart-synthetic-dataset)
- [Custom dataset](#custom-dataset)
- [Real workload](#real-workload)
- [AgentX MVP agentic-coding benchmark](#agentx-mvp-agentic-coding-benchmark)
- [Output structure vs. the Kit](#output-structure-vs-the-kit)

<!-- /TOC -->

# Overview

This folder runs NVIDIA's [aiperf](https://github.com/ai-dynamo/aiperf) CLI directly (`aiperf profile ...`) against a SambaNova endpoint. Unlike the Kit's own native evaluator in [`../kit/`](../kit/), there's no Python wrapper around the benchmark run itself — just aiperf's own CLI plus two bash quickstart scripts, which also call small Kit helper scripts for dataset generation, tokenizer resolution, and output standardization (see below).

Install with `pip install aiperf` (see the project's GitHub page for details). For the overall benchmarking suite, see [`../../README.md`](../../README.md).

# Verification status

[`quickstart.sh`](./quickstart.sh) is **live-verified** end-to-end (aiperf 0.12.0), including `.env` auto-loading, tokenizer resolution, both burst and Poisson-paced (`QPS`) runs, and output standardization. Real findings, none catchable from docs alone:
- **`--api-key` is required** — aiperf reads no env var for auth; every request 401s without it.
- **`--request-rate`/`--arrival-pattern poisson` work with a custom dataset** (`--input-file`/`--custom-dataset-type single_turn`) — confirmed live, so one command covers both burst and rate-paced runs.
- **`--request-rate inf` behaves like `--arrival-pattern concurrency-burst`** — confirmed by comparing `effective_concurrency`/`request_throughput`; aiperf itself records `rate: null` for `inf` either way.
- **`profile_export_aiperf.json`'s aggregate scalars (`benchmark_duration`, `request_throughput`, `effective_concurrency`, ...) are `{"unit": ..., "avg": ...}` dicts, not bare numbers** — caught while writing the output converter.
- **The run's own `--concurrency`/`--request-rate` are recorded verbatim** under `input_config.phases[]` — the converter reads these back directly.
- **aiperf silently disables all progress output when stdout isn't a terminal.** Without `--ui-type`, it auto-resolves to `--ui-type none` (a literal no-op UI, not degraded output) whenever stdout isn't a TTY. `quickstart.sh` run interactively is unaffected; the Streamlit app's subprocess runner pipes stdout, so it explicitly passes `--ui-type simple`. Piping/redirecting `quickstart.sh`'s own output hits the same issue — add `--ui-type simple` yourself if you need progress there.
- Nothing else needed correction — field names and custom-dataset conversion matched the docs once the API key was fixed.

The [AgentX MVP scenario](#agentx-mvp-agentic-coding-benchmark) is **partially verified**: every flag name is confirmed via `aiperf profile --help`, and two value-level bugs were caught and fixed (`--public-dataset`/`--cache-bust` use hyphens, not underscores). The `.env`/tokenizer preamble (shared with `quickstart.sh`) is live-verified in isolation, but a complete run of the scenario itself was not achieved even as a short smoke test — see that section's caveat before relying on it.

# Prerequisites

1. Install aiperf (`pip install aiperf`; requires **Python >=3.11,<3.14**).

2. Set `SAMBANOVA_API_BASE` and `SAMBANOVA_API_KEY` — see the top-level README's [Getting a SambaNova API key](../../../README.md#getting-a-sambanova-api-key-and-setting-your-generative-models) section.

   Both scripts read an already-exported shell variable first, then fall back to the repo-root `.env` (same pattern as [`../vllm/README.md`](../vllm/README.md#prerequisites)) — no `export` needed if they're already in `.env`. aiperf itself still needs the key passed explicitly via `--api-key`; it never reads env vars on its own.

# Quickstart (synthetic dataset)

[`quickstart.sh`](./quickstart.sh) covers dataset-driven runs, both closed-loop and rate-paced (everything except AgentX MVP — see [below](#agentx-mvp-agentic-coding-benchmark)).

1. Edit the **Parameters — edit these** section at the top of `quickstart.sh`:
   - `MODEL_NAME`: model name as exposed by the API.
   - `NUM_REQUESTS`, `NUM_CONCURRENT_REQUESTS`: total requests and concurrency ceiling.
   - `QPS`: `"inf"` (default) fires a closed-loop burst; a number paces as a Poisson process — see [Real workload](#real-workload).
   - `NUM_INPUT_TOKENS`, `NUM_OUTPUT_TOKENS`: prompt/generation length.
   - `ARTIFACT_DIR`: where aiperf, the generated dataset, and the standardized output all get written (`DATASET_PATH` derives from it by default).

   Everything below that (`.env` loading, `TOKENIZER_MODEL_NAME` auto-resolution via [`../kit/src/resolve_tokenizer_name.py`](../kit/src/resolve_tokenizer_name.py)) runs automatically — override `TOKENIZER_MODEL_NAME` yourself only if you need a different tokenizer.

2. Run `sh quickstart.sh`. Three steps:
   - Generates a synthetic dataset via [`../kit/src/generate_dataset.py`](../kit/src/generate_dataset.py) with `--prompt-key text` (aiperf's `single_turn` schema, vs. this repo's usual `{"prompt": ...}`).
   - Runs `aiperf profile` against it with `--request-rate "$QPS" --arrival-pattern poisson` (`QPS="inf"` = one concurrency-capped burst, live-verified equivalent to aiperf's dedicated burst mode).
   - Converts aiperf's native output into the Kit's `aiperf_individual_responses.json`/`aiperf_summary.json` — see [Output structure vs. the Kit](#output-structure-vs-the-kit).

   `--output-tokens-mean` is required since aiperf applies one output-length target uniformly, not per-row.

3. `ARTIFACT_DIR` then contains both aiperf's native files (`profile_export.jsonl`, `profile_export_aiperf.json`, plus `.csv`/console summary) and the standardized pair from Step 3.

`NUM_WARMUP_REQUESTS` (`0` disables) maps to aiperf's native `--warmup-request-count` — same semantics as the Kit's/vLLM's own warm-up. Confirmed via aiperf's source: warm-up runs through the same request/dataset pipeline as the measured run (real varied prompts, not one repeated probe like vLLM's `--num-warmups`) — closer to the Kit's warm-up behavior.

# Custom dataset

To benchmark your own prompts, skip Step 1 in `quickstart.sh` (comment it out) and point `DATASET_PATH` at a dataset already in aiperf's `{"text": "..."}` schema.

If your dataset uses the repo's standard `{"prompt": "..."}` schema (e.g. [`custom_prompt_example.jsonl`](../../prompts/custom_prompt_example.jsonl)), convert it first — regenerate via `generate_dataset.py --prompt-key text`, or with `jq`:
```bash
jq -c '{text: .prompt}' your_prompt_dataset.jsonl > aiperf_dataset.jsonl
```
or Python if you don't have `jq`:
```bash
python -c "
import json
with open('your_prompt_dataset.jsonl') as fin, open('aiperf_dataset.jsonl', 'w') as fout:
    for line in fin:
        row = json.loads(line)
        fout.write(json.dumps({'text': row['prompt']}) + '\n')
"
```
Then set `DATASET_PATH` to the converted file and `NUM_REQUESTS` to its line count.

# Real workload

`quickstart.sh`'s `QPS` switches pacing (live-verified both ways, same custom-dataset path used for burst runs):
- `QPS="inf"` (default): closed-loop — `NUM_REQUESTS` capped at `NUM_CONCURRENT_REQUESTS` in flight. Live-verified equivalent to `--arrival-pattern concurrency-burst`.
- `QPS=<number>`, e.g. `1`: open-loop — dispatched at that rate, paced as a Poisson process.

Both map to the same `aiperf profile` flags: `--request-rate "$QPS" --arrival-pattern poisson`. `--concurrency "$NUM_CONCURRENT_REQUESTS"` stays set alongside `--request-rate` even when paced, so a slow server can't cause unbounded queuing.

Edit `QPS` and run `bash quickstart.sh`; output files are the same as [Quickstart](#quickstart-synthetic-dataset).

The standardized summary's `qps`/`num_concurrent_requests` are read back from aiperf's own recorded config (`input_config.phases[].rate`/`.concurrency`), not re-derived from the script — so they always reflect what actually ran, and `qps` comes back `null` (not `Infinity`) for closed-loop runs. See [Output structure vs. the Kit](#output-structure-vs-the-kit).

# AgentX MVP agentic-coding benchmark

[`agentx_quickstart.sh`](./agentx_quickstart.sh) runs aiperf's `--scenario inferencex-agentx-mvp` preset. Same `.env`/tokenizer auto-resolution as [Quickstart](#quickstart-synthetic-dataset). No `DATASET_PATH` here — this scenario replays a public trace corpus (`--public-dataset`), not a generated file.

**What it measures**: a **duration-based replay** of a public multi-turn Claude-Code agentic-coding trace corpus (`semianalysis-cc-traces-weka-with-subagents` by default), including subagent turns, run for a fixed wall-clock duration rather than a fixed request count.

**Locked flags** — the scenario preset pins these because the benchmark's semantics depend on them:
- `--streaming` — required for TTFT/ITL timing to mean anything.
- `--extra-inputs ignore_eos:true` — forces the full recorded output length per turn.
- `--cache-bust first-turn-prefix` — busts the prefix cache each first turn so cache reuse doesn't mask true latency. **Hyphens, not underscores** (`first_turn_prefix` is rejected).
- `--system-idle-gap-cap-seconds 10` — caps injected idle gaps between turns.
- `--use-server-token-count` — a plain boolean flag; `--use-server-token-count true` errors with `Unused Tokens: ['true']`.

**Forbidden**: `--synthesis-max-isl`, `--trace-idle-gap-cap-seconds`, `--inter-turn-delay-cap-seconds`, `--fixed-schedule`, `--request-rate`, `--ignore-trace-delays` — these conflict with trace-replay semantics and are never passed.

`--public-dataset` also uses hyphens: `semianalysis-cc-traces-weka-with-subagents` (underscores rejected).

**900-second minimum duration**: confirmed live — a shorter `--benchmark-duration` triggers aiperf's own validator error (`required '>=900'`, "to reach steady state and trigger KV offloading"). `agentx_quickstart.sh` defaults to `1800`; going below 900 requires `--unsafe-override`, and the run is then **not submission-valid**.

**"Submission-valid"** means: full duration (≥900s), no `--unsafe-override`, no forbidden flags, all locked flags intact. A shortened/overridden smoke test is fine for confirming your command/credentials/endpoint work, but its numbers must never be reported as official.

**Smoke-testing — known rough edge**: the commented-out variant (`--unsafe-override --benchmark-duration 60 --num-dataset-entries 20`) converts the duration-floor violation into a warning as documented, but a short override did **not** produce a complete run here — `--benchmark-duration` 30/60/120s with `--num-dataset-entries 5` all ended in "Terminal warmup failure" (the internal warmup ramp spans ~1800s regardless of the override). Try a larger `--num-dataset-entries`, or budget closer to the real 900s+. Since no smoke test completed, the output-standardization step is only verified for its `.env`/tokenizer preamble here — check [`../kit/src/convert_aiperf_output.py`](../kit/src/convert_aiperf_output.py) against your real output if it errors.

Edit `MODEL_NAME`, `MAX_CONTEXT_LENGTH` (required), `NUM_CONCURRENT_REQUESTS`, `BENCHMARK_DURATION`, `PUBLIC_DATASET`, `ARTIFACT_DIR`, then run `bash agentx_quickstart.sh`.

**Output**: same aiperf-native files as other modes (under `--artifact-dir`, or `<artifact-dir>/aggregate/` for `--num-profile-runs > 1`), plus the standardized pair recorded with `workload_mode="agentic_coding"`.

# Output structure vs. the Kit

The Kit always writes two files: `*_individual_responses.json` (JSON array, one object per request) and `*_summary.json` (flat dict, `results_<metric>_<stat>` keys with mean/min/max/stddev/percentiles).

**aiperf's native output mirrors that same two-file split**, live-verified, just under different names/units/shape:

| | Kit | aiperf's native files |
|---|---|---|
| Per-request file | `*_individual_responses.json` (JSON array) | `profile_export.jsonl` (JSONL) |
| Per-request shape | flat dict (`client_ttft_s`, `number_input_tokens`) | nested `{"metadata": {...}, "metrics": {<name>: {"value": ..., "unit": ...}}}` |
| Aggregate file | `*_summary.json` (flat dict) | `profile_export_aiperf.json` (flat dict + `.csv`/console summary) |
| Aggregate per-metric shape | `{quantiles: {p5...p99}, mean, min, max, stddev}` | `{unit, avg, p1,p5,p10,...,p99, min, max, std, count, sum}` — more percentiles than the Kit; aggregate scalars (`benchmark_duration`, etc.) use this same shape too, not a bare number |
| Units | seconds (`_s` suffix) | milliseconds for latency — convert before comparing to Kit numbers |
| Field names | `client_ttft_s`, `client_end_to_end_latency_s`, `number_input_tokens`, `number_output_tokens` | `time_to_first_token`, `request_latency`, `input_sequence_length`, `output_sequence_length` |

**Both quickstart scripts standardize this automatically.** [`convert_aiperf_output.py`](../kit/src/convert_aiperf_output.py) reads aiperf's finished native files and writes an `aiperf_individual_responses.json`/`aiperf_summary.json` pair alongside them, reusing the Kit's own canonical models (`RequestMetric`/`BenchmarkSummary` in [`schemas.py`](../kit/src/schemas.py)) rather than a hand-rolled lookalike. Since aiperf's aggregate file already carries a full `p5`–`p99` set, the converter reads it directly rather than recomputing.

`num_concurrent_requests`/`qps` are read back from aiperf's own recorded config (`input_config.phases[]`, preferring the `'profiling'`-kind phase), falling back to an explicit CLI arg (used by `agentx_quickstart.sh`) and then to `effective_concurrency` if neither is present — this is what lets `quickstart.sh` use one command for both closed-loop and paced modes.

Fields with no Kit equivalent (`schema_version`, `aiperf_version`, `benchmark_id`) land in the summary's `extra` dict. `workload_mode` defaults to `'synthetic'`; `agentx_quickstart.sh` passes `--workload-mode agentic_coding` explicitly.

This is a **post-hoc conversion of already-written files**, not a live wrapper — aiperf's native files are untouched and coexist alongside the standardized ones. An earlier version of this integration wrapped execution live and suffered real schema-drift bugs; converting known, finished files (reusing `schemas.py` directly) keeps this version from drifting the same way.

See [`../vllm/README.md`](../vllm/README.md#output-structure-vs-the-kit) for vLLM's native shape (a single combined file with columnar arrays, less close to the Kit's own convention than aiperf's).
