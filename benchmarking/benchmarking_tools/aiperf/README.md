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

This folder runs NVIDIA's [aiperf](https://github.com/ai-dynamo/aiperf) CLI directly (`aiperf profile ...`) against a SambaNova endpoint. Unlike the Kit's own native evaluator in [`../kit/`](../kit/) (Python, unaffected by anything in this folder), aiperf is driven here purely through its own CLI plus the two bash quickstart scripts and this README -- there's no Python wrapper around the benchmark run itself, though both scripts do call a couple of small Kit helper scripts for dataset generation, tokenizer resolution, and output standardization (see below).

Install it with `pip install aiperf` (see the project's GitHub page for full details). For general information on this repo's benchmarking suite, see the top-level [`../../README.md`](../../README.md).

# Verification status

The `synthetic`/`custom`/`real_workload` modes in [`quickstart.sh`](./quickstart.sh) are **live-verified**: run end-to-end against a real SambaNova endpoint with aiperf 0.12.0 installed, including the `.env` auto-loading, tokenizer auto-resolution, and output-standardization step described below. Real bugs were found and fixed this way (neither was catchable from docs alone):
- **`--api-key` is required.** aiperf does not read any environment variable for auth on its own -- every request 401s ("You didn't provide an API key") without an explicit `--api-key "$SAMBANOVA_API_KEY"`.
- **`profile_export_aiperf.json`'s aggregate-level scalar metrics (`benchmark_duration`, `request_throughput`, `effective_concurrency`, ...) are all `{"unit": ..., "avg": ...}` dicts, not bare numbers** -- caught while writing the output-standardization converter (a bare-float assumption raised a validation error against real output).
- **`effective_concurrency` is a float, not an int** -- when `real_workload` mode falls back to it (since that mode has no fixed `--concurrency` to report), it needs rounding before it fits the Kit's `num_concurrent_requests` field.
- Nothing else needed correction for these three modes -- the field names, custom-dataset conversion, and rate-pacing flags all worked as documented on the first live run once the API key was fixed.

The [AgentX MVP scenario](#agentx-mvp-agentic-coding-benchmark) in [`agentx_quickstart.sh`](./agentx_quickstart.sh) is **partially verified**: every flag name is confirmed real via `aiperf profile --help`, and two value-level bugs were caught and fixed (`--public-dataset`/`--cache-bust` choices use hyphens, not underscores -- see that section for details). The `.env` auto-loading and tokenizer auto-resolution preamble (shared verbatim with `quickstart.sh`) is live-verified in isolation. However, a complete successful run of the scenario itself was not achieved even as a short `--unsafe-override` smoke test in this environment -- see the caveat in that section before relying on it.

# Prerequisites

1. Install aiperf:
   ```bash
   pip install aiperf
   ```
   aiperf requires **Python >=3.11,<3.14**.

2. Set `SAMBANOVA_API_BASE` and `SAMBANOVA_API_KEY`. Follow the instructions in the top-level repo README's [Getting a SambaNova API key and setting your generative models](../../../README.md#getting-a-sambanova-api-key-and-setting-your-generative-models) section.

   Both scripts read these from an already-exported shell variable first, then fall back to the repo-root `.env` file automatically (see [`../vllm/README.md`](../vllm/README.md#prerequisites) for the same pattern) -- so you don't need to `export` anything if they're already set in `.env`. Either way, aiperf itself needs the key passed explicitly via `--api-key`; it does not read any environment variable on its own.

# Quickstart (synthetic dataset)

[`quickstart.sh`](./quickstart.sh) covers the plain dataset/rate-driven modes (everything except the AgentX MVP scenario -- see [below](#agentx-mvp-agentic-coding-benchmark) for that).

1. Open `quickstart.sh` and edit the variables in the **Parameters -- edit these** section at the top:
   - `MODEL_NAME`: the model name as exposed by the API.
   - `MODE`: `"synthetic"` (fixed-count burst against a generated dataset) or `"real_workload"` (rate-paced, using aiperf's own synthetic token generation).
   - `NUM_REQUESTS`, `NUM_CONCURRENT_REQUESTS`: total requests and how many are allowed in flight at once (synthetic mode).
   - `NUM_INPUT_TOKENS`, `NUM_OUTPUT_TOKENS`: prompt/generation length.
   - `QPS`, `QPS_DISTRIBUTION`: target rate and inter-arrival pacing (real_workload mode only).
   - `ARTIFACT_DIR`: where aiperf (and the generated dataset, and the standardized output -- see below) all get written. `DATASET_PATH` is derived from it by default (`$ARTIFACT_DIR/aiperf_synthetic_dataset.jsonl`), so changing `ARTIFACT_DIR` alone moves everything together.

   Everything under **Everything below runs automatically** -- `.env` loading and `TOKENIZER_MODEL_NAME` -- normally doesn't need editing. `TOKENIZER_MODEL_NAME` is auto-resolved from `MODEL_NAME` via the Kit's own model registry ([`../kit/src/resolve_tokenizer_name.py`](../kit/src/resolve_tokenizer_name.py), wrapping `get_tokenizer_model_name` in [`../../benchmarking_utils.py`](../../benchmarking_utils.py)) -- override by exporting `TOKENIZER_MODEL_NAME` yourself before running the script if you ever need a different tokenizer.

2. Run it:
   ```bash
   sh quickstart.sh
   ```

   With `MODE="synthetic"`, this:
   - Generates a synthetic prompt dataset with [`../kit/src/generate_dataset.py`](../kit/src/generate_dataset.py), passing `--prompt-key text` so each line is written as `{"text": "..."}` -- the schema aiperf's `single_turn` custom-dataset-type expects (the repo's other tools use `{"prompt": ...}` instead).
   - Runs `aiperf profile` against that file as a custom dataset, firing all requests as one concurrency-capped burst.
   - Converts aiperf's own native output into the Kit's `aiperf_individual_responses.json`/`aiperf_summary.json` convention, written alongside it in `ARTIFACT_DIR` -- see [Output structure vs. the Kit](#output-structure-vs-the-kit) below.

   `--output-tokens-mean` is required because aiperf can't read a per-row target output length from the custom dataset file -- it applies one value uniformly to every request.

3. Check the output. `ARTIFACT_DIR` now contains **both**:
   - aiperf's own native output files -- `profile_export.jsonl` (per-request metrics) and `profile_export_aiperf.json` (aggregate statistics), plus a `.csv` and a console-text summary.
   - The standardized `aiperf_individual_responses.json` / `aiperf_summary.json` pair, generated automatically by the conversion step.

**Note**: this quickstart does not expose a warm-up-requests option, but aiperf does have a native equivalent -- confirmed via `--help`: `--warmup-request-count` (alias `--num-warmup-requests`) and/or `--warmup-duration`. Add either flag directly to the `aiperf profile` invocation in `quickstart.sh` if you want a warm-up phase.

# Custom dataset

To benchmark your own prompts instead of a Kit-generated synthetic set, skip the dataset-generation step in `quickstart.sh` (comment it out) and point `DATASET_PATH` directly at a dataset already in aiperf's `{"text": "..."}` schema.

If your dataset is already in the repo's standard `{"prompt": "..."}` schema (e.g. [`../../prompts/custom_prompt_example.jsonl`](../../prompts/custom_prompt_example.jsonl)), convert it first. Either regenerate it through `generate_dataset.py --prompt-key text` if it was Kit-generated, or convert an existing file with `jq`:
```bash
jq -c '{text: .prompt}' your_prompt_dataset.jsonl > aiperf_dataset.jsonl
```
or a one-liner in Python if you don't have `jq` handy:
```bash
python -c "
import json
with open('your_prompt_dataset.jsonl') as fin, open('aiperf_dataset.jsonl', 'w') as fout:
    for line in fin:
        row = json.loads(line)
        fout.write(json.dumps({'text': row['prompt']}) + '\n')
"
```
Then set `DATASET_PATH` in `quickstart.sh` to the converted file and set `NUM_REQUESTS` to the number of lines in your file.

# Real workload

`quickstart.sh`'s `MODE="real_workload"` variant paces requests over time by rate instead of firing a fixed-concurrency burst. It uses aiperf's own synthetic token generation directly -- no dataset file is generated or needed for this mode:
```bash
aiperf profile \
  --model <MODEL_NAME> --url <SAMBANOVA_API_BASE> --api-key <SAMBANOVA_API_KEY> \
  --artifact-dir <ARTIFACT_DIR> --endpoint-type chat --streaming --tokenizer <TOKENIZER_MODEL_NAME> \
  --synthetic-input-tokens-mean <N> --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean <N> --output-tokens-stddev 0 \
  --request-count <N> --request-rate <QPS> \
  --arrival-pattern constant|poisson \
  --concurrency 1000
```
- `--request-rate` sets the target queries-per-second (`QPS` in the script).
- `--arrival-pattern` accepts `constant` or `poisson`: set `QPS_DISTRIBUTION="constant"` in the script for evenly-spaced arrivals, or `"exponential"` for a Poisson process (mapped internally to `poisson`).
- `--concurrency 1000` is a high ceiling so the concurrency limit never caps the rate-paced, open-loop request firing -- a best-effort placeholder, not a tuned value.

Edit `MODE`, `QPS`, and `QPS_DISTRIBUTION` at the top of `quickstart.sh`, then run `bash quickstart.sh`. Output files are the same aiperf-native `profile_export.jsonl` / `profile_export_aiperf.json`, plus the standardized `aiperf_individual_responses.json`/`aiperf_summary.json` pair, described above.

Since real_workload mode has no fixed `--concurrency` to report, the standardized summary's `num_concurrent_requests` falls back to aiperf's own reported `effective_concurrency` for the run (rounded to the nearest int) instead of the configured value.

# AgentX MVP agentic-coding benchmark

[`agentx_quickstart.sh`](./agentx_quickstart.sh) runs aiperf's `--scenario inferencex-agentx-mvp` preset (NVIDIA's InferenceX AgentX MVP benchmark). Like `quickstart.sh`, it auto-loads `SAMBANOVA_API_BASE`/`SAMBANOVA_API_KEY` from `.env` if not already exported, and auto-resolves `TOKENIZER_MODEL_NAME` from `MODEL_NAME` via the same Kit model registry -- see [Quickstart](#quickstart-synthetic-dataset) above for details on both. There's no `DATASET_PATH` here since this scenario replays a public trace corpus (`--public-dataset`), not a generated file.

**What it measures**: unlike every mode above, this is not a token-count/request-count benchmark. It's a **duration-based replay** of a public multi-turn Claude-Code agentic-coding trace corpus (`semianalysis-cc-traces-weka-with-subagents` by default) -- real recorded multi-turn coding sessions (including subagent turns), replayed against your endpoint for a fixed wall-clock duration rather than a fixed request count.

**Why the flags are locked**: the scenario preset pins several flags to fixed values because the benchmark's semantics depend on them -- they aren't just defaults you can casually change:
- `--streaming` -- token-by-token streaming responses are required for the trace replay's TTFT/ITL timing to mean anything.
- `--extra-inputs ignore_eos:true` -- forces the model to generate the full recorded output length per turn instead of stopping early.
- `--cache-bust first-turn-prefix` -- busts the prefix cache on each first turn so cache reuse doesn't mask true first-turn latency. **(hyphens, not underscores** -- `aiperf profile --help` confirms `first-turn-prefix` is the valid choice; an earlier draft of this doc had `first_turn_prefix`, which is not accepted.)
- `--system-idle-gap-cap-seconds 10` -- caps the idle gaps the scenario injects between turns.
- `--use-server-token-count` -- a plain boolean flag (no value). Passing `--use-server-token-count true` errors with `Unused Tokens: ['true']`.

The scenario also **forbids** `--synthesis-max-isl`, `--trace-idle-gap-cap-seconds`, `--inter-turn-delay-cap-seconds`, `--fixed-schedule`, `--request-rate`, and `--ignore-trace-delays` -- these conflict with trace-based replay semantics (e.g. `--request-rate` implies a synthetic arrival process, which contradicts replaying the trace's own recorded timings) and are never passed by the script.

Also note **`--public-dataset`'s value uses hyphens**: `semianalysis-cc-traces-weka-with-subagents`, confirmed against `aiperf profile --help`'s listed choices (an earlier draft had underscores, which is rejected).

**The 900-second minimum duration**: this scenario has a hard floor of `min_benchmark_duration=900` seconds (15 minutes) -- **confirmed live**: running with a shorter duration prints aiperf's own validator warning verbatim: `--benchmark-duration: got <N>, required '>=900' (scenario 'inferencex-agentx-mvp' requires duration >= 900s to reach steady state and trigger KV offloading)`. `agentx_quickstart.sh` defaults `BENCHMARK_DURATION` to `1800` (30 minutes), but you can lower it as long as it stays >= 900. Going below 900s requires passing `--unsafe-override`, and the resulting run is **not submission-valid** -- see the smoke-test note below.

**What "submission-valid" means here**: a run only counts as an official AgentX MVP result if it satisfies the scenario's real requirements -- full-length duration (>= 900s), no `--unsafe-override`, none of the forbidden flags, and all of the locked flags intact. A shortened/overridden smoke test is useful for confirming your command, credentials, and endpoint all work, but its numbers must never be reported or compared as if they were an official result.

**How to smoke-test safely -- known rough edge**: `agentx_quickstart.sh` includes a commented-out smoke-test variant using:
```bash
--unsafe-override --benchmark-duration 60 --num-dataset-entries 20
```
`--unsafe-override` itself works exactly as documented (confirmed live: it converts the duration-floor violation into a non-blocking warning). However, **a short override did not produce a complete successful run** when tried here (`--benchmark-duration` values of 30s, 60s, and 120s combined with `--num-dataset-entries 5` all ended in "Terminal warmup failure" / "No profile results to export"). The run logs showed the scenario's internal agentic-replay warmup ramp spanning ~1800s (30 min) regardless of the `--benchmark-duration` override -- it appears calibrated to the full recorded trace spread, not to the shortened duration. If you hit the same thing, try a larger `--num-dataset-entries` (so trajectories with shorter recorded spans are available) or budget for something closer to the real 900s+ duration rather than expecting a fast dry run to complete cleanly. If you find a combination that reliably smoke-tests in under a few minutes, please update this note. Since the smoke-test variant is not known to complete successfully, the output-standardization step at the end of `agentx_quickstart.sh` has only been verified for its `.env`/tokenizer-resolution preamble, not against a real `profile_export.jsonl`/`profile_export_aiperf.json` pair from this scenario -- if it errors on your real run's output, check `../kit/src/convert_aiperf_output.py` against the actual files under `ARTIFACT_DIR` first.

Edit `MODEL_NAME`, `MAX_CONTEXT_LENGTH` (required, no default), `NUM_CONCURRENT_REQUESTS`, `BENCHMARK_DURATION`, `PUBLIC_DATASET`, and `ARTIFACT_DIR` at the top of `agentx_quickstart.sh`, then run:
```bash
bash agentx_quickstart.sh
```

**Output**: same aiperf-native files as the other modes -- `profile_export.jsonl` and `profile_export_aiperf.json` under `--artifact-dir` (or `<artifact-dir>/aggregate/` if you pass `--num-profile-runs > 1`) -- plus the standardized `aiperf_individual_responses.json`/`aiperf_summary.json` pair, recorded with `workload_mode="agentic_coding"`.

# Output structure vs. the Kit

The Kit's own evaluator (`../kit/`) always writes exactly two files per run: `*_individual_responses.json` (a JSON array, one object per request) and `*_summary.json` (a flat dict of aggregate stats, `results_<metric>_<stat>` keys with `mean`/`min`/`max`/`stddev`/percentile quantiles per metric).

**aiperf's native output (live-verified) closely mirrors that same two-file, per-request-vs-aggregate split** -- just under different names, units, and field names:

| | Kit | aiperf's native files |
|---|---|---|
| Per-request file | `*_individual_responses.json` (JSON array of objects) | `profile_export.jsonl` (JSONL, one object per line) |
| Per-request shape | flat dict, e.g. `client_ttft_s`, `number_input_tokens` | nested `{"metadata": {...}, "metrics": {<name>: {"value": ..., "unit": ...}}}` |
| Aggregate file | `*_summary.json` (flat dict) | `profile_export_aiperf.json` (flat dict, plus a `.csv` and a console-text summary) |
| Aggregate per-metric shape | `{quantiles: {p5,p25,p50,p75,p90,p95,p99}, mean, min, max, stddev}` | `{unit, avg, p1,p5,p10,p25,p50,p75,p90,p95,p99, min, max, std, count, sum}` -- aiperf reports *more* percentiles (p1, p10) than the Kit. **Aggregate-level scalar metrics** (`benchmark_duration`, `request_throughput`, `effective_concurrency`) use this same `{unit, avg, ...}` shape too, not a bare number -- confirmed live. |
| Units | seconds (`_s` suffix) | milliseconds (`ms`) for latency metrics -- convert before comparing directly against Kit numbers |
| Field name examples | `client_ttft_s`, `client_end_to_end_latency_s`, `number_input_tokens`, `number_output_tokens` | `time_to_first_token`, `request_latency`, `input_sequence_length`, `output_sequence_length` |

**Both quickstart scripts standardize this automatically**, as their final step. [`../kit/src/convert_aiperf_output.py`](../kit/src/convert_aiperf_output.py) reads aiperf's already-finished native files and writes an `aiperf_individual_responses.json` / `aiperf_summary.json` pair alongside them in `ARTIFACT_DIR`, in genuinely Kit-equivalent shape -- reusing the exact same canonical models (`RequestMetric`/`BenchmarkSummary` in [`../kit/src/schemas.py`](../kit/src/schemas.py)) and `to_legacy_flat_dict()` flattening the Kit's own evaluator uses, not a hand-rolled lookalike. Since aiperf's aggregate file already carries a full `p5`-`p99` percentile set per metric, the converter reads those percentiles directly rather than recomputing them -- only falling back to computing its own stats from the per-request file for a metric the aggregate happens to omit.

This is a **post-hoc conversion of already-written files**, not a live wrapper around the benchmark run itself -- aiperf's own native files are left untouched (all files coexist in `ARTIFACT_DIR`), and nothing about how `aiperf profile` executes is intercepted or duplicated. That distinction matters: an earlier version of this integration *did* wrap execution live and suffered real schema-drift bugs from doing so (mismatched kwargs, a summary that quietly ended up thinner than the Kit's own); converting known, finished files after the fact is a much smaller surface, and reusing `schemas.py`'s models directly (rather than re-deriving the mapping) is what keeps this version from drifting the same way.

A couple of fields don't survive the conversion cleanly, and are kept in the summary's `extra` dict rather than forced into a Kit field that doesn't really fit: `schema_version`, `aiperf_version`, and `benchmark_id` are aiperf-native concepts with no Kit equivalent. `workload_mode` is passed explicitly by each script -- `'synthetic'`/`'real_workload'` by `quickstart.sh` (matching its `MODE` variable) and `'agentic_coding'` by `agentx_quickstart.sh` -- rather than defaulted, since aiperf's own output never states which kind of run produced it.

See [`../vllm/README.md`](../vllm/README.md#output-structure-vs-the-kit) for how vLLM's native output differs (a single combined file with columnar per-request arrays, rather than aiperf's already-close-to-Kit two-file split) and how its own conversion step works.
