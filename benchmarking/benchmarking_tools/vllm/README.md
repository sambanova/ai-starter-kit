# vLLM benchmarking

This directory runs vLLM's own [`vllm bench serve`](https://docs.vllm.ai/) CLI **directly** against a SambaNova endpoint. There is no Python wrapper here: this repo is used only to generate a synthetic prompt dataset (via [`../kit/src/generate_dataset.py`](../kit/src/generate_dataset.py)), and the benchmark itself is vLLM's native, unmodified command.

This complements the Kit's own native evaluator in [`../kit/`](../kit/) -- see [`../../README.md`](../../README.md) for the full top-level benchmarking overview, including the Kit's evaluator and the shared `custom`/`synthetic`/`real_workload` workload modes.

**Verification status**: `quickstart.sh` is live-verified end-to-end against a real SambaNova endpoint (vLLM 0.11.2). One real bug was found and fixed this way: `SAMBANOVA_API_BASE` needs a trailing slash. vLLM builds its request URL by naive string concatenation of `--base-url` + `--endpoint` with no separator -- without the trailing slash, `"https://host/v1"` + `"chat/completions"` becomes the malformed `"https://host/v1chat/completions"` (confirmed: this 405s, and vLLM's own endpoint-readiness check treats that as "not up yet" and retries silently for up to 600s instead of surfacing the real error). `quickstart.sh` now normalizes this automatically.

---

## Prerequisites

1. Install vLLM:

   ```bash
   pip install vllm
   ```

   _Note: for API benchmarking against a remote endpoint like SambaNova Cloud, no local GPU or local model download is required -- vLLM's benchmark client just drives HTTP requests against the endpoint you configure below._

2. Set `SAMBANOVA_API_BASE` and `SAMBANOVA_API_KEY`. Follow the instructions in the top-level repo README's [Getting a SambaNova API key and setting your generative models](../../../README.md#getting-a-sambanova-api-key-and-setting-your-generative-models) section.

   vLLM's OpenAI-compatible client reads its API key from the `OPENAI_API_KEY` environment variable, not `SAMBANOVA_API_KEY` -- `quickstart.sh` maps one onto the other for you (`export OPENAI_API_KEY="$SAMBANOVA_API_KEY"`), so you only need to export `SAMBANOVA_API_KEY` yourself.

---

## Quickstart (synthetic dataset)

1. Open [`quickstart.sh`](./quickstart.sh) and edit the variables in the "Parameters" section at the top:
   - `MODEL_NAME`: the model name as exposed by the API (e.g. `Meta-Llama-3.3-70B-Instruct`). Sent to vLLM as `--served_model_name`.
   - `NUM_REQUESTS`, `NUM_CONCURRENT_REQUESTS`: total requests and how many are allowed in flight at once.
   - `NUM_WARMUP_REQUESTS`: throwaway requests sent (and excluded from reported metrics) before the measured run. `0` disables warm-up.
   - `NUM_INPUT_TOKENS`, `NUM_OUTPUT_TOKENS`: prompt/generation length for the synthetic dataset.
   - `RESULT_DIR`: where vLLM (and the generated dataset, and the standardized output -- see below) all get written. `DATASET_PATH` is derived from it by default (`$RESULT_DIR/vllm_synthetic_dataset.jsonl`), so changing `RESULT_DIR` alone moves everything together.

   Everything below that -- `.env` loading, the endpoint/trailing-slash fix, and `TOKENIZER_MODEL_NAME` -- runs automatically and normally doesn't need editing. `TOKENIZER_MODEL_NAME` in particular is auto-resolved from `MODEL_NAME` via the Kit's own model registry ([`../kit/src/resolve_tokenizer_name.py`](../kit/src/resolve_tokenizer_name.py), wrapping `get_tokenizer_model_name` in [`../../benchmarking_utils.py`](../../benchmarking_utils.py)) -- override by exporting `TOKENIZER_MODEL_NAME` yourself before running the script if you ever need a different tokenizer.

2. Run it:

   ```bash
   sh quickstart.sh
   ```

   This does three things:
   - **Step 1**: calls `../kit/src/generate_dataset.py` to write a synthetic `.jsonl` prompt dataset to `DATASET_PATH`.
   - **Step 2**: runs `vllm bench serve` against that dataset, printing vLLM's own progress and summary output to the terminal.
   - **Step 3**: converts vLLM's own native result file into the Kit's `_individual_responses.json`/`_summary.json` convention, written alongside it in `RESULT_DIR` -- see [Output structure vs. the Kit](#output-structure-vs-the-kit) below.

3. Check the output. `RESULT_DIR` now contains **both**:
   - vLLM's own native result file (a timestamped JSON with aggregate stats and, because `--save-detailed` is set, per-request arrays -- `ttfts`, `itls`, `input_lens`, `output_lens`, `errors`, ...).
   - The standardized `<same name>_individual_responses.json` / `<same name>_summary.json` pair, generated automatically by Step 3.

   Cross-check the exact native file naming/schema against your installed vLLM version -- see the **Reference** section below.

---

## Custom dataset

To benchmark your own prompts instead of a generated synthetic set, skip step 1 in `quickstart.sh` (comment it out) and point `DATASET_PATH` at your own `.jsonl` file. The schema is the same one used across this repo: one JSON object per line with a `prompt` key, e.g.

```json
{"prompt": "Describe the history of the Roman Empire in depth."}
```

See [`../../prompts/custom_prompt_example.jsonl`](../../prompts/custom_prompt_example.jsonl) for a full example file.

By default, if your file has fewer rows than `--num-prompts`, vLLM oversamples (repeats/shuffles) to fill the count. To make sure every row in your file is used exactly once, in file order, add these two flags to the `VLLM_ARGS` array in `quickstart.sh`:

```bash
--no-oversample
--disable-shuffle
```

(and set `NUM_REQUESTS` to match the number of rows in your file, so nothing is dropped or oversampled).

---

## Real workload (rate-paced)

`quickstart.sh` as written is closed-loop: it fires `NUM_REQUESTS` requests capped at `NUM_CONCURRENT_REQUESTS` in flight. For an open-loop, QPS-paced run instead -- where requests are dispatched at a target rate regardless of how fast prior ones complete -- add your own `QPS`/`BURSTINESS` variables near the top of `quickstart.sh`, then add `--request-rate` (and optionally `--burstiness`) to the `VLLM_ARGS` array:

```bash
--request-rate "$QPS"
--burstiness "$BURSTINESS"
```

- `--request-rate`: target requests per second. Arrival times are synthesized rather than fired all at once.
- `--burstiness`: shape of the inter-arrival distribution. vLLM has no literal constant-spacing mode, so:
  - `1.0` -- exponential inter-arrival times (a Poisson process). This is vLLM's default and an exact match for this repo's `exponential` pacing.
  - A large value such as `100` -- as burstiness grows, the gamma-distributed inter-arrival times collapse toward a fixed interval, approximating (not exactly reproducing) constant-cadence pacing.

You likely still want `--max-concurrency` set to a sane ceiling alongside `--request-rate` so a slow server can't cause unbounded request queuing.

---

## Reference: vLLM CLI flags used

This integration does not wrap or validate vLLM's CLI -- the flags below are what `quickstart.sh` passes, but `vllm bench serve --help` (or `--help=all` for every flag, on your own installed vLLM version) is the source of truth if anything here looks out of date.

| Flag | Purpose |
| --- | --- |
| `--backend openai-chat` | Use the OpenAI-compatible chat completions client. |
| `--base-url` | Base URL of the endpoint (`SAMBANOVA_API_BASE`). |
| `--dataset-name custom` | Read prompts from a local `.jsonl` file instead of vLLM's built-in datasets. |
| `--dataset-path` | Path to the `.jsonl` dataset (generated or custom). |
| `--endpoint chat/completions` | API endpoint path appended to `--base-url`. |
| `--model` | HuggingFace tokenizer id, used locally to count/build tokens (`TOKENIZER_MODEL_NAME`). |
| `--served_model_name` | Actual model name sent in the request body (`MODEL_NAME`). |
| `--custom-output-len` | Requested output token count, applied uniformly to every request (custom dataset only). |
| `--num-prompts` | Total number of requests to send. |
| `--max-concurrency` | Caps how many requests are in flight at once (closed-loop concurrency). |
| `--request-rate` | Target requests/sec for open-loop, rate-paced runs (real-workload mode only). |
| `--burstiness` | Shape of inter-arrival times when `--request-rate` is set; `1.0` = exponential/Poisson, larger values approximate constant spacing. |
| `--no-oversample` | Don't repeat dataset rows to reach `--num-prompts` (custom dataset mode). |
| `--disable-shuffle` | Use dataset rows in file order instead of shuffling (custom dataset mode). |
| `--num-warmups` | Throwaway warm-up requests sent first and excluded from reported metrics. |
| `--save-result` | Write a result JSON file. |
| `--save-detailed` | Include per-request arrays (TTFT, ITL, input/output lengths, errors) in the result JSON. |
| `--result-dir` | Directory vLLM writes its result file(s) into. |

---

## Output structure vs. the Kit

The Kit's own evaluator (`../kit/`) always writes exactly two files per run: `*_individual_responses.json` (a JSON array, one object per request) and `*_summary.json` (a flat dict of aggregate stats).

**vLLM's native output does NOT follow that split** -- confirmed live. `--save-result --save-detailed` write a single combined JSON file that mixes aggregate stats and per-request data in one place, and the per-request data is columnar (parallel arrays), not row-oriented:

| | Kit | vLLM's native file |
|---|---|---|
| File count | 2 (`_individual_responses.json` + `_summary.json`) | 1 (one timestamped result JSON) |
| Per-request shape | JSON array of per-request objects (row-oriented: `[{ttft: ..., ...}, {ttft: ..., ...}]`) | parallel top-level arrays (columnar: `"ttfts": [...], "itls": [...], "input_lens": [...], "output_lens": [...], "errors": [...], "generated_texts": [...]`) -- index `i` across every array is one request |
| Aggregate stats | separate `_summary.json`, percentile-based (`p5`...`p99`) per metric | same file as per-request data, only mean/median/p99 (no full percentile set) -- e.g. `mean_ttft_ms`, `median_ttft_ms`, `p99_ttft_ms` |
| Units | seconds | milliseconds for latency fields (`_ms` suffix), seconds for `ttfts`/durations elsewhere -- mixed, check each field |

**Step 3 standardizes this automatically.** [`../kit/src/convert_vllm_output.py`](../kit/src/convert_vllm_output.py) reads vLLM's already-finished native file and writes a `<same name>_individual_responses.json` / `<same name>_summary.json` pair alongside it, in genuinely Kit-equivalent shape -- reusing the exact same canonical models (`RequestMetric`/`BenchmarkSummary` in [`../kit/src/schemas.py`](../kit/src/schemas.py)) and `to_legacy_flat_dict()` flattening the Kit's own evaluator uses, not a hand-rolled lookalike. Concretely: row-oriented per-request objects instead of vLLM's columnar arrays, and a full `p5`-`p99` quantile set per metric instead of just mean/median/p99.

This is a **post-hoc conversion of an already-written file**, not a live wrapper around the benchmark run itself -- vLLM's own native file is left untouched (both files coexist in `RESULT_DIR`), and nothing about how `vllm bench serve` executes is intercepted or duplicated. That distinction matters: an earlier version of this integration *did* wrap execution live and suffered real schema-drift bugs from doing so (mismatched kwargs, a summary that quietly ended up thinner than the Kit's own); converting one known, finished file after the fact is a much smaller surface, and reusing `schemas.py`'s models directly (rather than re-deriving the mapping) is what keeps this version from drifting the same way.

A couple of fields don't survive the conversion cleanly, and are kept in the summary's `extra` dict rather than forced into a Kit field that doesn't really fit: vLLM never echoes the served/API model name back into its own output (only the tokenizer id, under `model_id`) -- the converter is told the real model name explicitly via `--model-name` (`quickstart.sh` passes `$MODEL_NAME`) rather than guessing it. `burstiness` and `backend` are vLLM-native concepts with no Kit equivalent, so they land in `extra` too. `workload_mode` defaults to `'synthetic'` (`convert_vllm_output.py --workload-mode`) since that's what `quickstart.sh` generates by default; pass a different value explicitly if you've swapped in your own dataset or a rate-paced run and want that reflected in the summary.

The same standardization is also wired into both aiperf quickstart scripts -- see [`../aiperf/README.md`](../aiperf/README.md#output-structure-vs-the-kit) for aiperf's (different) native shape, which already happens to be closer to the Kit's own two-file convention.
