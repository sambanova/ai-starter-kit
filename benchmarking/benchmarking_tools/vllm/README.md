# vLLM benchmarking

This directory runs vLLM's own [`vllm bench serve`](https://docs.vllm.ai/) CLI **directly** against a SambaNova endpoint — no Python wrapper. This repo is only used to generate a synthetic prompt dataset ([`../kit/src/generate_dataset.py`](../kit/src/generate_dataset.py)); the benchmark itself is vLLM's native, unmodified command.

Complements the Kit's own native evaluator in [`../kit/`](../kit/) — see [`../../README.md`](../../README.md) for the full benchmarking overview.

**Verification status**: `quickstart.sh` is live-verified end-to-end (vLLM 0.11.2). One real bug found and fixed: `SAMBANOVA_API_BASE` needs a trailing slash. vLLM concatenates `--base-url` + `--endpoint` with no separator, so without it `"https://host/v1"` + `"chat/completions"` becomes the malformed `"https://host/v1chat/completions"` (405s, and vLLM's readiness check silently retries for up to 600s instead of surfacing the error). `quickstart.sh` now normalizes this automatically.

---

## Prerequisites

1. Install vLLM (`pip install vllm`). _No local GPU or model download needed for API benchmarking — vLLM's client just drives HTTP requests against the endpoint you configure below._

2. Set `SAMBANOVA_API_BASE` and `SAMBANOVA_API_KEY` — see the top-level README's [Getting a SambaNova API key](../../../README.md#getting-a-sambanova-api-key-and-setting-your-generative-models) section.

   vLLM's OpenAI-compatible client reads its key from `OPENAI_API_KEY`, not `SAMBANOVA_API_KEY` — `quickstart.sh` maps one onto the other for you, so you only need to export `SAMBANOVA_API_KEY`.

---

## Quickstart (synthetic dataset)

1. Edit the "Parameters" section at the top of [`quickstart.sh`](./quickstart.sh):
   - `MODEL_NAME`: model name as exposed by the API (sent to vLLM as `--served_model_name`).
   - `NUM_REQUESTS`, `NUM_CONCURRENT_REQUESTS`: total requests and concurrency ceiling.
   - `NUM_WARMUP_REQUESTS`: throwaway requests excluded from reported metrics. `0` disables.
   - `QPS`: `"inf"` (default) fires a closed-loop burst; a number paces open-loop as a Poisson process — see [Real workload](#real-workload-rate-paced).
   - `NUM_INPUT_TOKENS`, `NUM_OUTPUT_TOKENS`: prompt/generation length.
   - `RESULT_DIR`: where vLLM, the generated dataset, and the standardized output all get written (`DATASET_PATH` derives from it by default).

   Everything below (`.env` loading, the trailing-slash fix, `TOKENIZER_MODEL_NAME` auto-resolution via [`../kit/src/resolve_tokenizer_name.py`](../kit/src/resolve_tokenizer_name.py)) runs automatically.

2. Run `sh quickstart.sh`. Three steps:
   - Generates a synthetic `.jsonl` dataset to `DATASET_PATH`.
   - Runs `vllm bench serve` against it, printing vLLM's own progress/summary.
   - Converts vLLM's native result into the Kit's `_individual_responses.json`/`_summary.json` — see [Output structure vs. the Kit](#output-structure-vs-the-kit).

3. `RESULT_DIR` then contains vLLM's native result file (timestamped JSON, per-request arrays via `--save-detailed`) and the standardized pair. Cross-check the exact native schema against your installed vLLM version if anything looks off — see [Reference](#reference-vllm-cli-flags-used).

---

## Custom dataset

Skip step 1 in `quickstart.sh` (comment it out) and point `DATASET_PATH` at your own `.jsonl`, one `{"prompt": "..."}` object per line, e.g.:
```json
{"prompt": "Describe the history of the Roman Empire in depth."}
```
See [`custom_prompt_example.jsonl`](../../prompts/custom_prompt_example.jsonl) for a full example.

By default, if your file has fewer rows than `--num-prompts`, vLLM oversamples (repeats/shuffles) to fill the count. To use every row exactly once in file order, add to `VLLM_ARGS` in `quickstart.sh`:
```bash
--no-oversample
--disable-shuffle
```
(and set `NUM_REQUESTS` to your file's row count).

---

## Real workload (rate-paced)

`quickstart.sh`'s `QPS` switches pacing (live-verified both ways):
- `QPS="inf"` (default): closed-loop — `NUM_REQUESTS` capped at `NUM_CONCURRENT_REQUESTS` in flight.
- `QPS=<number>`, e.g. `2`: open-loop — dispatched at that rate as a Poisson process (`--burstiness 1`, vLLM's own Poisson definition — confirmed live: `Namespace` shows `request_rate=2.0, burstiness=1.0`, and the result filename embeds the rate, e.g. `openai-chat-2.0qps-concurrency4-...json`).

Both map to `vllm bench serve`'s own flags: `--request-rate "$QPS" --burstiness 1`. `--max-concurrency "$NUM_CONCURRENT_REQUESTS"` stays set alongside `--request-rate` even when paced, so a slow server can't cause unbounded queuing.

For a non-Poisson arrival shape, `--burstiness` also accepts `0 < burstiness < 1` (burstier) or larger values (approximating constant cadence) — edit the hardcoded `1` in `VLLM_ARGS` if needed.

Since `QPS="inf"` is the default, the standardized `_summary.json` records `qps: null` for closed-loop runs rather than a literal `Infinity` (not valid JSON per RFC 8259) — `convert_vllm_output.py` normalizes this.

---

## Reference: vLLM CLI flags used

This integration doesn't wrap or validate vLLM's CLI — the table below is what `quickstart.sh` passes; `vllm bench serve --help` (or `--help=all`) on your installed version is the source of truth.

| Flag | Purpose |
| --- | --- |
| `--backend openai-chat` | OpenAI-compatible chat completions client. |
| `--base-url` | Endpoint base URL (`SAMBANOVA_API_BASE`). |
| `--dataset-name custom` | Read prompts from a local `.jsonl` file. |
| `--dataset-path` | Path to the `.jsonl` dataset. |
| `--endpoint chat/completions` | API endpoint path appended to `--base-url`. |
| `--model` | HF tokenizer id, used locally for token counting (`TOKENIZER_MODEL_NAME`). |
| `--served_model_name` | Actual model name sent in the request body (`MODEL_NAME`). |
| `--custom-output-len` | Output token count, applied uniformly (custom dataset only). |
| `--num-prompts` | Total requests to send. |
| `--max-concurrency` | Concurrency cap (closed-loop). |
| `--request-rate` | Target requests/sec (`QPS`); `"inf"` = closed-loop burst, a number paces open-loop. |
| `--burstiness` | Inter-arrival shape when `--request-rate` isn't `inf`; hardcoded to `1` (Poisson). |
| `--no-oversample` | Don't repeat rows to reach `--num-prompts` (custom dataset). |
| `--disable-shuffle` | Use dataset rows in file order (custom dataset). |
| `--num-warmups` | Throwaway warm-up requests, excluded from reported metrics. |
| `--save-result` | Write a result JSON file. |
| `--save-detailed` | Include per-request arrays (TTFT, ITL, lengths, errors). |
| `--result-dir` | Output directory. |

---

## Output structure vs. the Kit

The Kit always writes two files: `*_individual_responses.json` (JSON array, one object per request) and `*_summary.json` (flat dict of aggregate stats).

**vLLM's native output does not follow that split** — confirmed live. `--save-result --save-detailed` write one combined JSON mixing aggregate stats and columnar (parallel-array) per-request data:

| | Kit | vLLM's native file |
|---|---|---|
| File count | 2 (`_individual_responses.json` + `_summary.json`) | 1 (timestamped result JSON) |
| Per-request shape | row-oriented array of objects | columnar: `"ttfts": [...], "itls": [...], "input_lens": [...], "output_lens": [...], "errors": [...], "generated_texts": [...]` — index `i` across arrays = one request |
| Aggregate stats | separate `_summary.json`, full percentiles (p5...p99) | same file, only mean/median/p99 (e.g. `mean_ttft_ms`, `median_ttft_ms`, `p99_ttft_ms`) |
| Units | seconds | milliseconds for latency (`_ms`), seconds elsewhere — mixed, check each field |

**Step 3 standardizes this automatically.** [`convert_vllm_output.py`](../kit/src/convert_vllm_output.py) reads vLLM's finished native file and writes a Kit-equivalent `_individual_responses.json`/`_summary.json` pair alongside it, reusing the Kit's own canonical models (`RequestMetric`/`BenchmarkSummary` in [`schemas.py`](../kit/src/schemas.py)) rather than a hand-rolled lookalike — row-oriented objects instead of columnar arrays, and a full percentile set instead of just mean/median/p99.

This is a **post-hoc conversion of an already-written file**, not a live wrapper — vLLM's native file is untouched and coexists with the standardized one. An earlier version of this integration wrapped execution live and suffered real schema-drift bugs; converting one known, finished file (reusing `schemas.py` directly) keeps this version from drifting the same way.

Fields with no Kit equivalent (`burstiness`, `backend`) land in the summary's `extra` dict; vLLM never echoes the model name back either, so the converter is told it explicitly via `--model-name`. `workload_mode` defaults to `'synthetic'` — pass a different value if you've swapped in your own dataset or a rate-paced run.

Same standardization is wired into both aiperf quickstart scripts — see [`../aiperf/README.md`](../aiperf/README.md#output-structure-vs-the-kit) for aiperf's (already closer to the Kit's) native shape.
