# Benchmarking bundles

This directory runs **end-to-end benchmarking for model bundles** using the Kit's own native evaluator — **synthetic** (concurrency-based), **real workload** (QPS-based), and **custom** (your own dataset) inference tests, with optional **row-level concurrency**. Results include automatic **batching** and **switching time** estimations.

Bundles run **Kit jobs only**. vLLM and aiperf are driven via their own native CLIs directly (no Python job runner) — see [`../benchmarking_tools/vllm/README.md`](../benchmarking_tools/vllm/README.md) and [`../benchmarking_tools/aiperf/README.md`](../benchmarking_tools/aiperf/README.md) if you want to sweep those tools across configs; a plain shell loop over their quickstart scripts is the equivalent of a "bundle" for those tools.

---

## Required steps

### 1. Configuration file

Modify `<PATH TO AISK REPO HERE>/benchmarking/benchmarking_bundles/config.yaml`:

```yaml
jobs_path: '<PATH TO AISK REPO HERE>/benchmarking/benchmarking_bundles/jobs_example.yaml'
llm_api: 'sncloud'  # only option currently supported
output_files_dir: '<PATH TO AISK REPO HERE>/benchmarking/data/bundle_tests/output_files'
consolidated_results_dir: '<PATH TO AISK REPO HERE>/benchmarking/data/bundle_tests/consolidated_results'
timeout: 3600
time_delay: 0
# Row-level concurrency
concurrency_enabled: False
max_workers: 4
# Prompt behavior
use_multiple_prompts: False
# Batch sizes used to infer batching for the switching-time calculation
batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128]
```

#### Key notes:
- `timeout`: max seconds per job (overridable per job — see below).
- `concurrency_enabled`: if true, jobs run concurrently via a thread pool.
- `max_workers`: max parallel jobs.
- `time_delay`: optional sleep between runs (per job).
- `use_multiple_prompts`: if true, randomly uses prompts from `<PATH TO AISK REPO>/benchmarking/prompts/user-prompt_template-text_instruct.yaml`. Only applies to synthetic jobs; overridable per job.
- `batch_sizes`: allowed batch sizes for the switching-time calculation (see [Batching analysis](#batching-analysis)) — observed request groups snap **up** to the nearest value. Defaults to powers of two up to 128; add non-power-of-two sizes (e.g. `6`) only if your deployment actually serves them.

### 2. Jobs file

Modify `<PATH TO AISK REPO HERE>/benchmarking/benchmarking_bundles/jobs_example.yaml` — a YAML list of job dicts (not a CSV table), one per benchmark job. Each job's keys are the Kit evaluator's own parameters for that `mode`, so shape varies per mode without forcing everything into the same tabular columns.

```yaml
jobs:
  - mode: synthetic                            # workload shape: 'custom' | 'synthetic' | 'real_workload'
    model_name: MiniMax-M3
    num_input_tokens: 1000
    num_output_tokens: 1000
    num_requests: 8
    num_concurrent_requests: 8

  - mode: real_workload
    model_name: MiniMax-M3
    num_input_tokens: 1000
    num_output_tokens: 1000
    num_requests: 20
    qps: 4.0
    qps_distribution: constant

  - mode: synthetic                            # multimodal_image_size includes an image in the request
    model_name: gemma-4-31B-it
    num_input_tokens: 1000
    num_output_tokens: 1000
    num_requests: 8
    num_concurrent_requests: 8
    multimodal_image_size: large

  - mode: custom
    model_name: DeepSeek-V3.2
    input_file_path: ../prompts/custom_prompt_example.jsonl
    num_concurrent_requests: 8
    num_output_tokens: 1000
```

See `jobs_example.yaml` itself for the full set (3 example jobs per model, mixing modes so each mode has a worked example).

#### Configuration parameters

Key names match the Kit's CLI flags in `evaluator.py` one-to-one (e.g. `--num-input-tokens` → `num_input_tokens`), so anything that works on the CLI works here too.

- `mode` — `synthetic` (fixed tokens, closed-loop concurrency), `real_workload` (fixed tokens, open-loop QPS pacing), or `custom` (dataset via `input_file_path`) — see `jobs_example.yaml` for a worked example of each.
- `model_name` — model in the bundle.
- `num_input_tokens` — input tokens in the generated prompt.
- `num_output_tokens` — max output tokens. Same field for `mode: custom` (defaults to `150` there, since the API requires a concrete value).
- `num_requests` — total requests sent.
- `num_warmup_requests` — throwaway requests before the measured run (`0`/omit disables). Synthetic warm-ups match the test's concurrency; real-workload warm-ups fire immediately, ignoring the pacing rate.
- `num_concurrent_requests` — required for `synthetic`; optional for `custom` (default `10`); ignored for `real_workload`.
- `input_file_path` — required for `custom`: path (relative to the runner's CWD) to a `{"prompt": "..."}` JSONL dataset. `num_input_tokens`/`num_requests` are ignored (the dataset determines both); `num_output_tokens` still caps generation, unless overridden by an explicit `sampling_params: {max_tokens_to_generate: <n>}`.
- `qps` — required for `real_workload`; recommended `< 10`.
- `qps_distribution` — wait-time distribution for `real_workload`: `constant` (default) or `exponential`.
- `multimodal_image_size` — for multimodal models only: `small` (500×500px), `medium` (1000×1000px), `large` (2000×2000px); omit otherwise (defaults to `'na'`).
- `results_dir`, `timeout`, `llm_api`, `user_metadata` — optional per job, defaulting from `config.yaml` (and `{'model_idx': 0}`) when omitted.

> **Important:** `num_input_tokens + num_output_tokens` must not exceed the model's max sequence length (e.g. a 4096-token model could use `4000`/`64`) — and remember models may add their own internal prompting tokens on top.

### 3. Execution modes

#### 3.1 Sequential execution

If `concurrency_enabled: false`, jobs run **in order**. Useful for limited resources, exposing switching-time effects, and debugging.

#### 3.2 Concurrent execution

If `concurrency_enabled: true`, every job runs **in parallel** via a `ThreadPoolExecutor` (`max_workers` caps concurrency). Useful for more realistic load, or faster turnaround on large bundles.

### 4. Run the benchmark

From the **root of the benchmarking_bundles module**:

```bash
bash run_synthetic_perfomance_bundle_eval.sh
```

### 5. Results

#### 5.1 Results per model config

Stored under `output_files_dir`. Two files per job:

1. **Individual responses** (`*individual_responses.json`) — one entry per request: TTFT, end-to-end latency, output throughput.
2. **Summary** (`*summary.json`) — aggregated min/max/mean/median(p50)/stddev across those requests.

See the kit's CLI docs in the main [README](../README.md) for more detail on these files.

#### 5.2 Consolidated results

Written to `consolidated_results_dir` as one `<run_name>.xlsx` workbook:
- **`per_model`** — one row per job in `jobs_example.yaml`.
- **`bundle_summary`** — aggregated throughput for the bundle as a whole (see [Bundle-level summary](#53-bundle-level-summary)).

##### Terminology

- **Server metrics** (`server_` prefix) — from the inference server API.
- **Client metrics** (`client_` prefix) — computed on the client/sending machine.
- **Suffixes** — `min`/`max`/`p50`/`mean`; `s` = seconds.

##### Main consolidated metrics

- **TTFT** — typically higher variance under concurrency (request queueing); client-side also picks up network time.
- **End-to-end latency** — includes queueing + TTFT; client-side also picks up network time.
- **Output tokens/sec** — per-request throughput; may drop per-request as batch size grows even as overall throughput rises.
- **Acceptance rate** — for speculative decoding pairs.

##### Batching analysis

Batching is estimated automatically from request timing: requests with identical `server_ttft` are grouped, and the group size is snapped **up** to the nearest value in `batch_sizes` (defaults to powers of two up to 128).

Reported fields:
- `request_batching_frequencies` — frequency of observed batch sizes.
- `representative_batch_size` — the batch size accounting for >50% of requests.

__Note__: this is an estimate from close server TTFTs, not a guaranteed true batch size — a dedicated environment gives more reliable results.

##### Switching time

The overhead of loading a new model/config into HBM, triggered by changes in model name, sequence length, or batch size.

**How it's estimated**, per benchmark run (by UUID): find the largest estimated batch size, then `switching_time = max(server_ttft_s) - min(server_ttft_s)` across requests **at that batch level only**.

**Where to find it**: column `switching_time`, derived from the first requests in `individual_responses.json`.

**Notes**: include a warm-up set of models first (same or separate jobs file) so HBM already holds the configs you're testing, then run the jobs you want switching time for; include multiple sequence lengths/batch sizes matching your deployment; run multiple requests per row for stable estimates.

#### 5.3 Bundle-level summary

`per_model` reports throughput **per job** (one model, one batch size/QPS, one context length); `bundle_summary` reports throughput **for the bundle as a whole**.

##### Rows

- **`ALL`** — grand total across every job.
- **One row per model family** (inferred from `model_name`, same family detection used elsewhere in the kit) — e.g. compare all `llama3` rows against all `qwen` rows.

__Note__: an unrecognized model name falls back to the `llama2` family bucket rather than `unknown` — check naming if a model rolls up unexpectedly.

##### Columns

| Column | Meaning |
|---|---|
| `family` | `ALL` or the family name |
| `num_model_configs` | Jobs in this group |
| `total_num_requests_started` | Requests attempted/dispatched |
| `total_errors` | Requests that failed |
| `total_completed_requests` | Requests completed — throughput numerator |
| `total_duration_s` | Wall-clock time, first row start to last row finish |
| `delay_time_s` | Portion of `total_duration_s` from the configured `time_delay` |
| `total_effective_duration_s` | `total_duration_s - delay_time_s` — throughput denominator |
| `bundle_rps` | `total_completed_requests / total_effective_duration_s` |
| `bundle_rpm` | `bundle_rps * 60` |
| `concurrency_enabled` | Whether this run used row-level concurrency |

##### How it's calculated

Per-row timestamps aren't stored directly, so each row's window is estimated from its `num_completed_requests`, `num_completed_requests_per_min`, and completion `timestamp`. A group's `total_duration_s` spans its earliest estimated start to latest estimated end — avoiding double-counting overlapping time under `concurrency_enabled: true`.

`delay_time_s` accounts for the `time_delay` sleep after each row: in sequential mode, `time_delay * (num_model_configs - 1)` (a delay between every pair of rows, not after the last); in concurrent mode (or a single-row group), `0` (row-level sleeps don't chain into a shared timeline). This keeps `bundle_rps`/`bundle_rpm` from being deflated by dead time that's just a configuration artifact.
