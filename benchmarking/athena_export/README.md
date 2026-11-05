# Athena export

Uploads benchmark run summaries to S3 so they can be queried in Athena, in the table
`bkit_perf_runs` (one row per run) and the view `bkit_latest_perf` (latest run per configuration).
Kit, vLLM and aiperf runs are all supported, since they write the same `*_summary.json` shape.

The table and view are defined in the `athena_data_prep` submodule, under
`projects/benchmarking_kit/`. The uploader reads its column list from there, so check the
submodule out first: `git submodule update --init benchmarking/athena_data_prep`.

## Upload results

To upload automatically at the end of a run, pass `--upload-to-athena True` to any tool (or set
`UPLOAD_TO_ATHENA=True` in its quickstart script); see
[Publishing results to Grafana](../README.md#publishing-results-to-grafana). This script uploads results
after the fact, e.g. older runs or a retry after a failed upload.

Needs `AWS_REGION`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in the repo-root `.env`.

```bash
# Preview the rows and S3 keys without uploading
python benchmarking/athena_export/upload_perf_results.py <results dir or *_summary.json...> --dry-run

# Upload
python benchmarking/athena_export/upload_perf_results.py benchmarking/benchmarking_tools/aiperf/data/fireworks_agentx_20261104
```

Options:

- `--provider`: provider name. By default it is detected from the endpoint URL recorded in the
  summary (SambaNova, Fireworks, Groq, Cerebras, Together, Novita, OpenAI), or `unknown`.
  Pass it for runs made before the URL was recorded.
- `--config-env`: a deployment label such as `prod` or `staging`.
- `--local-out <dir>`: also write each row locally, mirroring the S3 keys.

Each run lands at `benchmarking_kit/perf_runs/run_date=YYYY-MM-DD/<run_id>/summary.jsonl`.
Uploading the same run again overwrites it, so re-running the uploader is safe.

## Changing the table

Follow `athena_data_prep/GUIDELINES.md`: edit the table YAML in the submodule, merge it to the
submodule's `main`, bump the submodule here, then `athena-prep apply --project benchmarking_kit`.
New columns go at the end and need `--allow-recreate`.

## Grafana dashboard

`benchmarking/grafana_dashboards/generate_benchmarking_kit_dashboard.py` writes
`benchmarking_kit_dashboard.json` next to itself. To import it in Grafana: Dashboards → New → Import →
upload the JSON → pick the Athena data source. Re-importing replaces the dashboard (same uid).
