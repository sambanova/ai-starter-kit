"""Uploads benchmark run summaries to S3 for the Athena table `bkit_perf_runs`.

Reads each tool's `*_summary.json` (Kit, vLLM and aiperf all write the same flat shape via
`BenchmarkSummary.to_legacy_flat_dict()`), maps it onto the table's columns and writes one JSONL
row per run to:

    s3://<bucket>/benchmarking_kit/perf_runs/run_date=YYYY-MM-DD/<run_id>/summary.jsonl

The column list and types come from the table definition in the athena_data_prep submodule
(`projects/benchmarking_kit/tables/bkit_perf_runs.yaml`), so the uploader can never write a
column the table doesn't have. Summary keys without a column are kept in `extra_json`.
Re-uploading a run overwrites the same key, so uploads are idempotent.

Usage:
    python benchmarking/athena_export/upload_perf_results.py <summary files or dirs...> \
        [--provider fireworks] [--config-env prod] [--dry-run] [--local-out ./athena_rows]
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ATHENA_PREP_DIR = REPO_ROOT / 'benchmarking' / 'athena_data_prep'
TABLE_YAML = ATHENA_PREP_DIR / 'projects' / 'benchmarking_kit' / 'tables' / 'bkit_perf_runs.yaml'

# Metric stats stored as columns: `<metric>_<stat>`. Everything else under `results_<metric>_*`
# (stddev, p5/p25/p75) stays in extra_json.
STAT_KEYS = {
    'mean': 'mean',
    'min': 'min',
    'max': 'max',
    'p50': 'quantiles_p50',
    'p90': 'quantiles_p90',
    'p95': 'quantiles_p95',
    'p99': 'quantiles_p99',
}

# Run-level scalars, written as `results_<name>` in the summary file.
RESULT_SCALARS = [
    'num_completed_requests',
    'number_errors',
    'error_rate',
    'error_code_frequency',
    'client_total_output_throughput',
]

# Top-level summary keys copied to the column of the same name.
TOP_LEVEL = [
    'model',
    'tool',
    'workload_mode',
    'num_concurrent_requests',
    'qps',
    'qps_distribution',
    'num_input_tokens',
    'num_output_tokens',
    'benchmark_duration_s',
    'request_throughput',
]

# Hostname substring -> provider, for summaries that record api_base_url.
PROVIDER_HOSTS = {
    'sambanova': 'sambanova',
    'fireworks': 'fireworks',
    'groq': 'groq',
    'cerebras': 'cerebras',
    'together': 'together',
    'novita': 'novita',
    'openai.com': 'openai',
}

MULTIMODAL_RE = re.compile(r'_multimodal_(small|medium|large)_')


def load_columns() -> List[Tuple[str, str]]:
    """Returns (name, type) for each table column, in table order."""
    table = yaml.safe_load(TABLE_YAML.read_text())
    columns = []
    for col in table['columns']:
        if isinstance(col, str):
            columns.append((col, 'string'))
        else:
            columns.append((col['name'], col.get('type', 'string')))
    return columns


def load_settings() -> Dict[str, Optional[str]]:
    settings = yaml.safe_load((ATHENA_PREP_DIR / 'athena_prep.yaml').read_text())
    return {
        'bucket': os.environ.get('ATHENA_PREP_BUCKET') or settings['bucket'],
        'region': os.environ.get('AWS_REGION') or settings.get('region'),
    }


def find_summary_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(p.rglob('*_summary.json')))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def detect_provider(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    host = urlparse(base_url).hostname or ''
    for needle, provider in PROVIDER_HOSTS.items():
        if needle in host:
            return provider
    return None


def cast(value: Any, col_type: str) -> Any:
    if value is None or value == '':
        return None
    if col_type == 'bigint':
        return int(round(float(value)))
    if col_type == 'double':
        return float(value)
    return value if isinstance(value, str) else json.dumps(value)


def build_row(
    path: Path,
    columns: List[Tuple[str, str]],
    provider: Optional[str],
    config_env: Optional[str],
    uploaded_at: str,
) -> Dict[str, Any]:
    raw_text = path.read_text()
    summary: Dict[str, Any] = json.loads(raw_text)
    remaining = dict(summary)
    row: Dict[str, Any] = {}

    for key in TOP_LEVEL:
        row[key] = remaining.pop(key, None)
    row['tool'] = row['tool'] or 'kit'  # Kit summaries written before the shared schema had no 'tool'
    row['run_name'] = remaining.pop('name', None)
    # Kit custom mode names the request count `request_count`; every other mode `num_requests`.
    row['num_requests'] = remaining.pop('num_requests', None) or remaining.pop('request_count', None)

    for key in RESULT_SCALARS:
        row[key] = remaining.pop(f'results_{key}', None)

    column_names = {name for name, _ in columns}
    for metric in {c.rsplit('_', 1)[0] for c in column_names if c.endswith('_p50')}:
        for stat, summary_suffix in STAT_KEYS.items():
            row[f'{metric}_{stat}'] = remaining.pop(f'results_{metric}_{summary_suffix}', None)

    # Kit uses run_uuid; aiperf's benchmark_id is also recorded as run_uuid by the converter.
    run_id = remaining.pop('run_uuid', None) or hashlib.sha1(raw_text.encode()).hexdigest()[:16]
    timestamp = remaining.pop('timestamp', None)
    if timestamp is None:
        print(f'warning: {path} has no timestamp, using file mtime', file=sys.stderr)
        timestamp = path.stat().st_mtime
    run_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    base_url = remaining.get('api_base_url')
    resolved_provider = provider or detect_provider(base_url)
    if resolved_provider is None:
        print(f'warning: {path}: provider unknown, pass --provider', file=sys.stderr)

    mm = MULTIMODAL_RE.search(row['run_name'] or '') or MULTIMODAL_RE.search(path.name)

    row.update(
        run_ts=run_dt.strftime('%Y-%m-%dT%H:%M:%S'),
        run_id=str(run_id),
        provider=resolved_provider or 'unknown',
        config_env=config_env,
        base_url=base_url,
        multimodal_img_size=mm.group(1) if mm else 'na',
        source_file=str(path.resolve()),
        uploaded_at=uploaded_at,
        extra_json=json.dumps(remaining, sort_keys=True),
    )

    unknown = set(row) - column_names
    if unknown:
        raise ValueError(f'row keys missing from {TABLE_YAML.name}: {sorted(unknown)}')
    out = {name: cast(row.get(name), col_type) for name, col_type in columns}
    out['_run_date'] = run_dt.strftime('%Y-%m-%d')  # partition value, not a column
    return out


def s3_key(row: Dict[str, Any]) -> str:
    return f'benchmarking_kit/perf_runs/run_date={row["_run_date"]}/{row["run_id"]}/summary.jsonl'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='*_summary.json files, or directories searched recursively.')
    parser.add_argument('--provider', help='Provider name; overrides detection from the recorded base URL.')
    parser.add_argument('--config-env', help='Deployment/environment label, e.g. prod or staging.')
    parser.add_argument('--dry-run', action='store_true', help='Print rows and S3 keys without uploading.')
    parser.add_argument('--local-out', help='Also write each row under this directory, mirroring the S3 keys.')
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / '.env')
    columns = load_columns()
    uploaded_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    files = find_summary_files(args.paths)
    if not files:
        sys.exit('no *_summary.json files found')

    rows = [build_row(f, columns, args.provider, args.config_env, uploaded_at) for f in files]

    s3 = None
    settings = load_settings()
    if not args.dry_run:
        import boto3

        s3 = boto3.client('s3', region_name=settings['region'])

    for row in rows:
        key = s3_key(row)
        body = json.dumps({k: v for k, v in row.items() if not k.startswith('_')}) + '\n'
        if args.local_out:
            local = Path(args.local_out) / key
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(body)
        if s3 is not None:
            s3.put_object(Bucket=settings['bucket'], Key=key, Body=body.encode(), ContentType='application/x-ndjson')
            print(f'uploaded s3://{settings["bucket"]}/{key}  ({row["tool"]} {row["model"]})')
        else:
            print(f'[dry-run] s3://{settings["bucket"]}/{key}  ({row["tool"]} {row["model"]})')


if __name__ == '__main__':
    main()
