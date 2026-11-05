#!/usr/bin/env python3
"""Generates the Grafana dashboard for benchmarking-kit speed results.

Reads the Athena table `bkit_perf_runs` (one row per benchmark run, uploaded by
benchmarking/athena_export/upload_perf_results.py) and writes
`benchmarking_kit_dashboard.json` next to this script.

Layout:
- "All tools, per model": latest run per model/provider/tool, so each model's Kit, vLLM and
  aiperf results sit side by side.
- One collapsible row per tool: the same panels for that tool only, split by provider.
- "Trends over time": p50 TTFT and per-request output speed per run.

"Latest" means the most recent run per (model, provider[, tool]) at or before the `end_date`
variable. Narrow `workload_mode` when one tool has several workload shapes for the same model.

Import: Grafana -> Dashboards -> New -> Import -> upload the JSON -> pick the Athena data
source. Re-importing with the same uid replaces the dashboard.

Usage:
    python benchmarking/grafana_dashboards/generate_benchmarking_kit_dashboard.py
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

UID = 'benchmarking-kit-speed'
OUTPUT = Path(__file__).resolve().parent / 'benchmarking_kit_dashboard.json'

CONNECTION_ARGS = {'catalog': 'AwsDataCatalog', 'database': 'starterkits_db', 'region': 'us-west-1'}

# One collapsible row per tool. Add new tools here.
TOOLS = [
    ('kit', 'Kit'),
    ('vllm', 'vLLM'),
    ('aiperf', 'aiperf'),
]

# Fixed provider colors, same as the fc-so-testing-suite dashboards. Add new providers here.
PROVIDER_COLORS = {
    'sambanova': '#5B2C6F',
    'groq': '#E74C3C',
    'cerebras': '#E67E22',
    'together': '#3498DB',
    'novita': '#27AE60',
    'fireworks': '#C39BD3',
}

# Bar charts: (column, title, unit). Lower is better for latency, higher for throughput.
BAR_METRICS = [
    ('client_ttft_s_p50', 'TTFT p50', 's'),
    ('client_output_token_per_s_per_request_p50', 'Output tok/s per request p50', 'none'),
    ('client_end_to_end_latency_s_p50', 'End-to-end latency p50', 's'),
    ('client_total_output_throughput', 'Total output throughput (tok/s)', 'none'),
]

TS = 'CAST(from_iso8601_timestamp(r.run_ts) AS timestamp)'  # run_ts is an ISO string
MODEL = 'COALESCE(mn.normalized_model, r.model)'

FILTERS = f"""r.provider IN (${{provider:sqlstring}})
      AND {MODEL} IN (${{model:sqlstring}})
      AND r.tool IN (${{tool:sqlstring}})
      AND r.workload_mode IN (${{workload_mode:sqlstring}})
      AND {TS} <= CAST('${{end_date}}' AS TIMESTAMP)"""


def datasource() -> Dict[str, str]:
    return {'type': 'grafana-athena-datasource', 'uid': '${datasource}'}


def target(sql: str, ref: str = 'A', fmt: int = 1) -> Dict[str, Any]:
    """fmt 1 = table, 0 = time series (long rows with string columns become labelled series)."""
    return {'connectionArgs': CONNECTION_ARGS, 'format': fmt, 'rawSQL': sql, 'refId': ref}


def latest_cte(by_tool: bool, tool: Optional[str]) -> str:
    """Latest run per (model, provider[, tool]) passing the dashboard filters."""
    partition = f'{MODEL}, r.provider' + (', r.tool' if by_tool else '')
    tool_filter = f"\n      AND r.tool = '{tool}'" if tool else ''
    return f"""WITH latest AS (
    SELECT r.*, {MODEL} AS model_name, {TS} AS run_time,
           ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY r.run_ts DESC) AS rn
    FROM bkit_perf_runs r
    LEFT JOIN model_normalization mn ON r.model = mn.raw_model
    WHERE {FILTERS}{tool_filter}
)"""


# --- Panels ---------------------------------------------------------------------------------


def row(panel_id: int, title: str, y: int, collapsed: bool, panels: Optional[List[Dict]] = None) -> Dict:
    return {
        'type': 'row',
        'id': panel_id,
        'title': title,
        'collapsed': collapsed,
        'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': y},
        'panels': panels or [],
    }


def latest_table(panel_id: int, title: str, tool: Optional[str], x: int, y: int, w: int, h: int) -> Dict:
    sql = f"""{latest_cte(by_tool=True, tool=tool)}
SELECT
    model_name AS model,
    provider,
    tool,
    workload_mode,
    num_concurrent_requests AS concurrency,
    run_time AS run_ts,
    ROUND(client_ttft_s_p50, 3) AS ttft_p50_s,
    ROUND(client_ttft_s_p90, 3) AS ttft_p90_s,
    ROUND(client_output_token_per_s_per_request_p50, 1) AS tok_s_per_req_p50,
    ROUND(client_end_to_end_latency_s_p50, 2) AS e2e_p50_s,
    ROUND(client_total_output_throughput, 1) AS total_tok_s,
    ROUND(error_rate * 100, 2) AS error_pct,
    num_requests AS requests,
    run_id
FROM latest
WHERE rn = 1
ORDER BY model, provider, tool"""
    return {
        'type': 'table',
        'id': panel_id,
        'title': title,
        'description': 'Latest run per model, provider and tool at or before the End date variable.',
        'datasource': datasource(),
        'gridPos': {'h': h, 'w': w, 'x': x, 'y': y},
        'fieldConfig': {
            'defaults': {'custom': {'align': 'auto', 'cellOptions': {'type': 'auto'}}},
            'overrides': [
                {
                    'matcher': {'id': 'byName', 'options': 'error_pct'},
                    'properties': [
                        {'id': 'custom.cellOptions', 'value': {'type': 'color-text'}},
                        {
                            'id': 'thresholds',
                            'value': {
                                'mode': 'absolute',
                                'steps': [
                                    {'color': 'green', 'value': None},
                                    {'color': 'orange', 'value': 1},
                                    {'color': 'red', 'value': 5},
                                ],
                            },
                        },
                    ],
                },
                {'matcher': {'id': 'byName', 'options': 'run_ts'}, 'properties': [{'id': 'custom.width', 'value': 180}]},
            ],
        },
        'options': {'cellHeight': 'sm', 'showHeader': True, 'footer': {'show': False}},
        'targets': [target(sql)],
    }


def bar_chart(
    panel_id: int, metric: str, title: str, unit: str, tool: Optional[str], x: int, y: int, w: int, h: int
) -> Dict:
    """Bars per model. Series are providers for one tool, or 'tool · provider' across tools."""
    series = 'provider' if tool else "tool || ' · ' || provider"
    sql = f"""{latest_cte(by_tool=True, tool=tool)}
SELECT model_name AS model, {series} AS series, ROUND({metric}, 3) AS value
FROM latest
WHERE rn = 1 AND {metric} IS NOT NULL
ORDER BY model, series"""
    overrides = []
    if tool:
        overrides = [
            {
                'matcher': {'id': 'byName', 'options': provider},
                'properties': [{'id': 'color', 'value': {'fixedColor': color, 'mode': 'fixed'}}],
            }
            for provider, color in PROVIDER_COLORS.items()
        ]
    return {
        'type': 'barchart',
        'id': panel_id,
        'title': title,
        'datasource': datasource(),
        'gridPos': {'h': h, 'w': w, 'x': x, 'y': y},
        'fieldConfig': {
            'defaults': {
                'color': {'mode': 'palette-classic'},
                'custom': {'fillOpacity': 80, 'lineWidth': 1, 'axisPlacement': 'auto'},
                'unit': unit,
                'min': 0,
            },
            'overrides': overrides,
        },
        'options': {
            'orientation': 'vertical',
            'showValue': 'auto',
            'groupWidth': 0.7,
            'barWidth': 0.8,
            'xTickLabelRotation': -45,
            'legend': {'displayMode': 'list', 'placement': 'bottom', 'showLegend': True},
            'tooltip': {'mode': 'multi', 'sort': 'desc'},
        },
        # Long rows (model, series, value) -> one column per series.
        'transformations': [
            {
                'id': 'groupingToMatrix',
                'options': {'columnField': 'series', 'rowField': 'model', 'valueField': 'value'},
            }
        ],
        'targets': [target(sql)],
    }


def trend(panel_id: int, metric: str, title: str, unit: str, x: int, y: int, w: int, h: int) -> Dict:
    sql = f"""SELECT
    {TS} AS time,
    {MODEL} || ' · ' || r.provider || ' · ' || r.tool AS series,
    {metric} AS value
FROM bkit_perf_runs r
LEFT JOIN model_normalization mn ON r.model = mn.raw_model
WHERE {FILTERS}
  AND {metric} IS NOT NULL
ORDER BY time"""
    return {
        'type': 'timeseries',
        'id': panel_id,
        'title': title,
        'datasource': datasource(),
        'gridPos': {'h': h, 'w': w, 'x': x, 'y': y},
        'fieldConfig': {
            'defaults': {
                'unit': unit,
                'custom': {'drawStyle': 'line', 'showPoints': 'always', 'pointSize': 6, 'spanNulls': True},
            },
            'overrides': [],
        },
        'options': {
            'legend': {'displayMode': 'list', 'placement': 'bottom', 'showLegend': True},
            'tooltip': {'mode': 'multi', 'sort': 'desc'},
        },
        'targets': [target(sql, fmt=0)],
    }


def bar_grid(first_id: int, tool: Optional[str], y: int) -> List[Dict]:
    """The four BAR_METRICS charts in a 2x2 grid."""
    panels = []
    for i, (metric, title, unit) in enumerate(BAR_METRICS):
        panels.append(bar_chart(first_id + i, metric, title, unit, tool, x=12 * (i % 2), y=y + 8 * (i // 2), w=12, h=8))
    return panels


# --- Variables ------------------------------------------------------------------------------


def query_variable(name: str, label: str, sql: str) -> Dict:
    return {
        'type': 'query',
        'name': name,
        'label': label,
        'datasource': datasource(),
        'query': {'connectionArgs': CONNECTION_ARGS, 'rawSQL': sql},
        'multi': True,
        'includeAll': True,
        'current': {'selected': True, 'text': ['All'], 'value': ['$__all']},
        'refresh': 1,
        'sort': 1,
    }


def variables() -> List[Dict]:
    return [
        {
            'type': 'datasource',
            'name': 'datasource',
            'label': 'Data source',
            'query': 'grafana-athena-datasource',
        },
        query_variable('provider', 'Provider', 'SELECT DISTINCT provider FROM bkit_perf_runs ORDER BY 1'),
        query_variable(
            'model',
            'Model',
            'SELECT DISTINCT COALESCE(mn.normalized_model, r.model) FROM bkit_perf_runs r '
            'LEFT JOIN model_normalization mn ON r.model = mn.raw_model ORDER BY 1',
        ),
        query_variable('tool', 'Tool', 'SELECT DISTINCT tool FROM bkit_perf_runs ORDER BY 1'),
        query_variable('workload_mode', 'Workload', 'SELECT DISTINCT workload_mode FROM bkit_perf_runs ORDER BY 1'),
        {
            'type': 'textbox',
            'name': 'end_date',
            'label': 'End date (latest as of)',
            'query': '2099-12-31 23:59:59',
            'current': {'text': '2099-12-31 23:59:59', 'value': '2099-12-31 23:59:59'},
        },
    ]


# --- Dashboard ------------------------------------------------------------------------------


def build() -> Dict[str, Any]:
    panels: List[Dict] = []

    # All tools, expanded: its panels follow the row at top level.
    panels.append(row(100, 'All tools, per model', y=0, collapsed=False))
    panels.append(latest_table(101, 'Latest results per model (all tools)', None, x=0, y=1, w=24, h=9))
    panels.extend(bar_grid(110, None, y=10))

    # One collapsed row per tool: its panels are nested inside the row.
    y = 26
    for i, (tool, label) in enumerate(TOOLS):
        base = 1000 * (i + 1)
        nested = [latest_table(base + 1, f'Latest results per model ({label})', tool, x=0, y=y + 1, w=24, h=8)]
        nested.extend(bar_grid(base + 10, tool, y=y + 9))
        panels.append(row(base, f'{label}, per model', y=y, collapsed=True, panels=nested))
        y += 1

    trends = [
        trend(9001, 'client_ttft_s_p50', 'TTFT p50 per run', 's', x=0, y=y + 1, w=12, h=9),
        trend(9002, 'client_output_token_per_s_per_request_p50', 'Output tok/s per request p50 per run', 'none',
              x=12, y=y + 1, w=12, h=9),
    ]
    panels.append(row(9000, 'Trends over time', y=y, collapsed=True, panels=trends))

    return {
        'uid': UID,
        'title': 'Benchmarking Kit - Speed Results',
        'tags': ['athena', 'benchmarking-kit', 'performance'],
        'schemaVersion': 38,
        'editable': True,
        'time': {'from': 'now-180d', 'to': 'now'},
        'timezone': 'utc',
        'templating': {'list': variables()},
        'panels': panels,
    }


def main() -> None:
    OUTPUT.write_text(json.dumps(build(), indent=2) + '\n')
    print(f'Wrote {OUTPUT}')


if __name__ == '__main__':
    main()
