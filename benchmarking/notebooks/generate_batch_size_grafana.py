import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Read the CSV file
results_dir = '<AISK_REPOSITORY_PATH>/benchmarking/data/results/grafana/'
input_filename = results_dir + 'grafana.csv'
output_filename = results_dir + 'grafana_grouped_interim.csv'
summary_output_filename = results_dir + 'grafana_grouped_summary.csv'

df = pd.read_csv(input_filename)

# Step 2: Convert timestamp to datetime and sort
# Use start_time instead to be more accurate
df['@timestamp'] = pd.to_datetime(df['@timestamp'])
# df = df.sort_values('@timestamp').reset_index(drop=True)
df = df.sort_values('start_time').reset_index(drop=True)

# Step 3: Calculate time differences and generate test_sequence
# df['time_diff'] = df['@timestamp'].diff().dt.total_seconds()
df['time_diff'] = df['start_time'].diff()
df['test_sequence'] = (df['time_diff'] >= 60).cumsum()

# Cleanup: Remove temporary column
# df.drop(columns=['time_diff'], inplace=True)

# 4. Create batch_size_groups column
# First calculate group sizes
group_sizes = df.groupby(['test_sequence', 'batch_size']).size().reset_index(name='count')

# Create dictionary {batch_size: count}
group_sizes['batch_size_groups'] = group_sizes.apply(lambda row: {row['batch_size']: row['count']}, axis=1)

# Merge back to original dataframe
df = df.merge(
    group_sizes[['test_sequence', 'batch_size', 'batch_size_groups']], on=['test_sequence', 'batch_size'], how='left'
)

# Interim Result
print(df.head(10))
df.to_csv(output_filename, index=False)


# 5. Create summary dataframe
# Helper function for unique dictionaries
def get_unique_dicts(series: pd.Series) -> List[Dict[Any, Any]]:
    seen = set()
    unique_dicts = []
    for d in series:
        # Represent dictionary as frozenset of items for hashability
        rep = frozenset(d.items())
        if rep not in seen:
            seen.add(rep)
            unique_dicts.append(d)
    return unique_dicts


# Pandas Named aggregation
# test_sequence_count: Number of rows per group
# all_batch_size_groupings: Unique batch size dictionaries
summary = df.groupby(['prompt_tokens_count', 'completion_tokens_count', 'test_sequence'], as_index=False).agg(
    test_sequence_count=('batch_size_groups', 'size'), all_batch_size_groupings=('batch_size_groups', get_unique_dicts)
)


# Helper function to determine dominant batch size
# 1. Collects all batch_size:count pairs from the group's dictionaries
# 2. Finds the maximum count value across all batch sizes
# 3. If multiple batch sizes share the max count: Returns the numerically largest batch size
def get_dominant(groupings: List[Dict[int, int]]) -> Optional[int]:
    if not groupings:
        return None

    # Flatten all dictionaries into key-value pairs
    all_items: List[Tuple[int, int]] = []
    for d in groupings:
        all_items.extend(d.items())

    if not all_items:
        return None

    # Find maximum count value
    max_count = max(count for _, count in all_items)

    # Get keys with max count, then return max key if tie
    candidate_keys = [key for key, count in all_items if count == max_count]
    return max(candidate_keys)  # Returns highest batch size if tie


# Add dominant_batch_size column
summary['dominant_batch_size'] = summary['all_batch_size_groupings'].apply(get_dominant)

# 6. Save summary dataframe to CSV
summary.to_csv(summary_output_filename, index=False)


# 7. System Throughput vs Batch Size chart
# Formula: (total_tokens * batch_size) / (TTFT + (output_tokens / completion_tokens_after_first_per_sec))
df['system_throughput'] = (
    (df['total_tokens_count'] * df['batch_size'])
    / (df['time_to_first_token'] + (df['completion_tokens_count'] / df['completion_tokens_after_first_per_sec']))
)

# Aggregate mean system throughput per batch size
throughput_by_batch = (
    df.groupby('batch_size')['system_throughput']
    .mean()
    .reset_index()
    .sort_values('batch_size')
)

# Derive prompt/completion token label from filename (e.g. "4k-4k-dyt-rodrigo.csv" -> "4k/4k")
base_name = os.path.basename(input_filename)
parts = base_name.split('-')
token_label = f'{parts[0]}/{parts[1]}' if len(parts) >= 2 else ''

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    throughput_by_batch['batch_size'],
    throughput_by_batch['system_throughput'],
    marker='o',
    color='steelblue',
    linewidth=2,
    markersize=8,
)

# Annotate each data point with its value
for _, row in throughput_by_batch.iterrows():
    ax.annotate(
        f"{row['system_throughput']:.1f}",
        xy=(row['batch_size'], row['system_throughput']),
        xytext=(-18, 8),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
    )

ax.set_title(f'System Throughput vs Batch Size {token_label}', fontsize=14)
ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('System Throughput (Tokens per Second)', fontsize=12)
ax.set_xticks(throughput_by_batch['batch_size'])
ax.grid(True, linestyle='--', alpha=0.6)

chart_output_filename = results_dir + os.path.splitext(base_name)[0] + '_throughput_vs_batch_size.png'
fig.tight_layout()
fig.savefig(chart_output_filename, dpi=150)
plt.close(fig)
print(f'Chart saved to {chart_output_filename}')
