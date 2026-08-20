"""Shared helpers for the Kit's benchmark executor classes."""

import argparse
from typing import Any, Callable, Optional

import pandas as pd

# (completed, total, phase_label) — 3-arg, matching Kit's existing convention
# (`self.ui_progress_bar(len(progress), num_requests, self.progress_phase_label)`).
ProgressCallback = Callable[[int, int, str], None]


def str2bool(value: Any) -> bool:
    """Shared argparse type= helper for boolean CLI flags."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseExecutorMixin:
    """Shared default `get_results_dataframe`, mixed into `BasePerformanceEvaluator`. Requires
    the including class to set `self.individual_responses_file_path` once `run_benchmark` has
    completed.
    """

    individual_responses_file_path: Optional[str] = None

    def get_results_dataframe(self) -> pd.DataFrame:
        if self.individual_responses_file_path is None:
            raise Exception('No results available. Run benchmark first.')
        return pd.read_json(self.individual_responses_file_path)
