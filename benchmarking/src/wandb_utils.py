import logging
import os
from typing import Any, Dict, List, Optional

import wandb

logger = logging.getLogger(__name__)


def get_wandb_api_key() -> Optional[str]:
    """Retrieve the W&B API key from environment or netrc.

    Returns:
        The API key if found, otherwise None.
    """
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    if wandb_api_key:
        return wandb_api_key

    try:
        import netrc
        netrc_path = os.path.expanduser('~/.netrc')
        netrc_data = netrc.netrc(netrc_path)
        auth = netrc_data.authenticators('api.wandb.ai')
        if auth and len(auth) == 3:
            return auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass

    return None


def init_wandb(
    project_name: str,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Optional[wandb.Run]:
    """Initialize a W&B run.

    Args:
        project_name: W&B project name
        entity: W&B entity (user or organization). If None, uses default.
        name: Run name. If None, uses a generated name.
        config: Run configuration dictionary.
        api_key: W&B API key. If None, attempts to get from environment/netrc.

    Returns:
        wandb.Run object if successful, None otherwise.
    """
    if api_key is None:
        api_key = get_wandb_api_key()

    if not api_key:
        logger.warning('W&B API key not found. Cannot log to W&B.')
        return None

    if not project_name:
        logger.warning('W&B project name not provided. Cannot log to W&B.')
        return None

    try:
        wandb.login(key=api_key)
        run = wandb.init(
            project=project_name,
            entity=entity,
            name=name,
            config=config,
        )
        logger.info(f'W&B run initialized: {run.url}')
        return run
    except Exception as e:
        logger.error(f'Failed to initialize W&B run: {e}')
        return None


def log_summary_metrics(
    run: wandb.Run,
    summary: Dict[str, Any],
    metrics_prefix: str = '',
) -> None:
    """Log summary metrics to W&B.

    Args:
        run: W&B run object
        summary: Summary metrics dictionary
        metrics_prefix: Optional prefix for metric names
    """
    if run is None:
        return

    metrics = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                metric_name = f'{metrics_prefix}{key}_{sub_key}' if metrics_prefix else f'{key}_{sub_key}'
                metrics[metric_name] = sub_value
        else:
            metric_name = f'{metrics_prefix}{key}' if metrics_prefix else key
            metrics[metric_name] = value

    try:
        run.log(metrics)
    except Exception as e:
        logger.error(f'Failed to log summary metrics to W&B: {e}')


def log_individual_request_metrics(
    run: wandb.Run,
    request_metrics: List[Dict[str, Any]],
    step_column: str = 'request_idx',
) -> None:
    """Log individual request metrics to W&B as a table.

    Args:
        run: W&B run object
        request_metrics: List of per-request metric dictionaries
        step_column: Column name to use as the step/index
    """
    if run is None:
        return

    if not request_metrics:
        return

    try:
        import pandas as pd
        df = pd.DataFrame(request_metrics)
        if step_column in df.columns:
            df = df.set_index(step_column)
        run.log({'individual_requests': wandb.Table(dataframe=df)})
    except Exception as e:
        logger.error(f'Failed to log individual request metrics to W&B: {e}')


def finish_wandb_run(run: wandb.Run) -> None:
    """Finish a W&B run.

    Args:
        run: W&B run object
    """
    if run is not None:
        try:
            run.finish()
            logger.info('W&B run finished successfully')
        except Exception as e:
            logger.error(f'Failed to finish W&B run: {e}')