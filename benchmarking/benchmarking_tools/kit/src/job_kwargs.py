"""Maps a job dict (from `evaluator.py`'s `vars(argparse.Namespace)`, or a job dict parsed from
the bundles module's jobs YAML) into the constructor/run kwargs for the Kit's own evaluator
classes (`CustomPerformanceEvaluator`, `SyntheticPerformanceEvaluator`,
`RealWorkLoadPerformanceEvaluator`), keyed on `mode`.

One function serves both callers so the mode -> kwargs mapping is never duplicated between the
CLI and the bundles runner.
"""

import argparse
import json
from typing import Any, Dict, Tuple

MODES = ('custom', 'synthetic', 'real_workload')


def _sampling_params(job: Dict[str, Any]) -> Dict[str, Any]:
    sampling_params = job.get('sampling_params', {})
    if isinstance(sampling_params, str):
        return dict(json.loads(sampling_params)) if sampling_params else {}
    return dict(sampling_params)


def _common_constructor_kwargs(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'model_name': job.get('model_name'),
        'results_dir': job.get('results_dir'),
        'timeout': job.get('timeout', 600),
        'user_metadata': job.get('user_metadata', {}),
        'llm_api': job.get('llm_api', 'sncloud'),
        'use_debugging_mode': job.get('use_debugging_mode', False),
        'num_warmup_requests': job.get('num_warmup_requests', 0),
    }


def build_job_kwargs(job: Dict[str, Any], mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sampling_params = _sampling_params(job)
    common = _common_constructor_kwargs(job)

    if mode == 'custom':
        constructor_kwargs = {
            **common,
            'num_concurrent_requests': job.get('num_concurrent_requests', 10),
            'input_file_path': job['input_file_path'],
            'save_response_texts': job.get('save_llm_responses', False),
        }
        run_kwargs = {'sampling_params': sampling_params}

    elif mode == 'synthetic':
        constructor_kwargs = {
            **common,
            'multimodal_image_size': job.get('multimodal_image_size', 'na'),
            'num_concurrent_requests': job.get('num_concurrent_requests', 10),
            'use_multiple_prompts': job.get('use_multiple_prompts', False),
            'save_response_texts': job.get('save_llm_responses', False),
        }
        run_kwargs = {
            'num_input_tokens': job.get('num_input_tokens', 550),
            'num_output_tokens': job.get('num_output_tokens', 150),
            'num_requests': job.get('num_requests', 10),
            'sampling_params': sampling_params,
        }

    elif mode == 'real_workload':
        constructor_kwargs = {
            **common,
            'multimodal_image_size': job.get('multimodal_image_size', 'na'),
            'qps': job.get('qps', 0.5),
            'qps_distribution': job.get('qps_distribution', 'constant'),
            'use_multiple_prompts': job.get('use_multiple_prompts', False),
            'save_response_texts': job.get('save_llm_responses', False),
        }
        run_kwargs = {
            'num_input_tokens': job.get('num_input_tokens', 550),
            'num_output_tokens': job.get('num_output_tokens', 150),
            'num_requests': job.get('num_requests', 10),
            'sampling_params': sampling_params,
        }

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {MODES}.")

    return constructor_kwargs, run_kwargs


def register_cli_args(parser: argparse.ArgumentParser, mode: str) -> None:
    """Adds this mode's argparse args. Common args (--results-dir, --llm-api, --timeout,
    --metadata, --sampling-params, --use-debugging-mode, --num-warmup-requests) are registered
    once by `evaluator.py` itself, shared across every mode.
    """
    from benchmarking.benchmarking_tools.kit.src.executor_base import str2bool

    if mode == 'custom':
        parser.add_argument(
            '--num-concurrent-requests',
            type=int,
            default=10,
            help='The number of concurrent requests used to send requests. (default: %(default)s)',
        )
        parser.add_argument(
            '--model-names',
            type=str,
            required=True,
            help='The name of the model(s) to use for this performance evaluation.',
        )
        parser.add_argument(
            '--input-file-path',
            type=str,
            required=True,
            help='The absolute path to the dataset to be used for running the custom performance evaluation.',
        )
        parser.add_argument(
            '--save-llm-responses',
            type=str2bool,
            required=False,
            default=False,
            help='Whether to save the llm responses to an output JSONL file. (default: %(default)s)',
        )

    elif mode == 'synthetic':
        parser.add_argument(
            '--num-concurrent-requests',
            type=int,
            default=10,
            help='The number of concurrent requests used to send requests. (default: %(default)s)',
        )
        parser.add_argument(
            '--model-names',
            type=str,
            required=True,
            help='The name of the models to use for this performance evaluation.',
        )
        parser.add_argument(
            '--multimodal-image-size',
            choices=['na', 'small', 'medium', 'large'],
            required=True,
            default='na',
            help="The image size to select if a vision model is going to be evaluated.\
                If no multimodal model will be used, select 'na'. (default: %(default)s)",
        )
        parser.add_argument(
            '--num-input-tokens',
            type=int,
            default=550,
            help="""The number of tokens to include in the prompt for each request made from the synthetic
                dataset. (default: %(default)s)""",
        )
        parser.add_argument(
            '--num-output-tokens',
            type=int,
            default=150,
            help="""The number of tokens to generate from each llm request. This is the `max_tokens` param for the
                completions API. (default: %(default)s)""",
        )
        parser.add_argument(
            '--num-requests',
            type=int,
            default=10,
            help="""The number of requests to make from the synthetic dataset. Note that it is possible for the test
                to timeout first. (default: %(default)s)""",
        )
        parser.add_argument(
            '--use-multiple-prompts',
            type=str2bool,
            required=True,
            default=False,
            help="""Whether to use multiple prompts selected randomly from prompt file.
                Only works on text instruct models. (default: %(default)s)""",
        )
        parser.add_argument(
            '--save-llm-responses',
            type=str2bool,
            required=False,
            default=False,
            help="Whether to save the kit's llm responses to an output JSONL file. (default: %(default)s)",
        )

    elif mode == 'real_workload':
        parser.add_argument(
            '--qps',
            type=float,
            default=0.5,
            help='The number of queries per second processed for a real workload. (default: %(default)s)',
        )
        parser.add_argument(
            '--qps-distribution',
            choices=['constant', 'uniform', 'exponential'],
            default='constant',
            help="The name of the distribution to use for a real workload. (default: %(default)s)",
        )
        parser.add_argument(
            '--model-names',
            type=str,
            required=True,
            help='The name of the models to use for this performance evaluation.',
        )
        parser.add_argument(
            '--multimodal-image-size',
            choices=['na', 'small', 'medium', 'large'],
            required=True,
            help="The image size to select if a vision model is going to be evaluated.\
                If no multimodal model will be used, select 'na'.",
        )
        parser.add_argument(
            '--num-input-tokens',
            type=int,
            default=550,
            help="""The number of synthetic tokens to include in the prompt for each request made.
                (default: %(default)s)""",
        )
        parser.add_argument(
            '--num-output-tokens',
            type=int,
            default=150,
            help="""The number of tokens to generate from each llm request. This is the `max_tokens` param for the
                completions API. (default: %(default)s)""",
        )
        parser.add_argument(
            '--num-requests',
            type=int,
            default=10,
            help="""The number of requests to make. Note that it is possible for the test
                to timeout first. (default: %(default)s)""",
        )
        parser.add_argument(
            '--use-multiple-prompts',
            type=str2bool,
            default=False,
            help="""Whether to use multiple prompts selected randomly from prompt file, instead of
                repeating one. Only works on text instruct models. (default: %(default)s)""",
        )
        parser.add_argument(
            '--save-llm-responses',
            type=str2bool,
            default=False,
            help="Whether to save the kit's llm responses to an output JSONL file. (default: %(default)s)",
        )

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {MODES}.")


def register_probe_args(parser: argparse.ArgumentParser) -> None:
    """Registers the union of every mode's discriminating args, all optional, so `infer_mode`
    can read them before the mode itself is known. Kept separate from `register_cli_args` because
    that function's args are mode-specific and `--required`; these are deliberately loose.

    Deliberately excludes `--qps`/`--qps-distribution`: a caller that always passes every flag
    unconditionally (e.g. a single shell script covering all three modes) would otherwise force
    real_workload every time just because those flags are present with their default values.
    `--num-concurrent-requests` vs `--num-requests` is the one signal that stays meaningful whether
    or not the caller passes every flag on every invocation.
    """
    parser.add_argument('--input-file-path', type=str, default=None)
    parser.add_argument('--num-concurrent-requests', type=int, default=None)
    parser.add_argument('--num-requests', type=int, default=None)


def infer_mode(probe_args: argparse.Namespace) -> str:
    """Picks a mode from the raw arguments the user passed, so the CLI caller never has to name
    `--mode` itself: an `--input-file-path` means you're bringing your own dataset (custom);
    otherwise, requests fired one-for-one with the concurrency (or with either left unset) means a
    single concurrent burst (synthetic), and any other combination means requests should instead be
    paced over time (real_workload).
    """
    if probe_args.input_file_path:
        return 'custom'
    ncr = probe_args.num_concurrent_requests
    nr = probe_args.num_requests
    if ncr is not None and nr is not None and ncr != nr:
        return 'real_workload'
    return 'synthetic'
