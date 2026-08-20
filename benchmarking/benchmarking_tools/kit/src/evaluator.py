import argparse
import os
import sys
from typing import Any, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
benchmarking_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
repo_dir = os.path.abspath(os.path.join(benchmarking_dir, '..'))

sys.path.append(benchmarking_dir)
sys.path.append(repo_dir)


from dotenv import load_dotenv

from benchmarking.benchmarking_tools.kit.src.executor_base import str2bool
from benchmarking.benchmarking_tools.kit.src.job_kwargs import (
    MODES,
    build_job_kwargs,
    infer_mode,
    register_cli_args,
    register_probe_args,
)
from benchmarking.benchmarking_tools.kit.src.performance_evaluation import (
    CustomPerformanceEvaluator,
    RealWorkLoadPerformanceEvaluator,
    SyntheticPerformanceEvaluator,
)

EXECUTOR_FOR_MODE = {
    'custom': CustomPerformanceEvaluator,
    'synthetic': SyntheticPerformanceEvaluator,
    'real_workload': RealWorkLoadPerformanceEvaluator,
}


def _parse_user_metadata(metadata_arg: str) -> Dict[str, Any]:
    user_metadata: Dict[str, Any] = {}
    if metadata_arg:
        for item in metadata_arg.split(','):
            key, value = item.split('=')
            user_metadata[key] = value
    return user_metadata


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Args shared by every --mode."""
    parser.add_argument('--results-dir', type=str, required=True, help='The output directory to save the results to.')

    parser.add_argument(
        '--llm-api',
        type=str,
        required=True,
        default='sncloud',
        help="The LLM API type. Currently only supporting 'sncloud'. (default: %(default)s)",
    )

    parser.add_argument(
        '--timeout',
        type=int,
        required=False,
        default=600,
        help='The amount of time to run the load test for. (default: %(default)s)',
    )

    parser.add_argument(
        '--metadata',
        type=str,
        required=False,
        default='',
        help="""A comma separated list of metadata to include in the results, e.g. name=foo,bar=1. These will be added
            to the metadata field of the results.""",
    )

    parser.add_argument(
        '--sampling-params',
        type=str,
        required=False,
        default='{}',
        help='Sampling parameters to send with the each request to the LLM API. (default: %(default)s)',
    )

    parser.add_argument(
        '--use-debugging-mode',
        type=str2bool,
        required=False,
        default=False,
        help='Whether to use or not the debug mode. \
            WARNING: Debug mode will provide more detailed response at the cost of increased latency. \
            (default: %(default)s)',
    )

    parser.add_argument(
        '--num-warmup-requests',
        type=int,
        required=False,
        default=0,
        help='Number of throwaway warm-up requests to send before the measured run. Warm-up requests are \
            sent at the test concurrency and their results are discarded, absorbing cold-start and batch \
            ramp-up costs so they do not skew the reported metrics. 0 disables warm-up. (default: %(default)s)',
    )


def _run_job(mode: str, job: Dict[str, Any]) -> Any:
    """Dispatches a single job (one model/config combo): builds constructor/run kwargs via
    `build_job_kwargs`, instantiates the evaluator for this mode, and runs it.
    """
    constructor_kwargs, run_kwargs = build_job_kwargs(job, mode)
    executor_cls = EXECUTOR_FOR_MODE[mode]
    executor = executor_cls(**constructor_kwargs)
    return executor.run_benchmark(**run_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="""Run a token throughput and latency benchmark using the Kit's own native evaluator.
            --mode is optional and, if omitted, is inferred from whichever other arguments you give:

            custom: You provide your own dataset via the `input-file-path` argument. We will run the performance
                evaluation with the provided dataset. Inferred whenever `--input-file-path` is given.

            synthetic: You provide the number of input tokens, number of output tokens, and number of requests.
                We will generate n input prompts for you where n is the number of requests specified. Inferred
                whenever `--num-concurrent-requests` equals `--num-requests` (or either is omitted).

            real_workload: You provide the queries per second (QPS), QPS distribution, number of requests, number
                of input and output tokens. We will generate requests randomly according to the distribution
                specified and the rest of the parameters. Inferred whenever `--num-concurrent-requests` and
                `--num-requests` are both given and differ."""
    )

    parser.add_argument(
        '--mode',
        required=False,
        default=None,
        choices=MODES,
        help='Workload shape to run: custom/synthetic/real_workload. Optional: if omitted, it is inferred '
        'from whichever other arguments are given (custom from --input-file-path, real_workload from '
        'unequal --num-concurrent-requests/--num-requests, synthetic otherwise).',
    )

    _add_common_args(parser)

    args, _ = parser.parse_known_args()

    if args.mode is None:
        probe_parser = argparse.ArgumentParser(add_help=False)
        register_probe_args(probe_parser)
        probe_args, _ = probe_parser.parse_known_args()
        args.mode = infer_mode(probe_args)
        print(f"[evaluator] --mode not given; inferred --mode {args.mode!r}.")

    # The final `parse_known_args()` call below re-derives `args` from scratch off the raw argv,
    # which would otherwise lose a mode inferred above (it was never actually in argv). Making it
    # the parser's default means that re-derivation lands on it too, whether inferred or explicit.
    parser.set_defaults(mode=args.mode)

    register_cli_args(parser, args.mode)
    args, unrecognized = parser.parse_known_args()
    if unrecognized:
        print(f'[evaluator] ignoring arguments not used by --mode {args.mode}: {unrecognized}')

    user_metadata = _parse_user_metadata(args.metadata)

    for model_idx, model_name in enumerate(args.model_names.strip().split()):
        job = {
            **vars(args),
            'model_name': model_name,
            'user_metadata': {**user_metadata, 'model_idx': model_idx},
        }
        _run_job(args.mode, job)


if __name__ == '__main__':
    # Resolved from the file's own location (repo_dir, computed above), not the caller's cwd --
    # this script is invoked both as `python evaluator.py` (from within kit/, e.g. quickstart.sh)
    # and as `python benchmarking_tools/kit/evaluator.py` (from benchmarking/), so a cwd-relative
    # '../.env' would only work for one of those.
    load_dotenv(os.path.join(repo_dir, '.env'), override=True)
    env_vars = dict(os.environ)

    main()
