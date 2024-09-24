import argparse
import json
import os

from dotenv import load_dotenv
from performance_evaluation import CustomPerformanceEvaluator, SyntheticPerformanceEvaluator


def str2bool(value: str) -> bool:
    """Transform str to bool

    Args:
        value (str): input value

    Raises:
        argparse.ArgumentTypeError: raises when value is another type than boolean

    Returns:
        bool: boolean value
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="""Run a token throughput and latency benchmark. You have the option of running in two different 
            modes - 'custom' or 'synthetic'.
            
            Custom: You provide your own dataset via the `input-file-path argument. We will run the performance 
                    evaluation with the provided dataset.
                    
            Synthetic: You provide the number of input tokens, number of output tokens, and number of requests. We 
                    will generate n input prompts for you where n is the number of requests specified."""
    )

    # Distinguish between custom and synthetic dataset runs
    parser.add_argument(
        '--mode',
        choices=['custom', 'synthetic'],
        required=True,
        help="""Run mode for the performance evaluation. You have two options to choose from - 'custom' or 'synthetic'.
            
            Custom: You provide your own dataset via the `input-file-path argument. We will run the performance 
                    evaluation with the provided dataset.
                
            Synthetic: You provide the number of input tokens, number of output tokens, and number of requests. We 
                    will generate n input prompts for you where n is the number of requests specified.""",
    )

    # Required Common Argurments
    parser.add_argument(
        '--model-name', type=str, required=True, help='The name of the model to use for this performance evaluation.'
    )

    parser.add_argument('--results-dir', type=str, required=True, help='The output directory to save the results to.')

    parser.add_argument(
        '--llm-api', 
        type=str, 
        required=True, 
        default="sncloud",
        help="The LLM API type. It could be either 'sambastudio' or 'sncloud'. Default value: 'sncloud'"
    )

    # Optional Common Arguments
    parser.add_argument(
        '--num-workers',
        type=int,
        default=10,
        help='The number of concurrent workers used to send requests. (default: %(default)s)',
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

    args, unknown = parser.parse_known_args()

    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(','):
            key, value = item.split('=')
            user_metadata[key] = value

    # Custom dataset evaluation path
    if args.mode == 'custom':
        # Custom dataset specific arguments
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

        # Parse arguments and instantiate evaluator
        args = parser.parse_args()
        evaluator = CustomPerformanceEvaluator(
            model_name=args.model_name,
            results_dir=args.results_dir,
            num_workers=args.num_workers,
            timeout=args.timeout,
            user_metadata=user_metadata,
            input_file_path=args.input_file_path,
            save_response_texts=args.save_llm_responses,
            llm_api=args.llm_api,
        )

        # Run performance evaluation
        evaluator.run_benchmark(sampling_params=json.loads(args.sampling_params))

    # Synthetic dataset evaluation path
    else:
        # Synthetic dataset specific arguments
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

        # Parse arguments and instantiate evaluator
        args = parser.parse_args()
        evaluator = SyntheticPerformanceEvaluator(
            model_name=args.model_name,
            results_dir=args.results_dir,
            num_workers=args.num_workers,
            timeout=args.timeout,
            user_metadata=user_metadata,
            llm_api=args.llm_api,
        )

        # Run performance evaluation
        evaluator.run_benchmark(
            num_input_tokens=args.num_input_tokens,
            num_output_tokens=args.num_output_tokens,
            num_requests=args.num_requests,
            sampling_params=json.loads(args.sampling_params),
        )


if __name__ == '__main__':
    load_dotenv('../.env', override=True)
    env_vars = dict(os.environ)

    main()
