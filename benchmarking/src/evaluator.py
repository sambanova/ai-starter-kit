import argparse
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)


from dotenv import load_dotenv


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
    from benchmarking.src.performance_evaluation import (
        CustomPerformanceEvaluator,
        RealWorkLoadPerformanceEvaluator,
        SyntheticPerformanceEvaluator,
    )

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
        choices=['custom', 'synthetic', 'real_workload'],
        required=True,
        help="""Run mode for the performance evaluation. You have three options to choose from - 'custom', 'synthetic'\
            or 'real workload'.
            
            Custom: You provide your own dataset via the `input-file-path argument. We will run the performance 
                    evaluation with the provided dataset.
                
            Synthetic: You provide the number of input tokens, number of output tokens, and number of requests. We 
                    will generate n input prompts for you where n is the number of requests specified.
            
            Real Workload: You provide the queries per second (QPS), QPS distribution, number of requests, number of
                    input and output tokens. We will generate requests randomly according to the distribution specified
                    and rest of parameters.""",
    )

    # Required Common Argurments
    parser.add_argument('--results-dir', type=str, required=True, help='The output directory to save the results to.')

    parser.add_argument(
        '--llm-api',
        type=str,
        required=True,
        default='sncloud',
        help="The LLM API type. It could be either 'sambastudio' or 'sncloud'. (default: %(default)s)",
    )

    # Optional Common Arguments

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
        help='Whether to use or not the debug mode. WARNING: Debug mode will provide more detailed response at the cost of increased latency. (default: %(default)s)',
    )

    args, _ = parser.parse_known_args()

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
            '--num-concurrent-requests',
            type=int,
            default=10,
            help='The number of concurrent requests used to send requests. (default: %(default)s)',
        )

        parser.add_argument(
            '--model-name',
            type=str,
            required=True,
            help='The name of the model to use for this performance evaluation.',
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

        # Parse arguments and instantiate evaluator
        args = parser.parse_args()
        custom_evaluator = CustomPerformanceEvaluator(
            model_name=args.model_name,
            results_dir=args.results_dir,
            num_concurrent_requests=args.num_concurrent_requests,
            timeout=args.timeout,
            user_metadata=user_metadata,
            input_file_path=args.input_file_path,
            save_response_texts=args.save_llm_responses,
            use_debugging_mode=args.use_debugging_mode,
            llm_api=args.llm_api,
        )

        # Run performance evaluation
        custom_evaluator.run_benchmark(sampling_params=json.loads(args.sampling_params))

    # Synthetic dataset evaluation path
    elif args.mode == 'synthetic':
        # Synthetic dataset specific arguments
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
            help="The image size to select if a vision model is going to be evaluated.\
                If no multimodal model will be used, select 'na'.",
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
            help='Whether to save the llm responses to an output JSONL file. (default: %(default)s)',
        )

        # Parse arguments and instantiate evaluator
        args = parser.parse_args()
        model_names = args.model_names.strip().split()

        # running perf eval for multiple bundle models
        for model_idx, model_name in enumerate(model_names):
            user_metadata['model_idx'] = model_idx
            # set synthetic evaluator
            synthetic_evaluator = SyntheticPerformanceEvaluator(
                multimodal_image_size=args.multimodal_image_size,
                model_name=model_name,
                results_dir=args.results_dir,
                num_concurrent_requests=args.num_concurrent_requests,
                timeout=args.timeout,
                user_metadata=user_metadata,
                use_multiple_prompts=args.use_multiple_prompts,
                save_response_texts=args.save_llm_responses,
                use_debugging_mode=args.use_debugging_mode,
                llm_api=args.llm_api,
            )

            # Run performance evaluation
            synthetic_evaluator.run_benchmark(
                num_input_tokens=args.num_input_tokens,
                num_output_tokens=args.num_output_tokens,
                num_requests=args.num_requests,
                sampling_params=json.loads(args.sampling_params),
            )

    # Real workload evaluation path
    elif args.mode == 'real_workload':
        parser.add_argument(
            '--qps',
            type=float,
            default=0.5,
            help='The number of queries per second processed for a real workload. (default: %(default)s)',
        )

        parser.add_argument(
            '--qps-distribution',
            type=str,
            default='constant',
            help='The name of the distribution to use for a real workload. Possible values are "constant",\
                "uniform", "exponential". (default: %(default)s)',
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

        args = parser.parse_args()
        model_names = args.model_names.strip().split()

        # running perf eval for multiple bundle models
        for model_idx, model_name in enumerate(model_names):
            user_metadata['model_idx'] = model_idx
            # set real workload evaluator
            real_workload_evaluator = RealWorkLoadPerformanceEvaluator(
                multimodal_image_size=args.multimodal_image_size,
                model_name=model_name,
                results_dir=args.results_dir,
                qps=args.qps,
                qps_distribution=args.qps_distribution,
                timeout=args.timeout,
                user_metadata=user_metadata,
                use_debugging_mode=args.use_debugging_mode,
                llm_api=args.llm_api,
            )

            # Run performance evaluation
            real_workload_evaluator.run_benchmark(
                num_input_tokens=args.num_input_tokens,
                num_output_tokens=args.num_output_tokens,
                num_requests=args.num_requests,
                sampling_params=json.loads(args.sampling_params),
            )

    else:
        raise Exception("Performance eval mode not valid. Available values are 'custom', 'synthetic', 'real_workload'")


if __name__ == '__main__':
    load_dotenv('../.env', override=True)
    env_vars = dict(os.environ)

    main()
