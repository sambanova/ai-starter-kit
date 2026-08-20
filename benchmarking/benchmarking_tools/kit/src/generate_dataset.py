"""Dumps a Kit synthetic dataset to a JSONL file, for the vLLM/aiperf quickstart scripts to
consume as a "custom dataset" input via their own native CLIs.

Reuses `SyntheticPerformanceEvaluator`'s existing tokenizer-bound prompt-building logic
(`SyntheticPromptSource.build_prompt` / `adjust_to_exact_tokens`) instead of re-deriving the
repeat-and-trim-to-token-count logic a second time — no network calls are made; the evaluator is
only constructed to reuse its prompt-generation helpers.

Usage:
    python generate_dataset.py --model-name Meta-Llama-3.3-70B-Instruct --num-requests 20 \
        --num-input-tokens 550 --output-path ./synthetic_dataset.jsonl

    # aiperf's `single_turn` custom-dataset-type expects {"text": ...} instead of {"prompt": ...}:
    python generate_dataset.py --model-name Meta-Llama-3.3-70B-Instruct --num-requests 20 \
        --num-input-tokens 550 --output-path ./synthetic_dataset.jsonl --prompt-key text
"""

import argparse
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
benchmarking_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
repo_dir = os.path.abspath(os.path.join(benchmarking_dir, '..'))
sys.path.append(benchmarking_dir)
sys.path.append(repo_dir)

from benchmarking.benchmarking_tools.kit.src.executor_base import str2bool
from benchmarking.benchmarking_tools.kit.src.performance_evaluation import SyntheticPerformanceEvaluator


def generate_dataset(
    model_name: str,
    num_requests: int,
    num_input_tokens: int,
    output_path: str,
    use_multiple_prompts: bool = False,
    prompt_key: str = 'prompt',
) -> None:
    # results_dir/num_output_tokens are unused (no requests are ever sent) but are required
    # constructor/kwargs by the evaluator's shared interface.
    evaluator = SyntheticPerformanceEvaluator(
        num_concurrent_requests=1,
        use_multiple_prompts=use_multiple_prompts,
        model_name=model_name,
        results_dir=os.path.dirname(os.path.abspath(output_path)) or '.',
    )
    request_configs = evaluator.build_request_configs(
        num_requests=num_requests,
        sampling_params={},
        num_input_tokens=num_input_tokens,
        num_output_tokens=1,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for request_config in request_configs:
            prompt_dict, _token_length = request_config.prompt_tuple
            f.write(json.dumps({prompt_key: prompt_dict['template']}) + '\n')

    print(f'Wrote {num_requests} synthetic prompts to {output_path} (key: "{prompt_key}").')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-name', type=str, required=True, help='Model name (used to pick a tokenizer).')
    parser.add_argument('--num-requests', type=int, required=True, help='Number of prompts to generate.')
    parser.add_argument(
        '--num-input-tokens', type=int, required=True, help='Exact input token count per generated prompt.'
    )
    parser.add_argument('--output-path', type=str, required=True, help='Where to write the generated JSONL file.')
    parser.add_argument(
        '--use-multiple-prompts',
        type=str2bool,
        default=False,
        help="Whether to cycle through the Kit's multiple prompt templates instead of repeating a single one. "
        '(default: %(default)s)',
    )
    parser.add_argument(
        '--prompt-key',
        type=str,
        default='prompt',
        help='JSON key each output line is written under. Use "prompt" for vLLM/Kit custom-dataset ingestion '
        '(default), or "text" for aiperf\'s `single_turn` custom-dataset-type.',
    )
    args = parser.parse_args()

    generate_dataset(
        model_name=args.model_name,
        num_requests=args.num_requests,
        num_input_tokens=args.num_input_tokens,
        output_path=args.output_path,
        use_multiple_prompts=args.use_multiple_prompts,
        prompt_key=args.prompt_key,
    )


if __name__ == '__main__':
    main()
