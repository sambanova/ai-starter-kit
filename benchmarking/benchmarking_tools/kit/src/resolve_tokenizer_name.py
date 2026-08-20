"""Resolves the HuggingFace tokenizer id a native CLI tool (vLLM, aiperf) should use for local
tokenization, given a Kit/API model name -- reuses the Kit's own model registry
(`benchmarking_utils.get_tokenizer_model_name`) instead of requiring each quickstart script to
hand-maintain (and risk drifting from) the same model-name-to-tokenizer mapping.

Usage:
    python resolve_tokenizer_name.py --model-name Meta-Llama-3.3-70B-Instruct
"""

import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
benchmarking_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
repo_dir = os.path.abspath(os.path.join(benchmarking_dir, '..'))
sys.path.append(benchmarking_dir)
sys.path.append(repo_dir)

from benchmarking.benchmarking_utils import get_tokenizer_model_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-name', type=str, required=True, help='Model name as exposed by the endpoint.')
    args = parser.parse_args()
    print(get_tokenizer_model_name(args.model_name))


if __name__ == '__main__':
    main()
