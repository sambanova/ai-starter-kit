import sys
import time
from typing import Any, Dict

sys.path.append('../')
sys.path.append('../src')
sys.path.append('../prompts')
sys.path.append('../src/llmperf')

import pandas as pd

from benchmarking.src.llmperf import llmperf_utils
from benchmarking.src.performance_evaluation import SyntheticPerformanceEvaluator

# This is a script to run multiple models for different benchmarking paramater combinations.
# Here's an example of how to set the paramteres needed before running the script.
#
# SambaNova Cloud example:
#
# General parameters:
# model_names = ['Meta-Llama-3.1-70B-Instruct', 'Meta-Llama-3.1-8B-Instruct', ...]
# llm_api = 'sncloud'
# results_dir = 'data/results/path/to/studio/name'
#
# Additional parameters:
# timeout = 60000
# sampling_params = {}
# user_metadata = {}
# time_delay = 60
#
# Perf parameters:
# num_input_tokens = [100, 1_000, 10_000, 50_000, 100_000]
# num_output_tokens = [1_000]
# num_concurrent_requests = [1, 10, 100]
# ratio = 5
#
# To run the script, go to the kit's root and run:

# General parameters:
model_names = ['Meta-Llama-3.3-70B-Instruct']  # add more models if necessary
llm_api = 'sncloud'  # it could be sncloud or sambastudio
results_dir = 'data/results/amit_llama3.3_70b/'  # set the path where results will be saved

# Additional parameters:
timeout = 60000
sampling_params: Dict[str, Any] = {}
user_metadata: Dict[str, Any] = {}
time_delay = 10  # delayed time in seconds between runs

# Perf parameters:
# norris configs
# 100, 100, 500, 2_000, 1_000, 2_000, 100, 5_000, 20_000 
# 100, 2_000, 2_000, 2_000, 1_000, 100, 4_000, 500, 2_000

# amit configs
# 1024, 2048, 4096, 8192, 16384, 2048, 2048, 2048, 2048, 2048, 2048, 1024, 2048, 4096, 8192, 16384, 1024, 2048, 4096, 8192, 16384
# 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512
# 1, 1, 1, 1, 1, 2, 4, 8, 16, 32, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128

num_input_tokens = [2048, 4096, 8192, 16384, 1024, 2048, 4096, 8192, 16384]  # add more input token numbers if needed
num_output_tokens = [256, 256, 256, 256, 512, 512, 512, 512, 512]  # add more output token numbers if needed
num_concurrent_requests = [128, 128, 128, 128, 128, 128, 128, 128, 128]  # add more concurrent requests token numbers if needed
ratio = 1  # ratio between num_requests/concurrent_requests
# e.g. means: for num_concurrent_requests = 10, there will be num_requests = 50

df_all_summary_results = pd.DataFrame()
for model_idx, model_name in enumerate(model_names):
    for input_tokens, output_tokens, concurrent_requests in zip(num_input_tokens, num_output_tokens, num_concurrent_requests):
        num_requests = concurrent_requests * ratio
        print(
            f'running model_name {model_name}, input_tokens {input_tokens}, output_tokens {output_tokens},'
            f'concurrent_requests {concurrent_requests}, num_requests {num_requests}'
        )
        user_metadata['model_idx'] = model_idx
        # Instantiate evaluator
        evaluator = SyntheticPerformanceEvaluator(
            model_name=model_name,
            results_dir=results_dir,
            num_concurrent_requests=concurrent_requests,
            timeout=timeout,
            user_metadata=user_metadata,
            llm_api=llm_api,
        )

        # Run performance evaluation
        model_results_summary, model_results_per_request = evaluator.run_benchmark(
            num_input_tokens=input_tokens,
            num_output_tokens=output_tokens,
            num_requests=num_requests,
            sampling_params=sampling_params,
        )

        flatten_model_results_summary = llmperf_utils.flatten_dict(model_results_summary)
        filtered_flatten_model_results_summary = {
            key: value for key, value in flatten_model_results_summary.items() if key not in ['model']
        }
        df_model_results_summary = pd.DataFrame.from_dict(
            filtered_flatten_model_results_summary,
            orient='index',
            columns=[flatten_model_results_summary['model']],
        )

        df_all_summary_results = pd.concat([df_all_summary_results, df_model_results_summary], axis=1)
        time.sleep(time_delay)
