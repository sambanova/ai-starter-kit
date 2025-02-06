import sys
import time
from typing import Any, Dict

sys.path.append('../')
sys.path.append('../src')
sys.path.append('../prompts')
sys.path.append('../src/llmperf')

import pandas as pd

from benchmarking.src.llmperf import llmperf_utils
from benchmarking.src.performance_evaluation import RealWorkLoadPerformanceEvaluator

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
# qps = [0.5, 1, 2]
# qps_distribution = 'constant' # it could be either 'constant', 'uniform' or 'exponential'
# num_requests = [10, 30, 60]
#
# To run the script, go to the kit's root and run:

# General parameters:
model_names = ['Meta-Llama-3.1-70B-Instruct']  # add more models if necessary
llm_api = 'sncloud'  # it could be sncloud or sambastudio
results_dir = 'data/results/path/'  # set the path where results will be saved

# Additional parameters:
timeout = 60000
sampling_params: Dict[str, Any] = {}
user_metadata: Dict[str, Any] = {}
time_delay = 0  # delayed time in seconds between runs

# Perf parameters:
num_input_tokens = [1_000]  # add more input token numbers if needed
num_output_tokens = [1_000]  # add more output token numbers if needed
qps_list = [0.5, 2]  # should be the same number of queries per second items as num_requests
qps_distribution = (
    'exponential'  # queries per second distribution. Available options: 'constant', 'uniform' or 'exponential'
)
num_requests_list = [10, 120]  # should be the same number or requests items as qps

assert len(qps_list) == len(num_requests_list), 'qps_list and num_requests_list should have the same length'

df_all_summary_results = pd.DataFrame()
for model_idx, model_name in enumerate(model_names):
    for input_tokens in num_input_tokens:
        for output_tokens in num_output_tokens:
            for qps, num_requests in zip(qps_list, num_requests_list):
                print(
                    f'running model_name {model_name}, input_tokens {input_tokens}, output_tokens {output_tokens},'
                    f'qps {qps}, qps distribution {qps_distribution}, num_requests {num_requests}'
                )
                user_metadata['model_idx'] = model_idx
                # Instantiate evaluator
                evaluator = RealWorkLoadPerformanceEvaluator(
                    model_name=model_name,
                    results_dir=results_dir,
                    qps=qps,
                    qps_distribution=qps_distribution,
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
