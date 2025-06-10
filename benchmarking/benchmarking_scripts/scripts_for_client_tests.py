
import os
# Get the absolute path of my_project
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

import sys

benchmarking_dir = "../"
sys.path.append(benchmarking_dir)
sys.path.append(benchmarking_dir + '../')

import yaml
from typing import Dict, Any
from dotenv import load_dotenv

from synthetic_performance_eval_script import create_consolidated_results
from benchmarking.src.performance_evaluation import RealWorkLoadPerformanceEvaluator, SyntheticPerformanceEvaluator

def run_test_1(output_files_dir: str, consolidated_results_dir: str, consolidated_results_file_name: str):
    
    user_metadata: Dict[str, Any] = {}
    sampling_params: Dict[str, Any] = {}
    
    # run configs
    model_name = 'Meta-Llama-3.3-70B-Instruct'
    llm_api = 'sncloud'
    input_tokens = [128]
    output_tokens = [1000]
    num_requests = 1_000_000
    qps = [5, 10, 20, 40, 50, 60, 70, 80, 90, 100]
    timeout = 60
    # qps = [0.5, 1]
    # timeout = 5
    
    for input_token in input_tokens:
        for output_token in output_tokens:
            for qps_value in qps:
                print(f"Running test 1: Input Tokens: {input_token}, Output Tokens: {output_token}, QPS: {qps_value}")
                
                # Instantiate evaluator
                user_metadata['model_idx'] = 0  # no need to change
                evaluator = RealWorkLoadPerformanceEvaluator(
                    multimodal_image_size='na',
                    model_name=model_name,
                    results_dir=os.path.expanduser(consolidated_results_dir),
                    qps=qps_value,
                    timeout=timeout,
                    user_metadata=user_metadata,
                    llm_api=llm_api,
                )
                
                # Run performance evaluation
                _, _ = evaluator.run_benchmark(
                    num_input_tokens=input_token,
                    num_output_tokens=output_token,
                    num_requests=num_requests,
                    sampling_params=sampling_params,
                )
                
    create_consolidated_results(output_files_dir=output_files_dir, consolidated_results_dir=consolidated_results_dir, run_name=consolidated_results_file_name)
    print("Test 1 completed successfully.")
                
def run_test_2(output_files_dir: str, consolidated_results_dir: str, consolidated_results_file_name: str):
    
    user_metadata: Dict[str, Any] = {}
    sampling_params: Dict[str, Any] = {}
    
    # run configs
    model_name = 'Meta-Llama-3.3-70B-Instruct'
    llm_api = 'sncloud'
    input_tokens = [128]
    output_tokens = [1000]
    num_requests = 1_000_000
    concurrent_requests = [100, 100, 100, 100, 100]
    timeout = 60
    # concurrent_requests = [1, 2]
    # timeout = 5
    
    for input_token in input_tokens:
        for output_token in output_tokens:
            for concurrent_requests_value in concurrent_requests:
                print(f"Running test 2: Input Tokens: {input_token}, Output Tokens: {output_token}, Concurrent requests: {concurrent_requests_value}")
                
                # Instantiate evaluator
                user_metadata['model_idx'] = 0  # no need to change
                evaluator = SyntheticPerformanceEvaluator(
                    multimodal_image_size='na',
                    model_name=model_name,
                    results_dir=os.path.expanduser(output_files_dir),
                    num_concurrent_requests=concurrent_requests_value,
                    timeout=timeout,
                    user_metadata=user_metadata,
                    llm_api=llm_api,
                )

                # Run performance evaluation
                _, _ = evaluator.run_benchmark(
                    num_input_tokens=input_token,
                    num_output_tokens=output_token,
                    num_requests=num_requests,
                    sampling_params=sampling_params,
                )
                
    create_consolidated_results(output_files_dir=output_files_dir, consolidated_results_dir=consolidated_results_dir, run_name=consolidated_results_file_name)
    print("Test 2 completed successfully.")

def run_test_3(output_files_dir: str, consolidated_results_dir: str, consolidated_results_file_name: str):
    
    user_metadata: Dict[str, Any] = {}
    sampling_params: Dict[str, Any] = {}
    
    # run configs
    model_name = 'Meta-Llama-3.3-70B-Instruct'
    llm_api = 'sncloud'
    input_tokens = [128]
    output_tokens = [1000]
    qps = [1]
    num_requests = 1_000_000
    timeout = 3600
    # timeout = 5
    
    for input_token in input_tokens:
        for output_token in output_tokens:
            for qps_value in qps:
                print(f"Running test 3: Input Tokens: {input_token}, Output Tokens: {output_token}, QPS: {qps_value}")
                
                # Instantiate evaluator
                user_metadata['model_idx'] = 0  # no need to change
                evaluator = RealWorkLoadPerformanceEvaluator(
                    multimodal_image_size='na',
                    model_name=model_name,
                    results_dir=os.path.expanduser(output_files_dir),
                    qps=qps_value,
                    timeout=timeout,
                    user_metadata=user_metadata,
                    llm_api=llm_api,
                )
                
                # Run performance evaluation
                _, _ = evaluator.run_benchmark(
                    num_input_tokens=input_token,
                    num_output_tokens=output_token,
                    num_requests=num_requests,
                    sampling_params=sampling_params,
                )
    
    create_consolidated_results(output_files_dir=output_files_dir, consolidated_results_dir=consolidated_results_dir, run_name=consolidated_results_file_name)
    print("Test 3 completed successfully.")
    
def run_test_4(output_files_dir: str, consolidated_results_dir: str, consolidated_results_file_name: str):
    
    user_metadata: Dict[str, Any] = {}
    sampling_params: Dict[str, Any] = {}
    
    # run configs
    model_name = 'Meta-Llama-3.3-70B-Instruct'
    llm_api = 'sncloud'
    input_tokens = [128]
    output_tokens = [1000]
    concurrent_requests = [100]
    timeout = 600
    num_requests = 100
    # num_requests = 20
    
    
    for input_token in input_tokens:
        for output_token in output_tokens:
            for concurrent_requests_value in concurrent_requests:
                print(f"Running test 4: Input Tokens: {input_token}, Output Tokens: {output_token}, Concurrent requests: {concurrent_requests_value}")
                
                # Instantiate evaluator
                user_metadata['model_idx'] = 0  # no need to change
                evaluator = SyntheticPerformanceEvaluator(
                    multimodal_image_size='na',
                    model_name=model_name,
                    results_dir=os.path.expanduser(output_files_dir),
                    num_concurrent_requests=concurrent_requests_value,
                    timeout=timeout,
                    user_metadata=user_metadata,
                    use_multiple_prompts=True,
                    llm_api=llm_api,
                )

                # Run performance evaluation
                _, _ = evaluator.run_benchmark(
                    num_input_tokens=input_token,
                    num_output_tokens=output_token,
                    num_requests=num_requests,
                    sampling_params=sampling_params,
                )
                
    create_consolidated_results(output_files_dir=output_files_dir, consolidated_results_dir=consolidated_results_dir, run_name=consolidated_results_file_name)
    print("Test 4 completed successfully.")

if __name__ == '__main__':
    load_dotenv(os.path.join(project_root, '.env'), override=True)

    run_test_1(output_files_dir=f"{project_root}/benchmarking/data/results/sett/test_1", 
               consolidated_results_dir=f"{project_root}/benchmarking/data/results/sett/test_1",
               consolidated_results_file_name="consolited_resulst-test_1")
    
    # run_test_2(output_files_dir=f"{project_root}/benchmarking/data/results/sett/test_2", 
    #            consolidated_results_dir=f"{project_root}/benchmarking/data/results/sett/test_2",
    #            consolidated_results_file_name="consolited_resulst-test_2")
    
    # run_test_3(output_files_dir=f"{project_root}/benchmarking/data/results/sett/test_3", 
    #            consolidated_results_dir=f"{project_root}/benchmarking/data/results/sett/test_3",
    #            consolidated_results_file_name="consolited_resulst-test_3")
    
    # run_test_4(output_files_dir=f"{project_root}/benchmarking/data/results/sett/test_4", 
    #            consolidated_results_dir=f"{project_root}/benchmarking/data/results/sett/test_4",
    #            consolidated_results_file_name="consolited_resulst-test_4")
    