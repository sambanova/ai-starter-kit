import sys
sys.path.append('../../')

import os
import yaml
import asyncio
import time
import statistics
import tqdm

from langchain_community.llms.sambanova import SambaStudio
from transformers import AutoTokenizer

from dotenv import load_dotenv
load_dotenv('../.env', override=True)

# Load Llam2 7B tokenizer
access_token = os.getenv("HUGGINGFACE_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token)
latencies = []

def get_snovastudio_llm(max_tokens_to_generate: int) -> SambaStudio:
    """Instantiate a SambaStudio llm object.

    Args:
        max_tokens_to_generate (int): number of max output tokens.

    Returns:
        SambaStudio: SambaStudio llm object.
    """
    
    temperature = 0.01
    do_sample = False
    llm = SambaStudio(
        model_kwargs={
            "do_sample": do_sample,
            "temperature": temperature,
            "max_tokens_to_generate": max_tokens_to_generate,
            #"stop_sequences": { "type":"str", "value":""},
            # "repetition_penalty": {"type": "float", "value": "1"},
            # "top_k": {"type": "int", "value": "50"},
            # "top_p": {"type": "float", "value": "1"}
        }
    )
    return llm

def get_out_tokens(generated_string: str) -> int:
    """Gets number of output tokens using HuggingFace tokenizer.

    Args:
        generated_string (str): text to get the number of tokens from.

    Returns:
        int: number of tokens.
    """
    
    encoded = tokenizer(generated_string)
    return len(encoded['input_ids'])

def get_artificial_prompt(input_tokens: int) -> tuple[str, int]:
    """Gets an artificial prompt based on the number of input tokens.
    Artificial prompt has a similar number of tokens as input_tokens.

    Args:
        input_tokens (int): number of desired input tokens.

    Returns:
        tuple[str, int]: tuple that contains the artificial prompt and its number of tokens
    """
    
    template = 'I will repeat this sentence many times for testing. '
    template_encoded = tokenizer(template)
    groups = int(input_tokens / (len(template_encoded['input_ids'])-1))

    artificial_prompt = [template*(groups)]
    prompt_encoded = tokenizer(artificial_prompt[0])
    artificial_input_tokens = len(prompt_encoded['input_ids'])

    return artificial_prompt[0], artificial_input_tokens

async def worker(index: int, num_requests: int, in_tokens: int = 2048, max_out_tokens: int = 256) -> None:
    """It's a single worker, which processes the given number of requests, one after the other.

    Args:
        index (int): index of the current worker
        num_requests (int): number of requests to be processed
        in_tokens (int, optional): number of input tokens. Defaults to 2048.
        max_out_tokens (int, optional): number of max ouput tokens. Defaults to 256.
    """
    
    print(f"Processing concurrent worker {index}")
    prompt, artificial_in_tokens = get_artificial_prompt(in_tokens)
    llm = get_snovastudio_llm(max_out_tokens)
    # Sleep some time to offset the threads.
    await asyncio.sleep(0.01*index)

    # for _ in tqdm(range(num_requests)):
    for _ in range(num_requests):
        request_start_time = time.time()
        generated_ouput = llm.invoke(prompt[0])
        latency = time.time() - request_start_time
        out_tokens = get_out_tokens(generated_ouput)
        latencies.append((artificial_in_tokens, out_tokens, latency))

async def single_benchmark(num_requests_per_worker: int, num_workers: int, in_tokens: int = 2048, out_tokens: int = 256) -> None:
    """Runs num_workers's parallel sets of queries with num_requests_per_worker queries per worker.

    Args:
        num_requests_per_worker (int): number of requests per worker
        num_workers (int): number of workers to run in parallel
        out_tokens (int, optional): number of max output tokens. Defaults to 256.
        in_tokens (int, optional): number of input tokens. Defaults to 2048.
    """
    
    tasks = []
    for i in range(num_workers):
        # run worker in background
        # task = asyncio.create_task(worker(i, num_requests_per_worker, in_tokens, out_tokens))
        task = asyncio.create_task(worker(i, num_requests_per_worker, in_tokens, out_tokens))
        tasks.append(task)
        
    # gather results from concurrent workers 
    _ = [await f for f in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]

async def benchmark(parallel_workers: int = 1, in_tokens: int = 2048, out_tokens: int = 256, num_requests_per_worker: int = 5) -> dict:
    """Runs the benchmark with 1, n/2 and n output tokens to enable deriving time to first token (from 1 output token)
    and the time per token by looking at the difference in latency between 64 and 128 output tokens.

    Args:
        parallel_workers (int, optional): number of parallel workers. Defaults to 1.
        in_tokens (int, optional): number of desired input tokens. Defaults to 2048.
        out_tokens (int, optional): number of desired max output tokens. Defaults to 256.
        num_requests_per_worker (int, optional): number of requests per worker. Defaults to 5.

    Returns:
        dict: dictionary with performance metrics for each number of workers.  
    """
    
    # store statistics about the number of input/outpu and the latency for each setup.
    avg_num_input_tokens = [0, 0, 0]
    avg_num_output_tokens = [0, 0, 0]
    median_latency = [0, 0, 0]    
    
    print(f"Parallel queries {parallel_workers}")
    for i, out_tokens in enumerate([1, out_tokens/2, out_tokens]):

        # Clear the latencies array so that we get fresh statistics.
        latencies.clear()
        await single_benchmark(num_requests_per_worker, parallel_workers, in_tokens, out_tokens)

        # Compute the median latency and the mean number of tokens.
        avg_num_input_tokens[i] = statistics.mean([inp for inp, _, _ in latencies])
        avg_num_output_tokens[i] = statistics.mean([outp for _, outp, _ in latencies])
        median_latency[i] = statistics.median([latency for _, _, latency in latencies])

        tokens_per_sec = (avg_num_input_tokens[i]+avg_num_output_tokens[i])*parallel_workers/median_latency[i]
        print(f'Output tokens {avg_num_output_tokens[i]}, median latency (s): {round(median_latency[i], 2)}, tokens per second {round(tokens_per_sec, 1)}')

    # We use difference in the time between out_tokens/2 and out_tokens to generate find the time per output token
    # these are stored in median_latency[1] and median_latency[2] respectively
    # We use the time to generate just 1 token to get the time to first token, this is stored in median_latency[0]
    output_token_time = (median_latency[2] - median_latency[1])*1000/(avg_num_output_tokens[2]-avg_num_output_tokens[1])
    print(f'Time to first token (s): {round(median_latency[0],2)}, Time per output token (ms) {round(output_token_time,2)}')

    output = {
        'parallel_workers': parallel_workers,
        'performance_metrics': {
            'TTFT': round(median_latency[0],2),
            'TPOT': output_token_time,
            'latency': median_latency[2],
            'throughput': (avg_num_input_tokens[2]+avg_num_output_tokens[2])*parallel_workers/median_latency[2]
        }
    }
    return output

if __name__ == '__main__':
    
    config_path = '../config.yaml'

    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    data = []
    
    # Number of input and outut tokens to benchmark
    input_tokens = config['performance_eval']['input_tokens']
    output_tokens = config['performance_eval']['output_tokens']
    # Number of requests per worker(thread), higher gives more accurate results
    num_requests_per_worker = config['performance_eval']['num_requests_per_worker']     
    
    for parallel_workers in [1, 2, 4, 8]:
        benchmark_output = asyncio.run(benchmark(parallel_workers, input_tokens, output_tokens, num_requests_per_worker))
        data.append(benchmark_output)
        # Break if the throughput doesn't increase by more than 10%
        if len(data) > 1 and (data[-1]['performance_metrics']['throughput'] - data[-2]['performance_metrics']['throughput'])/data[-2]['performance_metrics']['throughput'] < 0.1:
            break
    
    print(data)