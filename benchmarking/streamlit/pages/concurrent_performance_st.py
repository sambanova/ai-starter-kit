import os
import sys
sys.path.append('../../')

import time
import streamlit as st
import pandas as pd
import math
import numpy as np
import asyncio
import yaml
from src.performance_evaluation import benchmark

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, "../../"))

CONFIG_PATH = os.path.join(kit_dir,'config.yaml')

with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

def parse_to_df(json_data: dict) -> pd.DataFrame:
    """Parses dictionary to Pandas DataFrame.

    Args:
        json_data (dict): dictionary with metrics for each number of workers.

    Returns:
        pd.DataFrame: Dataframe with metrics for each number of workers.
    """
    
    output_dict = {
        'number_workers': [],
        'ttft': [],
        'tpot': [],
        'latency': [],
        'throughput': []
    }
    for datapoint in json_data:
        output_dict['number_workers'].append(datapoint['parallel_workers'])
        output_dict['ttft'].append(datapoint['performance_metrics']['TTFT'])
        output_dict['tpot'].append(datapoint['performance_metrics']['TPOT'])
        output_dict['latency'].append(datapoint['performance_metrics']['latency'])
        output_dict['throughput'].append(datapoint['performance_metrics']['throughput'])
        
    return pd.DataFrame(output_dict)
        
def get_list_parallel_workers(num_max_workers):
    exp = int(math.log(num_max_workers, 2))
    range = np.logspace(0, exp+1, num=exp+1, base=2, endpoint=False).tolist()
    range = [int(number) for number in range]
    return range

def run_performance_evaluation(input_tokens, output_tokens, num_requests_worker, num_max_workers) -> pd.DataFrame:
    """Runs the performance evaluation process for different number of workers that will run in parallel.
    Each worker will run num_requests_per_worker requests.

    Returns:
        pd.DataFrame: Dataframe with metrics for each number of workers.
    """
    
    data = []
    progress_text = "Starting process. Please wait."
    progress_bar = st.progress(0, text=progress_text)

    # Range of workers to benchmark
    list_workers = get_list_parallel_workers(num_max_workers)
    
    for parallel_workers in list_workers:
        progress = (list_workers.index(parallel_workers)+1)/len(list_workers)
        progress_bar.progress(progress, text=f'Processing benchmark for {parallel_workers} concurrent worker(s).')
        benchmark_output = asyncio.run(benchmark(parallel_workers, input_tokens, output_tokens, num_requests_worker))
        data.append(benchmark_output)
        # Break if the throughput doesn't increase by more than 10%
        if len(data) > 1 and (data[-1]['performance_metrics']['throughput'] - data[-2]['performance_metrics']['throughput'])/data[-2]['performance_metrics']['throughput'] < 0.1:
            break
    progress_bar.empty()
    
    return parse_to_df(data)


st.set_page_config(
    page_title="AI Starter Kit",
    page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
)

st.title(":orange[SambaNova]Performance evaluation")    
st.header("Concurrency")    
st.markdown("This performance evaluation assesses the following LLM's performance metrics using concurrent processes.")
st.markdown('**Latency:** TTFT + (TPOT) * (the number of tokens to be generated)')
st.markdown('**Throughput:** Number of output tokens per second across all concurrency requests')
st.markdown('**Time to first token (TTFT):** This metric is driven by the time required to process the prompt and then generate the first output token.')
st.markdown('**Time per output token (TPOT):** Time to generate an output token for each user that is querying the system.')

with st.sidebar:
    st.title("Evaluation process")
    st.markdown("**Introduce inputs before running the process**")
    
    input_tokens = st.number_input('Number of input tokens', min_value=10, max_value=2000, value=50)
    output_tokens = st.number_input('Number of output tokens', min_value=10, max_value=2000, value=50)
    num_requests_worker = st.number_input('Number of requests per worker', min_value=1, max_value=15, value=5)
    num_max_workers = st.selectbox('Maximum number of concurrent workers', options = [2,4,8,16], index=2)
    
    sidebar_option = st.sidebar.button("Run!")

if sidebar_option:
    with st.spinner("Processing"):
        # Process data
        performance_eval_start = time.time()
        df = run_performance_evaluation(input_tokens, output_tokens, num_requests_worker, num_max_workers)
        performance_eval_end = time.time()
        process_duration = performance_eval_end-performance_eval_start
        print(f'Performance evaluation process took {time.strftime("%H:%M:%S", time.gmtime(process_duration))}')

        st.subheader("Results visualization:")

        # Create two rows for the charts
        # First row
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Latency (s)")
            st.pyplot(df.plot.line(x='number_workers', y='latency').figure)
        with col2:
            st.subheader("Throughput (tokens/s)")
            st.pyplot(df.plot.line(x='number_workers', y='throughput').figure)

        # Second row
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Time To First Token (s)")
            st.pyplot(df.plot.line(x='number_workers', y='ttft').figure)
        with col4:
            st.subheader("Time Per Output Token (ms/token)")
            st.pyplot(df.plot.line(x='number_workers', y='tpot').figure)

        # Display the table
        st.subheader("Summary table:")
        st.write(df.set_index('number_workers'))