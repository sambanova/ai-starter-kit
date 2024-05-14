import os
import sys
sys.path.append('../')

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import yaml
from src.performance_evaluation import benchmark

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))

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
        

def run_performance_evaluation() -> pd.DataFrame:
    """Runs the performance evaluation process for different number of workers that will run in parallel.
    Each worker will run num_requests_per_worker requests.

    Returns:
        pd.DataFrame: Dataframe with metrics for each number of workers.
    """
    
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
    
    return parse_to_df(data)


def main():
    
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )
    
    # Set page title
    st.title(":orange[SambaNova]Performance evaluation")

    with st.sidebar:
        st.title("Evaluation process")
        st.markdown("**Click the button bellow to run the process**")
        
        sidebar_option = st.sidebar.button("Run!")

    if sidebar_option:
        with st.spinner("Processing"):
            # Process data
            df = run_performance_evaluation()

            st.subheader("Results visualization:")

            # Create two rows for the charts
            # First row
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Latency")
                st.pyplot(df.plot.line(x='number_workers', y='latency').figure)
            with col2:
                st.subheader("Throughput")
                st.pyplot(df.plot.line(x='number_workers', y='throughput').figure)

            # Second row
            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Time To First Token")
                st.pyplot(df.plot.line(x='number_workers', y='ttft').figure)
            with col4:
                st.subheader("Time Per Output Token")
                st.pyplot(df.plot.line(x='number_workers', y='tpot').figure)

            # Display the table
            st.subheader("Summary table:")
            st.write(df, hide_index=True)

if __name__ == "__main__":
    main()
