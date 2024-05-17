import os
import re
import ray
import time
import streamlit as st
import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import yaml
# from benchmarking.src.concurrent_performance_evaluation import benchmark
from token_benchmark_ray import run_token_benchmark
# from benchmarking.src.concurrent_performance_evaluation import get_snovastudio_llm

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

def run_performance_evaluation(model, input_tokens, input_tokens_std, output_tokens, output_tokens_std, number_requests, number_concurrent_requests, timeout) -> pd.DataFrame:
    """Runs the performance evaluation process for different number of workers that will run in parallel.
    Each worker will run num_requests_per_worker requests.

    Returns:
        pd.DataFrame: Dataframe with metrics for each number of workers.
    """
    
    run_token_benchmark(
        llm_api='sambanova',
        model=model,
        test_timeout_s=timeout,
        max_num_completed_requests=number_requests,
        mean_input_tokens=input_tokens,
        stddev_input_tokens=input_tokens_std,
        mean_output_tokens=output_tokens,
        stddev_output_tokens=output_tokens_std,
        num_concurrent_requests=number_concurrent_requests,
        additional_sampling_params='{"process_prompt":"False"}',
        results_dir="./../data/results/llmperf",
        user_metadata="",
    )
    
    # path to the individual responses json file
    df = pd.DataFrame()
    model = re.sub('\/|\.','-',model)
    df_user = pd.read_json(f"./../data/results/llmperf/{model}_{input_tokens}_{output_tokens}_{number_concurrent_requests}_individual_responses.json")
    df_user['concurrent_user'] = number_concurrent_requests
    df = pd.concat([df,df_user])
    
    valid_df = df[(df["error_code"] != "")]
    final_df = rename_metrics_df(valid_df)
    
    return final_df

def rename_metrics_df(valid_df):
    final_df = pd.DataFrame()
    final_df["number_input_tokens"] = valid_df["number_input_tokens"]
    final_df["number_output_tokens"] = valid_df["number_output_tokens"]
    final_df["ttft_s"] = valid_df["ttft_s"]
    final_df["end_to_end_latency_s"] = valid_df["end_to_end_latency_s"]
    final_df["generation_throughput"] = valid_df["request_output_throughput_token_per_s"]
    final_df["concurrent_user"] = valid_df["concurrent_user"]
    return final_df

def get_model_options():
    llm_options = [
        'COE/Mistral-7B-Instruct-v0.2',
        'COE/zephyr-7b-beta',
        'COE/Mistral-T5-7B-v1',
        'COE/v1olet_merged_dpo_7B',
        'COE/Lil-c3po',
        'COE/DonutLM-v1',
        'COE/Rabbit-7B-DPO-Chat',
        'COE/Snorkel-Mistral-PairRM-DPO',
        'COE/Llama-2-7b-chat-hf',
        'COE/LlamaGuard-7b',
        'COE/base_llama'
    ]
    return llm_options

st.set_page_config(
    page_title="AI Starter Kit",
    page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
)

st.title(":orange[SambaNova]Performance evaluation")    
st.markdown("This performance evaluation assesses the following LLM's performance metrics using concurrent processes.")
st.markdown('**Latency:** TTFT + (TPOT) * (the number of tokens to be generated)')
st.markdown('**Throughput:** Number of output tokens per second across all concurrency requests')
st.markdown('**Time to first token (TTFT):** This metric is driven by the time required to process the prompt and then generate the first output token.')
st.markdown('**Time per output token (TPOT):** Time to generate an output token for each user that is querying the system.')

with st.sidebar:
    st.title("Evaluation process")
    st.markdown("**Introduce inputs before running the process**")
    
    llm_options = get_model_options()
    model_selected = st.selectbox('Choose a LLM model', llm_options, index=0, format_func=lambda x: x.split('/')[-1])
    
    input_tokens = st.number_input('Number of input tokens', min_value=50, max_value=1000, value=150)
    input_tokens_std = st.number_input('Input tokens standard deviation', min_value=10, max_value=500, value=50)
    
    output_tokens = st.number_input('Number of output tokens', min_value=50, max_value=1000, value=150)
    output_tokens_std = st.number_input('Output tokens standard deviation', min_value=10, max_value=500, value=50)
    
    # number_requests = st.number_input('Number of total requests', min_value=10, max_value=100, value=32)
    number_requests = st.number_input('Number of total requests', min_value=1, max_value=100, value=4)
    # number_concurrent_requests = st.number_input('Number of concurrent requests', min_value=1, max_value=100, value=8)
    number_concurrent_requests = st.number_input('Number of concurrent requests', min_value=1, max_value=100, value=2)
    
    timeout = st.number_input('Timeout', min_value=60, max_value=1800, value=600)
    
    sidebar_option = st.sidebar.button("Run!")

if sidebar_option:
    with st.spinner("Processing"):
        # Process data
        performance_eval_start = time.time()
        df = run_performance_evaluation(model_selected, input_tokens, input_tokens_std, output_tokens, output_tokens_std, number_requests, number_concurrent_requests, timeout)
        performance_eval_end = time.time()
        process_duration = performance_eval_end-performance_eval_start
        print(f'Performance evaluation process took {time.strftime("%H:%M:%S", time.gmtime(process_duration))}')

        st.subheader("Input tokens and Output tokens")
        
        fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8,12))
        sns.scatterplot(data=df, x="number_input_tokens", y="ttft_s", hue="concurrent_user", ax=ax[0]).set_title("Number of Input Tokens vs. TTFT")
        sns.scatterplot(data=df, x="number_output_tokens", y="generation_throughput", hue="concurrent_user", ax=ax[1]).set_title("Number of output Tokens vs. Throughput")
        st.pyplot(fig)
        
        st.subheader("TTFT and Throughput")
        
        fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8,12))
        sns.boxplot(data=df, x="ttft_s", hue="concurrent_user", ax=ax[0])
        sns.boxplot(data=df, x="generation_throughput", hue="concurrent_user", ax=ax[1])
        st.pyplot(fig)

        # Display the table
        # st.subheader("Summary table:")
        # st.write(df)