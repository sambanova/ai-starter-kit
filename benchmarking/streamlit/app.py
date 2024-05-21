import os
import sys
sys.path.append('../')
sys.path.append('../src')

import re
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from st_pages import Page, show_pages
from dotenv import load_dotenv
from token_benchmark_ray import run_token_benchmark

# paths added to PYTHONPATH for Ray
streamlit_dir = os.path.dirname(os.path.abspath(__file__))
benchmarking_dir = os.path.dirname(os.path.abspath(streamlit_dir))
src_dir = os.path.abspath(f'{benchmarking_dir}/src')
llmperf_dir = os.path.abspath(f'{src_dir}/llmperf')
os.environ["PYTHONPATH"] = streamlit_dir + ":" + src_dir + ":" + llmperf_dir + ":" + benchmarking_dir + ":" + os.environ.get("PYTHONPATH", "")   
    
load_dotenv('../../.env', override=True)

def _rename_metrics_df(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Rename metric names from input dataframe.

    Args:
        valid_df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: dataframe with renamed fields
    """
    
    final_df = pd.DataFrame()
    final_df["number_input_tokens"] = valid_df["number_input_tokens"]
    final_df["number_output_tokens"] = valid_df["number_output_tokens"]
    final_df["ttft_s"] = valid_df["ttft_s"]
    final_df["end_to_end_latency_s"] = valid_df["end_to_end_latency_s"]
    final_df["generation_throughput"] = valid_df["request_output_throughput_token_per_s"]
    final_df["concurrent_user"] = valid_df["concurrent_user"]
    return final_df

def _run_performance_evaluation(model: str, 
                               input_tokens: int, 
                               input_tokens_std: int, 
                               output_tokens: int, 
                               output_tokens_std: int, 
                               number_requests: int, 
                               number_concurrent_requests: int, 
                               timeout: int
    ) -> pd.DataFrame:
    """Runs the performance evaluation process for different number of workers that will run in parallel.
    Each worker will run num_requests_per_worker requests.

    Args:
        model (str): LLM model selected
        input_tokens (int): mean number of input tokens
        input_tokens_std (int): standard deviation number for input tokens
        output_tokens (int): mean number of output tokens
        output_tokens_std (int): standard deviation number for output tokens
        number_requests (int): number of total requests
        number_concurrent_requests (int): number of concurrent requests
        timeout (int): maximum time out of the benchmark process 

    Returns:
        pd.DataFrame: Dataframe with metrics for each number of workers.
    """
    
    results_path = "./../data/results/llmperf"
    mode = 'stream' # static for now
    
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
        additional_sampling_params='{}',
        results_dir=results_path,
        user_metadata="",
        mode=mode
    )
    
    # read generated json and output formatted results
    df = pd.DataFrame()
    model = re.sub('\/|\.','-',model)
    df_user = pd.read_json(f"{results_path}/{model}_{input_tokens}_{output_tokens}_{number_concurrent_requests}_{mode}_individual_responses.json")
    df_user['concurrent_user'] = number_concurrent_requests
    df = pd.concat([df,df_user])
    valid_df = df[(df["error_code"] != "")]
    final_df = _rename_metrics_df(valid_df)
    
    return final_df

def _get_model_options() -> list:
    """Gets a list of COE LLM model names

    Returns:
        list: list with COE LLM model names
    """
    llm_options = [
        'COE/Mistral-7B-Instruct-v0.2',
        'COE/zephyr-7b-beta',
        'COE/Mistral-T5-7B-v1',
        'COE/v1olet_merged_dpo_7B',
        'COE/Lil-c3po',
        'COE/DonutLM-v1',
        'COE/Rabbit-7B-DPO-Chat',
        'COE/Snorkel-Mistral-PairRM-DPO',
        'COE/llama-2-7b-chat-hf',
        'COE/LlamaGuard-7b',
    ]
    llm_options.sort(key=lambda x: x.split('/')[-1].upper())
    return llm_options

def main():
    
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )

    show_pages(
        [
            Page("app.py", "Performance evaluation"),
            Page("pages/chat_performance_st.py", "Performance on chat")
        ]
    )

    st.title(":orange[SambaNova]Performance evaluation")    
    st.markdown("This performance evaluation assesses the following LLM's performance metrics using concurrent processes.")
    st.markdown('**Latency:** TTFT + (TPOT) * (the number of tokens to be generated)')
    st.markdown('**Throughput:** Number of output tokens per second across all concurrency requests')
    st.markdown('**Time to first token (TTFT):** This metric is driven by the time required to process the prompt and then generate the first output token.')
    st.markdown('**Time per output token (TPOT):** Time to generate an output token for each user that is querying the system.')

    with st.sidebar:
        st.title("Configuration")
        st.markdown("**Modify the following parameters before running the process**")
        
        llm_options = _get_model_options()
        model_selected = st.selectbox('Choose a LLM model', llm_options, index=0, format_func=lambda x: x.split('/')[-1])
        
        input_tokens = st.number_input('Number of input tokens', min_value=50, max_value=1000, value=150)
        input_tokens_std = st.number_input('Input tokens standard deviation', min_value=10, max_value=500, value=50)
        
        output_tokens = st.number_input('Number of output tokens', min_value=50, max_value=1000, value=150)
        output_tokens_std = st.number_input('Output tokens standard deviation', min_value=10, max_value=500, value=50)
        
        number_requests = st.number_input('Number of total requests', min_value=10, max_value=100, value=32)
        number_concurrent_requests = st.number_input('Number of concurrent requests', min_value=1, max_value=100, value=8)
        
        timeout = st.number_input('Timeout', min_value=60, max_value=1800, value=600)
        
        sidebar_option = st.sidebar.button("Run!")

    if sidebar_option:
        
        st.toast('Performance evaluation processing now. It should take few minutes.')
        with st.spinner("Processing"):
            
            performance_eval_start = time.time()
            df = _run_performance_evaluation(model_selected, input_tokens, input_tokens_std, output_tokens, output_tokens_std, number_requests, number_concurrent_requests, timeout)
            performance_eval_end = time.time()
            process_duration = performance_eval_end-performance_eval_start
            print(f'Performance evaluation process took {time.strftime("%H:%M:%S", time.gmtime(process_duration))}')

            st.subheader("Input tokens and Output tokens")
            
            fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8,12))
            # Number of Input Tokens vs. TTFT
            sns.scatterplot(data=df, x="number_input_tokens", y="ttft_s", hue="concurrent_user", ax=ax[0]).set_title("Number of Input Tokens vs. TTFT")
            ax[0].set(xlabel='Number of Input Tokens', ylabel='Time to First Token')
            ax[0].legend(title='Concurrent Users')
            # Number of output Tokens vs. Throughput
            sns.scatterplot(data=df, x="number_output_tokens", y="generation_throughput", hue="concurrent_user", ax=ax[1]).set_title("Number of output Tokens vs. Throughput")
            ax[1].set(xlabel='Number of Output Tokens', ylabel='Generation Throughput')
            ax[1].legend(title='Concurrent Users')
            st.pyplot(fig)
            
            st.subheader("TTFT and Throughput")
            
            fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8,12))
            # Time to First Token boxplot
            sns.boxplot(data=df, x="ttft_s", hue="concurrent_user", ax=ax[0])
            ax[0].set(xlabel='Time to First Token')
            # Generation Throughput boxplot
            sns.boxplot(data=df, x="generation_throughput", hue="concurrent_user", ax=ax[1])
            ax[1].set(xlabel='Generation Throughput')
            st.pyplot(fig)
            
if __name__ == '__main__':
    main()