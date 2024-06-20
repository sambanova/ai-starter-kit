import sys

sys.path.append("./src")
sys.path.append("./streamlit")

import re
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from st_pages import Page, show_pages

from token_benchmark import run_token_benchmark

from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")


@st.cache_data
def _init():
    load_dotenv("../.env", override=True)


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
    final_df["number_total_tokens"] = valid_df["number_total_tokens"]
    final_df["concurrent_user"] = valid_df["concurrent_user"]

    # server metrics
    final_df["ttft_server_s"] = valid_df["ttft_server_s"]
    final_df["end_to_end_latency_server_s"] = valid_df["end_to_end_latency_server_s"]
    final_df["generation_throughput_server"] = valid_df[
        "request_output_throughput_server_token_per_s"
    ]

    # client metrics
    final_df["ttft_s"] = valid_df["ttft_s"]
    final_df["end_to_end_latency_s"] = valid_df["end_to_end_latency_s"]
    final_df["generation_throughput"] = valid_df[
        "request_output_throughput_token_per_s"
    ]

    return final_df


def _transform_df_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms input dataframe into another with server and client types

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: transformed dataframe with server and client type
    """

    df_server = df[
        [
            "ttft_server_s",
            "number_input_tokens",
            "number_total_tokens",
            "generation_throughput_server",
            "number_output_tokens",
            "end_to_end_latency_server_s",
        ]
    ].copy()
    df_server = df_server.rename(
        columns={
            "ttft_server_s": "ttft",
            "generation_throughput_server": "generation_throughput",
            "end_to_end_latency_server_s": "e2e_latency",
        }
    )
    df_server["type"] = "Server side"

    df_client = df[
        [
            "ttft_s",
            "number_input_tokens",
            "number_total_tokens",
            "generation_throughput",
            "number_output_tokens",
            "end_to_end_latency_s",
        ]
    ].copy()
    df_client = df_client.rename(
        columns={"ttft_s": "ttft", "end_to_end_latency_s": "e2e_latency"}
    )
    df_client["type"] = "Client side"

    df_ttft_throughput_latency = pd.concat([df_server, df_client], ignore_index=True)

    return df_ttft_throughput_latency


def _run_performance_evaluation() -> pd.DataFrame:
    """Runs the performance evaluation process for different number of workers that will run in parallel.
    Each worker will run num_requests_per_worker requests.

    Returns:
        pd.DataFrame: Dataframe with metrics for each number of workers.
    """

    results_path = "./data/results/llmperf"
    num_concurrent_workers = st.session_state.number_concurrent_workers

    # Call benchmarking process. Static param values are intentional and still WIP.
    mode = "stream"
    llm_api = "sambastudio"

    run_token_benchmark(
        model=st.session_state.llm,
        mean_input_tokens=st.session_state.input_tokens,
        mean_output_tokens=st.session_state.output_tokens,
        timeout_s=st.session_state.timeout,
        max_num_completed_requests=st.session_state.number_requests,
        num_concurrent_workers=num_concurrent_workers,
        additional_sampling_params="{}",
        results_dir=results_path,
        user_metadata="",
        llm_api=llm_api,
        mode=mode,
    )

    # read generated json and output formatted results
    df = pd.DataFrame()
    model = re.sub("\/|\.", "-", st.session_state.llm)
    df_user = pd.read_json(
        f"{results_path}/{model}_{st.session_state.input_tokens}_{st.session_state.output_tokens}_{num_concurrent_workers}_{mode}_individual_responses.json"
    )
    df_user["concurrent_user"] = num_concurrent_workers
    df = pd.concat([df, df_user])
    valid_df = df[(df["error_code"] != "")]
    renamed_df = _rename_metrics_df(valid_df)
    df_ttft_throughput_latency = _transform_df_for_plotting(renamed_df)

    return df_ttft_throughput_latency


def _initialize_sesion_variables():

    if "llm" not in st.session_state:
        st.session_state.llm = None

    # Initialize llm params
    if "input_tokens" not in st.session_state:
        st.session_state.input_tokens = None
    if "output_tokens" not in st.session_state:
        st.session_state.output_tokens = None
    if "number_requests" not in st.session_state:
        st.session_state.number_requests = None
    if "number_concurrent_workers" not in st.session_state:
        st.session_state.number_concurrent_workers = None
    if "timeout" not in st.session_state:
        st.session_state.timeout = None


def main():

    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )

    show_pages(
        [
            Page("streamlit/app.py", "Performance evaluation"),
            Page("streamlit/pages/chat_performance_st.py", "Performance on chat"),
        ]
    )

    _init()
    _initialize_sesion_variables()

    st.title(":orange[SambaNova]Performance evaluation")
    st.markdown(
        "This performance evaluation assesses the following LLM's performance metrics using concurrent processes. _client represent the metrics computed from the client-side and _server represents the metrics computed from the server-side."
    )
    st.markdown(
        "**Time to first token (TTFT):** This metric is driven by the time required to process the prompt and then generate the first output token. Client metric is calculated using another request with output tokens = 1."
    )
    st.markdown(
        "**E2E Latency:** TTFT + (Time per Output Token) * (the number of tokens to be generated)"
    )
    st.markdown(
        "**Throughput:** Number of output tokens per second across all concurrency requests. Client metric is calculated as *Number of Output Tokens / (E2E Latency - TTFT)*"
    )

    with st.sidebar:
        st.title("Configuration")
        st.markdown("**Modify the following parameters before running the process**")

        llm_model = st.text_input(
            "Introduce a valid LLM model name",
            value="COE/Meta-Llama-3-8B-Instruct",
            help="Look at your model card in SambaStudio and introduce the same name of the model/expert here.",
        )
        st.session_state.llm = f"{llm_model}"

        st.session_state.input_tokens = st.slider(
            "Number of input tokens", min_value=50, max_value=2048, value=1000
        )

        st.session_state.output_tokens = st.slider(
            "Number of output tokens", min_value=50, max_value=2048, value=1000
        )

        st.session_state.number_requests = st.slider(
            "Number of total requests", min_value=10, max_value=100, value=32
        )

        st.session_state.number_concurrent_workers = st.slider(
            "Number of concurrent workers", min_value=1, max_value=10, value=1
        )

        st.session_state.timeout = st.slider(
            "Timeout", min_value=60, max_value=1800, value=600
        )

        sidebar_option = st.sidebar.button("Run!")

    if sidebar_option:

        st.toast("Performance evaluation processing now. It should take few minutes.")
        with st.spinner("Processing"):

            performance_eval_start = time.time()

            try:

                df = _run_performance_evaluation()

                performance_eval_end = time.time()
                process_duration = performance_eval_end - performance_eval_start
                print(
                    f'Performance evaluation process took {time.strftime("%H:%M:%S", time.gmtime(process_duration))}'
                )

                st.subheader("TTFT, Throughput and E2E Latency box plots")

                fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 20))
                sns.boxplot(data=df, x="ttft", y="type", ax=ax[0]).set(
                    xlabel="Time to First Token (secs)",
                    ylabel="Type",
                    title="Time to First Token Distribution",
                )
                sns.boxplot(data=df, x="e2e_latency", y="type", ax=ax[1]).set(
                    xlabel="E2E Latency (secs)",
                    ylabel="Type",
                    title="End-to-end Latency Distribution",
                )
                sns.boxplot(data=df, x="generation_throughput", y="type", ax=ax[2]).set(
                    xlabel="Throughput (tokens/sec)",
                    ylabel="Type",
                    title="Throughput Distribution",
                )
                st.pyplot(fig)

            except Exception as e:
                st.error(
                    f"Error: {e} For more error details, please look at the terminal."
                )


if __name__ == "__main__":
    main()
