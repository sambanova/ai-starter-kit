import sys

sys.path.append("../")
sys.path.append("./src")
sys.path.append("./streamlit")

import re
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from st_pages import Page, show_pages
from typing import List
from matplotlib.axes._axes import Axes


from benchmarking.src.token_benchmark import run_token_benchmark

from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")


@st.cache_data
def _init():
    load_dotenv("../.env", override=True)


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
        num_input_tokens=st.session_state.input_tokens,
        num_output_tokens=st.session_state.output_tokens,
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
    model = re.sub("\/|\.", "-", st.session_state.llm)
    df_user = pd.read_json(
        f"{results_path}/{model}_{st.session_state.input_tokens}_{st.session_state.output_tokens}_{num_concurrent_workers}_{mode}_individual_responses.json"
    )
    df_user["concurrent_user"] = num_concurrent_workers
    df_user = df_user[(df_user["error_code"] != "")]
    # renamed_df = _rename_metrics_df(valid_df)
    # df_ttft_throughput_latency = _transform_df_for_plotting(valid_df)

    return df_user


def plot_client_vs_server_barplots(
    df_user: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str,
    ylabel: str,
    ax: Axes,
) -> None:
    """
    Plots bar plots for client vs server metrics from a DataFrame.

    Args:
        df_user (pd.DataFrame): The DataFrame containing the data to plot.
        x_col (str): The column name to be used as the x-axis.
        y_cols (List[str]): A list of column names to be used as the y-axis.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.

    Returns:
        None
    """
    # Melt the DataFrame to have a long-form DataFrame suitable for Seaborn
    df_melted = df_user.melt(
        id_vars=[x_col], value_vars=y_cols, var_name="Metric", value_name="Value"
    )

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melted, x=x_col, y="Value", hue="Metric", ax=ax)

    # Customize the plot
    plt.title(title)
    plt.xlabel("Batch Size Used")
    plt.ylabel(ylabel)

    # Show the plot
    plt.legend(title="Metric")
    plt.show()


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
        "**Time to first token (TTFT):** This metric is driven by the time required to process the prompt and then generate the first output token."
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
            "Number of concurrent workers", min_value=1, max_value=100, value=1
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

                df_req_info = _run_performance_evaluation()
                performance_eval_end = time.time()
                process_duration = performance_eval_end - performance_eval_start
                print(
                    f'Performance evaluation process took {time.strftime("%H:%M:%S", time.gmtime(process_duration))}'
                )

                st.subheader("Performance menterics plots")
                fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 24))
                plot_client_vs_server_barplots(
                    df_req_info,
                    "batch_size_used",
                    ["server_ttft_s", "client_ttft_s"],
                    "Boxplots for Server token/s and Client token/s per request",
                    "seconds",
                    ax[0],
                )
                plot_client_vs_server_barplots(
                    df_req_info,
                    "batch_size_used",
                    ["server_end_to_end_latency_s", "client_end_to_end_latency_s"],
                    "Boxplots for Server latency and Client latency",
                    "seconds",
                    ax[1],
                )
                plot_client_vs_server_barplots(
                    df_req_info,
                    "batch_size_used",
                    [
                        "server_output_token_per_s_per_request",
                        "client_output_token_per_s_per_request",
                    ],
                    "Boxplots for Server token/s and Client token/s per request",
                    "tokens/s",
                    ax[2],
                )
                # Compute total throughput per batch
                plot_dataframe_summary(df_req_info, ax[3])
                st.pyplot(fig)

            except Exception as e:
                st.error(
                    f"Error: {e}. For more error details, please look at the terminal."
                )


def plot_dataframe_summary(df_req_info, ax):
    df_req_summary = (
        df_req_info.groupby("batch_size_used")[
            [
                "server_output_token_per_s_per_request",
                "client_output_token_per_s_per_request",
            ]
        ]
        .mean()
        .reset_index()
    )
    df_req_summary["server_throughput_token_per_s"] = (
        df_req_summary["server_output_token_per_s_per_request"]
        * df_req_summary["batch_size_used"]
    )
    df_req_summary["client_throughput_token_per_s"] = (
        df_req_summary["client_output_token_per_s_per_request"]
        * df_req_summary["batch_size_used"]
    )
    df_melted = pd.melt(
        df_req_summary,
        id_vars="batch_size_used",
        value_vars=[
            "server_throughput_token_per_s",
            "client_throughput_token_per_s",
        ],
        var_name="Value Type",
        value_name="Value",
    )
    sns.barplot(x="batch_size_used", y="Value", hue="Value Type", data=df_melted, ax=ax)


if __name__ == "__main__":
    main()
