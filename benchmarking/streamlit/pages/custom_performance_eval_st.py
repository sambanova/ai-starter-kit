import pandas as pd
import streamlit as st
from streamlit_utils import plot_client_vs_server_barplots, plot_dataframe_summary
import matplotlib.pyplot as plt

from benchmarking.src.performance_evaluation import CustomPerformanceEvaluator

import warnings

warnings.filterwarnings("ignore")


def _initialize_sesion_variables():
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "perf_metrics_history" not in st.session_state:
        st.session_state.perf_metrics_history = []
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "chat_disabled" not in st.session_state:
        st.session_state.chat_disabled = True

    # Initialize llm params
    if "do_sample" not in st.session_state:
        st.session_state.do_sample = None
    if "max_tokens_to_generate" not in st.session_state:
        st.session_state.max_tokens_to_generate = None
    if "repetition_penalty" not in st.session_state:
        st.session_state.repetition_penalty = None
    if "temperature" not in st.session_state:
        st.session_state.temperature = None
    if "top_k" not in st.session_state:
        st.session_state.top_k = None
    if "top_p" not in st.session_state:
        st.session_state.top_p = None


def _run_custom_performance_evaluation():

    results_path = "./data/results/llmperf"

    custom_performance_evaluator = CustomPerformanceEvaluator(
        model_name=st.session_state.llm,
        results_dir=results_path,
        num_workers=st.session_state.number_concurrent_workers,
        timeout=st.session_state.timeout,
        input_file_path=st.session_state.file_path,
    )

    custom_performance_evaluator.run_benchmark(
        sampling_params={},
    )

    df_user = pd.read_json(custom_performance_evaluator.individual_responses_file_path)
    df_user["concurrent_user"] = custom_performance_evaluator.num_workers
    valid_df = df_user[(df_user["error_code"] != "")]

    if valid_df["batch_size_used"].isnull().all():
        valid_df["batch_size_used"] = 1

    return valid_df


def main():

    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )

    _initialize_sesion_variables()

    st.title(":orange[SambaNova] Custom Performance Evaluation")
    st.markdown(
        "Here you can select a custom dataset that you want to benchmark performance with. Note that with models that support dynamic \
            batching, you are limited to the number of cpus available on your machine to send concurrent requests."
    )

    with st.sidebar:

        ##################
        # File Selection #
        ##################
        st.title("File Selection")
        st.text_input("Full File Path", help="", key="file_path")  # TODO: Fill in help

        #########################
        # Runtime Configuration #
        #########################
        st.title("Configuration")

        st.text_input(
            "Model Name",
            value="COE/Meta-Llama-3-8B-Instruct",
            key="llm",
            help="Look at your model card in SambaStudio and introduce the same name of the model/expert here.",
        )

        st.slider(
            "Num Concurrent Workers",
            min_value=1,
            max_value=100,
            value=1,
            key="number_concurrent_workers",
        )

        st.slider("Timeout", min_value=60, max_value=1800, value=600, key="timeout")

        #####################
        # Tuning Parameters #
        #####################
        st.title("Tuning Parameters")

        st.slider(
            "Max Output Tokens",
            min_value=1,
            max_value=2048,
            value=256,
            key="max_tokens",
        )

        # TODO: Add more tuning params below (temperature, top_k, etc.)

        job_submitted = st.sidebar.button("Run!")

    if job_submitted:

        st.toast(
            "Performance evaluation in progress. This could take a while depending on the dataset size and max tokens setting."
        )
        with st.spinner("Processing"):

            try:

                results_df = _run_custom_performance_evaluation()

                st.subheader("Performance metrics plots")
                fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 24))
                plot_client_vs_server_barplots(
                    results_df,
                    "batch_size_used",
                    ["server_ttft_s", "client_ttft_s"],
                    "Boxplots for Server token/s and Client token/s per request",
                    "seconds",
                    ax[0],
                )
                plot_client_vs_server_barplots(
                    results_df,
                    "batch_size_used",
                    ["server_end_to_end_latency_s", "client_end_to_end_latency_s"],
                    "Boxplots for Server latency and Client latency",
                    "seconds",
                    ax[1],
                )
                plot_client_vs_server_barplots(
                    results_df,
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
                plot_dataframe_summary(results_df, ax[3])
                st.pyplot(fig)

            except Exception as e:
                st.error(
                    f"Error: {e}. For more error details, please look at the terminal."
                )


if __name__ == "__main__":
    main()
