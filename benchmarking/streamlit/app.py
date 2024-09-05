import sys
import os
import streamlit as st
from st_pages import Page, show_pages
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import warnings
import yaml

sys.path.append("../")
sys.path.append("./src")
sys.path.append("./streamlit")


from benchmarking.src.performance_evaluation import SyntheticPerformanceEvaluator
from streamlit_utils import plot_client_vs_server_barplots, plot_dataframe_summary

import os
import streamlit as st
from typing import Tuple, Optional, List
import netrc
from streamlit.runtime.scriptrunner import add_script_run_ctx


DEFAULT_FASTAPI_URL = "https://fast-api.snova.ai/v1/chat/completions"

def initialize_env_variables(prod_mode: bool = False, additional_env_vars: Optional[List[str]] = None) -> None:
    if additional_env_vars is None:
        additional_env_vars = []

    if not prod_mode:
        # In non-prod mode, prioritize environment variables
        st.session_state.FASTAPI_URL = os.environ.get("FASTAPI_URL", st.session_state.get("FASTAPI_URL", DEFAULT_FASTAPI_URL))
        st.session_state.FASTAPI_API_KEY = os.environ.get("FASTAPI_API_KEY", st.session_state.get("FASTAPI_API_KEY", ""))
        for var in additional_env_vars:
            st.session_state[var] = os.environ.get(var, st.session_state.get(var, ""))
    else:
        # In prod mode, only use session state
        if 'FASTAPI_URL' not in st.session_state:
            st.session_state.FASTAPI_URL = DEFAULT_FASTAPI_URL
        if 'FASTAPI_API_KEY' not in st.session_state:
            st.session_state.FASTAPI_API_KEY = ""
        for var in additional_env_vars:
            if var not in st.session_state:
                st.session_state[var] = ""

def set_env_variables(api_key, additional_vars=None, prod_mode=False):
    st.session_state.FASTAPI_API_KEY = api_key
    if additional_vars:
        for key, value in additional_vars.items():
            st.session_state[key] = value
    if not prod_mode:
        # In non-prod mode, also set environment variables
        os.environ["FASTAPI_API_KEY"] = api_key
        if additional_vars:
            for key, value in additional_vars.items():
                os.environ[key] = value

def env_input_fields(mode, additional_env_vars=None):
    if additional_env_vars is None:
        additional_env_vars = []

    additional_vars = {}
    
    if mode == "SambaNova Cloud":
        api_key = st.text_input("SambaNova API Key", value=st.session_state.get("FASTAPI_API_KEY", ""), type="password")
    else:  # SambaStudio
        api_key = st.text_input("SambaStudio API Key", value=st.session_state.get("SAMBASTUDIO_API_KEY", ""), type="password")
        for var in additional_env_vars:
            if var != "SAMBASTUDIO_API_KEY":
                additional_vars[var] = st.text_input(f"{var}", value=st.session_state.get(var, ""), type="password")

    return api_key, additional_vars

def are_credentials_set(additional_env_vars=None) -> bool:
    if additional_env_vars is None:
        additional_env_vars = []

    base_creds_set = bool(st.session_state.FASTAPI_API_KEY)
    additional_creds_set = all(bool(st.session_state.get(var, "")) for var in additional_env_vars)
    
    return base_creds_set and additional_creds_set

def save_credentials(api_key, additional_vars=None, prod_mode=False):
    if api_key is not None:
        st.session_state.FASTAPI_API_KEY = api_key
        if not prod_mode:
            os.environ["FASTAPI_API_KEY"] = api_key

    if additional_vars:
        for key, value in additional_vars.items():
            st.session_state[key] = value
            if not prod_mode:
                os.environ[key] = value

    return "Credentials saved successfully!"


def get_wandb_key():
    # Check for WANDB_API_KEY in environment variables
    env_wandb_api_key = os.getenv('WANDB_API_KEY')

    # Check for WANDB_API_KEY in ~/.netrc
    try:
        netrc_path = os.path.expanduser('~/.netrc')
        netrc_data = netrc.netrc(netrc_path)
        netrc_wandb_api_key = netrc_data.authenticators('api.wandb.ai')
    except (FileNotFoundError, netrc.NetrcParseError):
        netrc_wandb_api_key = None

    # If both are set, handle the conflict
    if env_wandb_api_key and netrc_wandb_api_key:
        print("WANDB_API_KEY is set in both the environment and ~/.netrc. Prioritizing environment variable.")
        # Optionally, you can choose to remove one of them, here we remove the env variable
        del os.environ['WANDB_API_KEY']  # Remove from environment to prioritize ~/.netrc
        return netrc_wandb_api_key[2] if netrc_wandb_api_key else None  # Return the key from .netrc
    
    # Return the key from environment if available, otherwise from .netrc
    if env_wandb_api_key:
        return env_wandb_api_key
    elif netrc_wandb_api_key:
        return netrc_wandb_api_key[2] if netrc_wandb_api_key else None
    
    # If neither is set, return None
    return None


warnings.filterwarnings("ignore")

LLM_API_OPTIONS = ["fastapi", "sambastudio"]

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

    # Call benchmarking process
    performance_evaluator = SyntheticPerformanceEvaluator(
        model_name=st.session_state.llm,
        results_dir=results_path,
        num_workers=st.session_state.number_concurrent_workers,
        timeout=st.session_state.timeout,
        llm_api=st.session_state.llm_api,
    )

    performance_evaluator.run_benchmark(
        num_input_tokens=st.session_state.input_tokens,
        num_output_tokens=st.session_state.output_tokens,
        num_requests=st.session_state.number_requests,
        sampling_params={},
    )

    # Read generated json and output formatted results
    df_user = pd.read_json(performance_evaluator.individual_responses_file_path)
    df_user["concurrent_user"] = st.session_state.number_concurrent_workers
    valid_df = df_user[(df_user["error_code"] != "")]

    # For non-batching endpoints, batch_size_used will be 1
    if valid_df["batch_size_used"].isnull().all():
        valid_df["batch_size_used"] = 1

    return valid_df

def _initialize_session_variables():
    if "llm" not in st.session_state:
        st.session_state.llm = None
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
    if "llm_api" not in st.session_state:
        st.session_state.llm_api = None
    if "mode" not in st.session_state:
        st.session_state.mode = "SambaNova Cloud"
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False

def main():
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )

    show_pages(
        [
            Page("streamlit/app.py", "Synthetic Performance Evaluation"),
            Page("streamlit/pages/custom_performance_eval_st.py", "Custom Performance Evaluation"),
            Page("streamlit/pages/chat_performance_st.py", "Performance on Chat"),
        ]
    )

    _init()
    _initialize_session_variables()

    prod_mode = False

    if not st.session_state.setup_complete:
        st.title("Setup")
        
        # Mode selection
        st.session_state.mode = st.radio("Select Mode", ["SambaNova Cloud", "SambaStudio"])

        if st.session_state.mode == "SambaNova Cloud":
            additional_env_vars = []
            st.session_state.llm_api = "fastapi"
        else:  # SambaStudio
            additional_env_vars = [
                "SAMBASTUDIO_BASE_URL",
                "SAMBASTUDIO_PROJECT_ID",
                "SAMBASTUDIO_ENDPOINT_ID",
                "SAMBASTUDIO_API_KEY"
            ]
            st.session_state.llm_api = "sambastudio"

        initialize_env_variables(prod_mode, additional_env_vars)

        if not are_credentials_set(additional_env_vars):
            api_key, additional_vars = env_input_fields(st.session_state.mode, additional_env_vars)
            if st.button("Save Credentials"):
                if st.session_state.mode == "SambaNova Cloud":
                    message = save_credentials(api_key, None, prod_mode)
                else:  # SambaStudio
                    additional_vars["SAMBASTUDIO_API_KEY"] = api_key
                    message = save_credentials(api_key, additional_vars, prod_mode)
                st.success(message)
                st.session_state.setup_complete = True
                st.rerun()
        else:
            st.success("Credentials are set")
            if st.button("Clear Credentials"):
                if st.session_state.mode == "SambaNova Cloud":
                    save_credentials("", None, prod_mode)
                else:
                    save_credentials("", {var: "" for var in additional_env_vars}, prod_mode)
                st.session_state.setup_complete = False
                st.rerun()
            if st.button("Continue to App"):
                st.session_state.setup_complete = True
                st.rerun()

    else:
        st.title(":orange[SambaNova] Synthetic Performance Evaluation")
        
        # Sidebar
        with st.sidebar:
            st.title("Configuration")
            
            # Display the current mode
            st.info(f"Current Mode: {st.session_state.mode}")
            
            st.markdown("**Modify the following parameters before running the process**")

            llm_model = st.text_input(
                "Model Name",
                value="llama3-405b",
                help="Look at your model card in SambaStudio and introduce the same name of the model/expert here.",
            )
            st.session_state.llm = f"{llm_model}"

            st.session_state.input_tokens = st.number_input(
                "Number of input tokens", min_value=50, max_value=2000, value=1000, step=1
            )

            st.session_state.output_tokens = st.number_input(
                "Number of output tokens", min_value=50, max_value=2000, value=1000, step=1
            )

            st.session_state.number_requests = st.number_input(
                "Number of total requests", min_value=10, max_value=1000, value=32, step=1
            )

            st.session_state.number_concurrent_workers = st.number_input(
                "Number of concurrent workers", min_value=1, max_value=100, value=1, step=1
            )

            st.session_state.timeout = st.number_input(
                "Timeout", min_value=60, max_value=1800, value=600, step=1
            )

            sidebar_option = st.sidebar.button("Run!")

            if st.button("Back to Setup"):
                st.session_state.setup_complete = False
                st.rerun()

        # Main content
        st.markdown("This performance evaluation assesses the following LLM's performance metrics using concurrent processes. _client represent the metrics computed from the client-side and _server represents the metrics computed from the server-side.")
        st.markdown("**Time to first token (TTFT):** This metric is driven by the time required to process the prompt and then generate the first output token.")
        st.markdown("**E2E Latency:** TTFT + (Time per Output Token) * (the number of tokens to be generated)")
        st.markdown("**Throughput:** Number of output tokens per second across all concurrency requests. Client metric is calculated as *Number of Output Tokens / (E2E Latency - TTFT)*")
        st.markdown("**Total Throughput:** Number of total output tokens per batch and per second")

        if sidebar_option:
            st.toast("Performance evaluation processing now. It should take few minutes.")
            with st.spinner("Processing"):
                try:
                    df_req_info = _run_performance_evaluation()

                    st.subheader("Performance metrics plots")
                    expected_output_tokens = st.session_state.output_tokens
                    generated_output_tokens = df_req_info.server_number_output_tokens.unique()[0]
                    if not pd.isnull(generated_output_tokens):
                        st.markdown(
                            f"Difference between expected output tokens {expected_output_tokens} and generated output tokens {generated_output_tokens} is: {abs(expected_output_tokens-generated_output_tokens)} token(s)"
                        )

                    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 24))
                    plot_client_vs_server_barplots(
                        df_req_info,
                        "batch_size_used",
                        ["server_ttft_s", "client_ttft_s"],
                        "Barplots for Server TTFT and Client TTFT per request",
                        "seconds",
                        ax[0],
                    )
                    plot_client_vs_server_barplots(
                        df_req_info,
                        "batch_size_used",
                        ["server_end_to_end_latency_s", "client_end_to_end_latency_s"],
                        "Barplots for Server latency and Client latency",
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
                        "Barplots for Server token/s and Client token/s per request",
                        "tokens/s",
                        ax[2],
                    )
                    # Compute total throughput per batch
                    plot_dataframe_summary(df_req_info, ax[3])
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error: {e}.")

if __name__ == "__main__":
    main()