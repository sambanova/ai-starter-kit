import pandas as pd



def rename_metrics_df(valid_df: pd.DataFrame) -> pd.DataFrame:
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
    final_df["server_ttft_s"] = valid_df["server_ttft_s"]
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


def transform_df_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
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