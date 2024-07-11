from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List


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
    ).rename(
        columns={
            "server_output_token_per_s_per_request": "server_output_token_per_s_mean",
            "client_output_token_per_s_per_request": "client_output_token_per_s_mean",
        }
    )
    df_req_summary["server_throughput_token_per_s"] = (
        df_req_summary["server_output_token_per_s_mean"]
        * df_req_summary["batch_size_used"]
    )
    df_req_summary["client_throughput_token_per_s"] = (
        df_req_summary["client_output_token_per_s_mean"]
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
    sns.barplot(
        x="batch_size_used", y="Value", hue="Value Type", data=df_melted, ax=ax
    ).set(
        xlabel="Batch Size Used",
        ylabel="tokens/s",
        title="Total throughput per batch",
    )

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
    sns.barplot(data=df_melted, x=x_col, y="Value", hue="Metric", ax=ax).set(
        xlabel="Batch Size Used",
        ylabel=ylabel,
        title=title,
    )
    ax.legend(title="Metric")
