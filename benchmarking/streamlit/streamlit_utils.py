import pandas as pd
from typing import List
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_dataframe_summary(df_req_info: pd.DataFrame):
    """
    Plots a throughput summary across all batch sizes

    Args:
        df_req_info (pd.DataFrame): The DataFrame containing the data to plot.

    Returns:
        fig (go.Figure): The plotly figure container
    """   
    df_req_summary = (
        df_req_info.groupby('batch_size_used')[
            [
                'server_output_token_per_s_per_request',
                'client_output_token_per_s_per_request',
            ]
        ]
        .mean()
        .reset_index()
    ).rename(
        columns={
            'server_output_token_per_s_per_request': 'server_output_token_per_s_mean',
            'client_output_token_per_s_per_request': 'client_output_token_per_s_mean',
        }
    )
    df_req_summary['server_throughput_token_per_s'] = (
        df_req_summary['server_output_token_per_s_mean'] * df_req_summary['batch_size_used']
    )
    df_req_summary['client_throughput_token_per_s'] = (
        df_req_summary['client_output_token_per_s_mean'] * df_req_summary['batch_size_used']
    )
    df_req_summary.rename(columns={
        "batch_size_used": "Batch size",
        "server_throughput_token_per_s": "Server",
        "client_throughput_token_per_s": "Client",
    }, inplace=True)
    df_melted = pd.melt(
        df_req_summary,
        id_vars="Batch size",
        value_vars=["Server", "Client"],
        var_name="Side type",
        value_name="Total throughput (tokens per second)",
    )
    df_melted["Batch size"] = [str(x) for x in df_melted["Batch size"]]
    fig = px.bar(
        df_melted, 
        x="Batch size", 
        y="Total throughput (tokens per second)", 
        color="Side type", 
        barmode="group",
        color_discrete_sequence=["#325c8c", "#ee7625"]
    )
    fig.update_layout(
        title_text="Total throughput per batch size",
        template="plotly_dark",
    )
    return fig

def plot_client_vs_server_barplots(df_user: pd.DataFrame, x_col: str, y_cols: List[str], legend_labels: List[str], title: str, ylabel: str, xlabel: str) -> None:
    """
    Plots bar plots for client vs server metrics from a DataFrame.

    Args:
        df_user (pd.DataFrame): The DataFrame containing the data to plot.
        x_col (str): The column name to be used as the x-axis.
        y_cols (List[str]): A list of column names to be used as the y-axis.
        legend_labels (List[str]): Human-readable labels for each grouping in y_cols.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        xlabel (str): The label for the x-axis.

    Returns:
        fig (go.Figure): The plotly figure container
    """    
    value_vars = y_cols
    title_text = title
    yaxis_title = ylabel
    xaxis_title = xlabel

    df_melted = df_user.melt(
        id_vars=[x_col], 
        value_vars=value_vars, 
        var_name='Metric', 
        value_name='Value',
    )
    xgroups = [str(x) for x in sorted(pd.unique(df_melted[x_col]))]
    df_melted[x_col] = [str(x) for x in df_melted[x_col]]

    valsl = {}
    valsr = {}
    for i in xgroups:
        maskl = (df_melted["Metric"] == value_vars[0]) & (df_melted[x_col] == i)
        valsl[i] = np.percentile(df_melted["Value"][maskl], [5, 50, 95])
        maskr = (df_melted["Metric"] == value_vars[1]) & (df_melted[x_col] == i)
        valsr[i] = np.percentile(df_melted["Value"][maskr], [5, 50, 95])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x = xgroups,
            y = [0 for _ in xgroups],
            base = [valsl[i][1] for i in xgroups],
            customdata=[legend_labels[0] for _ in xgroups],
            marker={"color":"#325c8c","line":{"color":"#325c8c", "width":2}},
            offsetgroup=0,
            legendgroup=legend_labels[0],
            name=legend_labels[0],
            showlegend=False,
            hovertemplate="<extra></extra><b>%{customdata}</b> median: %{base:.2f}",
        )
    )
    fig.add_trace(
        go.Bar(
            x = xgroups,
            y = [valsl[i][2] - valsl[i][0] for i in xgroups],
            base = [valsl[i][0] for i in xgroups],
            customdata = [valsl[i][2] for i in xgroups],
            marker={"color":"#325c8c"},
            opacity=0.5,
            offsetgroup=0,
            legendgroup=legend_labels[0],
            name=legend_labels[0],
            hovertemplate="<extra></extra>5–95 pctile range: %{base:.2f}–%{customdata:.2f}",
        )
    )
    fig.add_trace(
        go.Bar(
            x = xgroups,
            y= [0 for _ in xgroups],
            base = [valsr[i][1] for i in xgroups],
            customdata=[legend_labels[1] for _ in xgroups],
            marker={"color":"#ee7625","line":{"color":"#ee7625", "width":2}},
            offsetgroup=1,
            legendgroup=legend_labels[1],
            name=legend_labels[1],
            showlegend=False,
            hovertemplate="<extra></extra><b>%{customdata}</b> median: %{base:.2f}",
        )
    )
    fig.add_trace(
        go.Bar(
            x = xgroups,
            y = [valsr[i][2] - valsr[i][0] for i in xgroups],
            base = [valsr[i][0] for i in xgroups],
            customdata = [valsr[i][2] for i in xgroups],
            marker={"color":"#ee7625"},
            opacity=0.5,
            offsetgroup=1,
            legendgroup=legend_labels[1],
            name=legend_labels[1],
            hovertemplate="<extra></extra>5–95 pctile range: %{base:.2f}–%{customdata:.2f}",
        )
    )

    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        barmode="group",
        template="plotly_dark",
        hovermode="x unified",
    )

    fig.update_xaxes(hoverformat="foo")
    return fig

def plot_requests_gantt_chart(df_user: pd.DataFrame):
    """
    Plots a Gantt chart of response timings across all requests

    Args:
        df_user (pd.DataFrame): The DataFrame containing the data to plot.

    Returns:
        fig (go.Figure): The plotly figure container
    """    
    requests = df_user.index+1
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=requests,
            x=1000*df_user["client_ttft_s"],
            base=[str(x) for x in df_user["start_time"]],
            name="TTFT",
            orientation="h",
            marker_color="#ee7625",
        )
    )
    fig.add_trace(
        go.Bar(
            y=requests,
            x=1000*df_user["client_end_to_end_latency_s"],
            base=[str(x) for x in df_user["start_time"]],
            name="End-to-end latency",
            orientation="h",
            marker_color="#325c8c",
        )
    )
    for i in range(0, len(df_user.index), 2):
        fig.add_hrect(y0=i+0.5, y1=i+1.5, line_width=0, fillcolor="grey", opacity=0.1)
    fig.update_xaxes(
        type="date",
        tickformat="%H:%M:%S",
        hoverformat="%H:%M:%S.%2f",)
    fig.update_layout(
        title_text="LLM requests across time",
        xaxis_title="Time stamp",
        yaxis_title="Request index",
        template="plotly_dark",
    )
    return fig
