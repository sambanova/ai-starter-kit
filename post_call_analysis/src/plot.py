from typing import Any, Dict

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from matplotlib.figure import Figure


def plot_diarization(audio_path: str, transcript: pd.DataFrame, dark_mode: bool = True) -> Figure:
    if dark_mode:
        plt.style.use('dark_background')
    else:
        plt.style.use('classic')

    signal, sr = librosa.load(audio_path, sr=None)

    # Calculate the time axis
    time = np.arange(0, len(signal)) / sr

    # Initialize dictionary to store speaker-color mapping
    speaker_colors: Dict[str, Any] = {}

    plot = plt.figure(figsize=(20, 4))

    # plt.plot(time, signal, color='b', linewidth=0.05, label='Audio Signal')  # Entire waveform in blue

    # Iterate over each row in the dataframe
    for index, row in transcript.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        speaker = row['speaker']

        # Check if speaker already has a color assigned
        if speaker not in speaker_colors:
            # Assign a new color for the speaker
            color = plt.cm.get_cmap('tab10')(len(speaker_colors))
            speaker_colors[speaker] = color

        # Find the indices corresponding to start and end time
        start_index = int(start_time * sr)
        end_index = int(end_time * sr)

        # Plot the waveform for the specified time range with the speaker's color
        plt.plot(
            time[start_index:end_index:10],
            signal[start_index:end_index:10],
            color=speaker_colors[speaker],
            linewidth=0.07,
            label=speaker,
        )

    plt.title('Audio Signal Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Create a legend without duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    legend = plt.legend(unique_handles, unique_labels)
    for legent_handles in legend.legend_handles:
        assert legent_handles is not None
        assert hasattr(legent_handles, 'set_linewidth')
        legent_handles.set_linewidth(2)

    return plot


def plot_diarization_plotly(audio_path: str, transcript: pd.DataFrame) -> Any:
    color_scheme = [
        'rgb(31, 119, 180)',
        'rgb(214, 39, 40)',
        'rgb(255, 127, 14)',
        'rgb(44, 160, 44)',
        'rgb(148, 103, 189)',
        'rgb(140, 86, 75)',
        'rgb(227, 119, 194)',
        'rgb(127, 127, 127)',
        'rgb(188, 189, 34)',
        'rgb(23, 190, 207)',
    ]

    signal, sr = librosa.load(audio_path, sr=None)

    # Calculate the time axis
    time = np.arange(0, len(signal)) / sr

    # Initialize dictionary to store speaker-color mapping
    speaker_colors: Dict[str, Any] = {}

    # Initialize dictionary to store aggregated segments for each speaker
    aggregated_segments: Dict[str, Any] = {}

    # Iterate over each row in the dataframe
    for index, row in transcript.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        speaker = row['speaker']

        # Check if speaker already has a color assigned
        if speaker not in speaker_colors:
            # Assign a color from the predefined color scheme
            color_index = len(speaker_colors) % len(color_scheme)
            color = color_scheme[color_index]
            speaker_colors[speaker] = color

        # Find the indices corresponding to start and end time
        start_index = int(start_time * sr)
        end_index = int(end_time * sr)

        # Aggregate segments for each speaker
        if speaker not in aggregated_segments:
            aggregated_segments[speaker] = {'x': [], 'y': []}

        # Insert NaN values to break the line between segments
        if aggregated_segments[speaker]['x'] and aggregated_segments[speaker]['y']:
            aggregated_segments[speaker]['x'].append(np.nan)
            aggregated_segments[speaker]['y'].append(np.nan)

        aggregated_segments[speaker]['x'].extend(time[start_index:end_index:20])
        aggregated_segments[speaker]['y'].extend(signal[start_index:end_index:20])

    # Create traces list to store all the aggregated segments
    traces = []

    # Create traces for each aggregated segment
    for speaker, data in aggregated_segments.items():
        trace = go.Scatter(
            x=data['x'],  # Downsampled for better performance
            y=data['y'],
            mode='lines',
            line=dict(color=speaker_colors[speaker], width=1),
            name=speaker,
        )
        traces.append(trace)

    layout = go.Layout(
        title='Diarized Audio Signal Waveform', xaxis=dict(title='Time (s)'), yaxis=dict(title='Amplitude')
    )

    # Plot the waveform with Plotly
    fig = go.Figure(data=traces, layout=layout)

    return fig


def plot_quallity_gauge(score: Any) -> Any:
    # Create a gauge chart
    fig = go.Figure(
        go.Indicator(
            mode='gauge+number',
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            # title = {'text': "Call Quality Assessment", 'font': {'size': 35}},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#6699cc'},  # Soft blue color
                'borderwidth': 1,
                'steps': [
                    {'range': [0, 50], 'color': '#ff6666'},  # Soft red color
                    {'range': [50, 80], 'color': '#ffcc66'},  # Soft yellow color
                    {'range': [80, 100], 'color': '#99cc66'},
                ],  # Soft green color
                'threshold': {'line': {'color': 'Green', 'width': 3}, 'thickness': 0.9, 'value': score},
            },
        )
    )
    fig.update_layout(width=450, height=300)
    return fig
