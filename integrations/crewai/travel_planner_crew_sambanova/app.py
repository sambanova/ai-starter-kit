#!/usr/bin/env python
"""Travel planner based on Agentic AI workflow. Implemented using Gradio and Crew AI
   A deployment is available at https://huggingface.co/spaces/sambanovasystems/trip-planner
"""
import sys
import json
import gradio as gr
import plotly.graph_objects as go
import logging
from crew import TravelCrew, AddressSummaryCrew

def filter_map(text_list, lat, lon):
    fig = go.Figure(go.Scattermapbox(
            lat=lat,
            lon=lon,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=11
            ),
            hovertext=text_list
        ))

    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=lat[1],
                lon=lon[1]
            ),
            pitch=0,
            zoom=10
        ),
    )
    return fig

def run(origin, destination, age, trip_duration, children, budget):
    logger.info(f"Origin: {origin}, Destination: {destination}, Age: {age}, Duration: {trip_duration}, Children: {children}, Daily Budget: {budget}")
    inputs = {
        'origin': origin,
        'destination': destination,
        'age': age,
        'trip_duration': trip_duration,
        'children': children,
        'budget': budget
    }
    result = TravelCrew().crew().kickoff(inputs=inputs)
    inputs_for_address = {
        'text': str(result)
    }

    addresses = AddressSummaryCrew().crew().kickoff(inputs=inputs_for_address)
    json_addresses = None
    if addresses.json_dict:
        json_addresses = addresses.json_dict
    if not json_addresses:
        try:
            json_addresses = json.loads(addresses.raw)
        except json.JSONDecodeError as e:
            # Try with different format of result data generated with ```json and ending with ```
            try:
                json_addresses = json.loads(addresses.raw[8:-4])
            except json.JSONDecodeError as e:
                logger.error("Error loading Crew Output for addresses")
                logger.info(addresses.raw)
                return (result, None)
    fig = filter_map(json_addresses["name"], json_addresses["lat"], json_addresses["lon"])
    return (result, fig)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

demo = gr.Interface(
    title="Plan your itinerary with the help of AI",
    description="Use this app to create a detailed itinerary on how to explore a new place. Itinerary is customized to your taste. Powered by Sambanova Cloud.",
    fn=run,
    inputs=[gr.Textbox(label="Origin"), gr.Textbox(label="Destination"),
            gr.Slider(label="Your age?", value=30, minimum=15, maximum=90, step=5),
            gr.Slider(label="How many days are you travelling?", value=5, minimum=1, maximum=14, step=1),
            gr.Checkbox(label="Check if children are travelling with you"),
            gr.Slider(label="Total budget of trip in USD", show_label=True, value=1000, minimum=500, maximum=10000, step=500)],
    outputs=[
        gr.Textbox(label="Complete Itinerary", show_copy_button=True, autoscroll=False),
        gr.Plot(label="Venues on a Map. Please verify with a Navigation System before traveling.")
    ]
)
demo.launch()
