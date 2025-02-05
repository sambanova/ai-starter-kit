#!/usr/bin/env python
"""
Travel planner based on Agentic AI workflow.

This module deploys a portal which can customize a day to day travel itinerary
for a person using multiple specialized AI crews.

Implemented using Gradio and Crew AI
A deployment is available at https://huggingface.co/spaces/sambanovasystems/trip-planner
"""

import json
import logging
from typing import List

import gradio as gr
import plotly.graph_objects as go
from crew import AddressSummaryCrew, TravelCrew


def filter_map(text_list: List[str], lat: List[str], lon: List[str]) -> go.Figure:
    """
    Create a Map showing the points specified in the inputs.

    Args:
        text_list: List of the description of all locations that will be shown on the map
        lat:       List of latitude coordinates of the locations
        lon:       List of longitude coordinates of the locations

    Returns:
        Figure: Map with the points specified in the inputs
    """

    fig = go.Figure(
        go.Scattermapbox(lat=lat, lon=lon, mode='markers', marker=go.scattermapbox.Marker(size=11), hovertext=text_list)
    )

    fig.update_layout(
        mapbox_style='open-street-map',
        hovermode='closest',
        mapbox=dict(bearing=0, center=go.layout.mapbox.Center(lat=lat[1], lon=lon[1]), pitch=0, zoom=10),
    )
    return fig


def run(
    origin: str,
    destination: str,
    age: int,
    trip_duration: int,
    interests: List,
    cuisine_preferences: List,
    children: bool,
    budget: int,
) -> (str, go.Figure):
    """
    Run the specfied query using Crew AI agents

    Args:
        origin: Origin city of the traveller
        destination: Destination that traveller is going to
        age: Age profile of traveller
        interests: Specific interests of the traveller
        cuisine_preferences: Specific cuisine preferences of the traveller
        children: Whether traveller has children travelling with them
        budget: Total budget of traveller in US Dollars

    Returns:
        Returns a tuple containing the itinerary and map
    """
    logger.info(
        f'Origin: {origin}, Destination: {destination}, Age: {age}, Duration: {trip_duration},'
        f' Interests: {interests}, Cuisines: {cuisine_preferences}, Children: {children}, Daily Budget: {budget}'
    )
    inputs = {
        'origin': origin,
        'destination': destination,
        'age': age,
        'trip_duration': trip_duration,
        'interests': interests,
        'cuisine_preferences': cuisine_preferences,
        'children': children,
        'budget': budget,
    }
    result = TravelCrew().crew().kickoff(inputs=inputs)
    inputs_for_address = {'text': str(result)}

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
                # Try with different format of result data generated with ``` and ending with ```
                try:
                    json_addresses = json.loads(addresses.raw[4:-4])
                except json.JSONDecodeError as e:
                    logger.error('Error loading Crew Output for addresses')
                    logger.info(addresses.raw)
                    return (result, None)
    fig = filter_map(json_addresses['name'], json_addresses['lat'], json_addresses['lon'])
    return (result, fig)


logger = logging.getLogger()
logger.setLevel(logging.INFO)

demo = gr.Interface(
    title='Plan your itinerary with the help of AI',
    description='Use this app to create a detailed itinerary on how to explore a new place.'
    ' Itinerary is customized to your taste. Powered by Sambanova Cloud.',
    fn=run,
    inputs=[
        gr.Textbox(label='Where are you travelling from?'),
        gr.Textbox(label='Where are you going?'),
        gr.Slider(label='Your age?', value=30, minimum=15, maximum=90, step=5),
        gr.Slider(label='How many days are you travelling?', value=5, minimum=1, maximum=14, step=1),
        gr.CheckboxGroup(
            ['Museums', 'Shopping', 'Entertainment', 'Nightlife', 'Outdoor Adventures'],
            label='Checkbox your specific interests.',
        ),
        gr.CheckboxGroup(
            [
                'Ethnic',
                'American',
                'Italian',
                'Mexican',
                'Chinese',
                'Japanese',
                'Indian',
                'Thai',
                'French',
                'Vietnamese',
                'Vegan',
            ],
            label='Checkbox your cuisine preferences.',
        ),
        gr.Checkbox(label='Check if children are travelling with you'),
        gr.Slider(
            label='Total budget of trip in USD', show_label=True, value=1000, minimum=500, maximum=10000, step=500
        ),
    ],
    outputs=[
        gr.Textbox(label='Complete Personalized Itinerary of your Trip', show_copy_button=True, autoscroll=False),
        gr.Plot(label='Venues on a Map. Please verify with a Navigation System before traveling.'),
    ],
)
demo.launch()
