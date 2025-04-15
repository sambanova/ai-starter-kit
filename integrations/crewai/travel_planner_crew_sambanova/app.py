"""
Travel planner based on Agentic AI workflow.

This module deploys a portal which can customize a day to day travel itinerary
for a person using multiple specialized AI crews.

Implemented using Sambanova Cloud, Gradio and Crew AI.
A deployment is available at https://huggingface.co/spaces/sambanovasystems/trip-planner
"""

import datetime
import json
import logging
from typing import List, Tuple

import gradio as gr  # type: ignore
import plotly.graph_objects as go

from .crews.crew import AddressSummaryCrew, TravelCrew


def filter_map(text_list: List[str], lat: List[str], lon: List[str]) -> go.Figure:
    """
    Create a Map showing the points specified in the inputs.

    Args:
        text_list: List of the description of all locations that will be shown on the map.
        lat:       List of latitude coordinates of the locations.
        lon:       List of longitude coordinates of the locations.

    Returns:
        Figure: Map with the points specified in the inputs
    """

    # Creating a map with the provided markers using their latitude and longitude coordinates.
    fig = go.Figure(
        go.Scattermapbox(lat=lat, lon=lon, mode='markers', marker=go.scattermapbox.Marker(size=11), hovertext=text_list)
    )

    # Update the map by centering it on of the the provided longitude and latitude coordinates
    fig.update_layout(
        mapbox_style='open-street-map',
        hovermode='closest',
        mapbox=dict(bearing=0, center=go.layout.mapbox.Center(lat=lat[1], lon=lon[1]), pitch=0, zoom=10),
    )
    return fig


def run(
    origin: str,
    destination: str,
    arrival_date: int,
    age: int,
    trip_duration: int,
    interests: List[str],
    cuisine_preferences: List[str],
    children: bool,
    budget: int,
) -> Tuple[str, go.Figure]:
    """
    Run the specfied query using Crew AI agents.

    Args:
        origin: Origin city of the traveller.
        destination: Destination to which the traveller is going.
        arrival_date: Approximate date when the trip will begin in epoch time.
        age: Age profile of traveller.
        interests: Specific interests of the traveller.
        cuisine_preferences: Specific cuisine preferences of the traveller.
        children: Whether traveller has children travelling with them.
        budget: Total budget of traveller in US Dollars.

    Returns:
        Returns a tuple containing the itinerary and map
    """
    arrival_date_input = datetime.datetime.fromtimestamp(arrival_date).strftime('%m-%d-%Y')
    logger.info(
        f'Origin: {origin}, Destination: {destination}, Arrival Date: {arrival_date_input},'
        f' Age: {age}, Duration: {trip_duration},'
        f' Interests: {interests}, Cuisines: {cuisine_preferences},'
        f' Children: {children}, Daily Budget: {budget}'
    )

    # Creating a dictionary of user provided preferences and providing these to the crew agents
    # to work on.

    user_preferences = {
        'origin': origin,
        'destination': destination,
        'arrival_date': arrival_date_input,
        'age': age,
        'trip_duration': trip_duration,
        'interests': interests,
        'cuisine_preferences': cuisine_preferences,
        'children': children,
        'budget': budget,
    }
    result = TravelCrew().crew().kickoff(inputs=user_preferences)

    """
        Now we will pass the result to a address summary crew whose job is to extract position
        coordinates of the addresses (latitude and longitude), so that the addresses in the
        result can be displayed in map coordinates
    """

    inputs_for_address = {'text': str(result)}

    addresses = AddressSummaryCrew().crew().kickoff(inputs=inputs_for_address)

    """
        We have requested the crew agent to return latitude, longitude coordinates.
        But the exact way the LLMs return varies. Hence we try multiple different ways of
        extracting addresses in JSON format from the result.
    """
    json_addresses = None
    if addresses.json_dict is not None:
        json_addresses = addresses.json_dict
    if json_addresses is None:
        try:
            json_addresses = json.loads(addresses.raw)
        except json.JSONDecodeError as e:
            # Try with different format of result data generated with ```json and ending with ```.
            try:
                json_addresses = json.loads(addresses.raw[8:-4])
            except json.JSONDecodeError as e:
                # Try with different format of result data generated with ``` and ending with ```.
                try:
                    json_addresses = json.loads(addresses.raw[4:-4])
                except json.JSONDecodeError as e:
                    logger.error('Error loading Crew Output for addresses')
                    logger.info(addresses.raw)
                    return (result, None)
    fig = filter_map(json_addresses.get('name'), json_addresses.get('lat'), json_addresses.get('lon'))
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
        gr.DateTime(label='Approximate arrival date'),
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
        gr.Textbox(
            label='Complete Personalized Itinerary of your Trip',
            show_label=True,
            show_copy_button=True,
            autoscroll=False,
        ),
        gr.Plot(label='Venues on a Map. Please verify with a Navigation System before traveling.'),
    ],
)
demo.launch()
