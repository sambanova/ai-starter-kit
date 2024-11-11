import logging
import os
import uuid
from typing import Any, Dict, Optional

import requests

logging.basicConfig(level=logging.INFO)

MIXPANEL_TRACK_URL = 'https://api.mixpanel.com/track'


class MixpanelEvents:
    """class to handle Mixpanel event tracking for a specific application."""

    def __init__(
        self,
        token: Optional[str] = None,
        st_session_id: Optional[str] = None,
        kit_name: Optional[str] = None,
        track: bool = True,
    ) -> None:
        """
        Initializes the MixpanelEvents instance.

        Args:
            token (Optional[str]): The Mixpanel project token.
            If not provided, it is fetched from the environment variable 'MIXPANEL_TOKEN'.
            st_session_id (Optional[str]): The streamlit session ID for tracking.
            kit_name (Optional[str]): The name of the kit.
            track (bool): A flag to enable or disable event tracking. Defaults to True.
        """
        if token is None:
            token = os.getenv('MIXPANEL_TOKEN')
        self.token = token
        self.url = MIXPANEL_TRACK_URL
        self.track = track

        if self.track:
            if self.token is not None:
                logging.info('sending mixpanel events')
            else:
                logging.warning('Mixpanel token not provided or not found in environment variables')

        self.st_session_id = st_session_id
        self.kit_name = kit_name

    def track_event(self, event_name: str, extra_properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Tracks a Mixpanel event.

        Args:
            event_name (str): The name of the event to track.
            extra_properties (Optional[Dict[str, Any]]): Additional properties to include with the event.
            Defaults to None.
        """
        if self.track:
            if extra_properties is None:
                extra_properties = {}
            payload = {
                'event': event_name,
                'properties': {
                    'token': self.token,
                    '$insert_id': str(uuid.uuid4()),
                    'kit_name': self.kit_name,
                    'st_session_id': self.st_session_id,
                    **extra_properties,
                },
            }
            payload = {k: v for k, v in payload.items() if v is not None}
            headers = {'Content-Type': 'application/json', 'Accept': 'text/plain'}
            response = requests.post(self.url, json=[payload], headers=headers)

    def input_submitted(
        self, submission_type: Optional[str] = None, extra_properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Tracks an 'input_submitted' event.

        Args:
            submission_type (Optional[str]): The type of submission.
            extra_properties (Optional[Dict[str, Any]]): Additional properties to include with the event.
            Defaults to None.
        """
        if extra_properties is None:
            extra_properties = {}
        self.track_event(
            'hosted_aisk', {'event_type': 'input_submitted', 'submission_type': submission_type, **extra_properties}
        )

    def api_key_saved(self, extra_properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Tracks an 'api_key_saved' event.

        Args:
            extra_properties (Optional[Dict[str, Any]]): Additional properties to include with the event.
            Defaults to None.
        """
        if extra_properties is None:
            extra_properties = {}
        self.track_event('hosted_aisk', {'event_type': 'api_key_saved', **extra_properties})

    def demo_launch(self, extra_properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Tracks a 'demo_launch' event.

        Args:
            extra_properties (Optional[Dict[str, Any]]): Additional properties to include with the event.
            Defaults to None.
        """
        if extra_properties is None:
            extra_properties = {}
        self.track_event('hosted_aisk', {'event_type': 'demo_launch', **extra_properties})
