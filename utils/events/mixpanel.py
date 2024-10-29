import logging
import os
import uuid
from typing import Any, Dict, Optional

import requests

logging.basicConfig(level=logging.INFO)


class MixpanelEvents:
    def __init__(
        self,
        token: Optional[str] = None,
        st_session_id: Optional[str] = None,
        kit_name: Optional[str] = None,
        track: bool = True,
    ) -> None:
        self.url = 'https://api.mixpanel.com/track'
        if token is None:
            self.token = os.getenv('MIXPANEL_TOKEN')
        else:
            self.token = token
        self.track = track
        self.st_session_id = st_session_id
        self.kit_name = kit_name
        if self.token is not None and self.track:
            logging.info('sending mixpanel events')

    def track_event(self, event_name: str, extra_properties: Optional[Dict[str, Any]] = None) -> None:
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

    def input_submitted(self, submission_type: Optional[str] = None) -> None:
        self.track_event('hosted_aisk', {'event_type': 'input_submitted', 'submission_type': submission_type})

    def api_key_saved(self) -> None:
        self.track_event('hosted_aisk', {'event_type': 'api_key_saved'})

    def demo_launch(self) -> None:
        self.track_event('hosted_aisk', {'event_type': 'demo_launch'})
