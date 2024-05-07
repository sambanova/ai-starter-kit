import re
from typing import List, Dict

def get_device_name(sentence: str) -> List[str]:
    """
    Extract device names from the user queries.
    Possible devices: MAX..., MAXM...
                    LT..., LTC..., LTM..., LTP...
                    ADP...
                    RH...

    Args:
        sentence: The user queries

    Returns:
        matches: A list of device names
    """

    # Define a regular expression pattern to match the device name
    pattern = r'\b(?:MAX|MAXM|LT|LTC|LTM|LTP|ADP|RH)\d+\w*(?:-\d+)?\b'

    # Use re.findall to find all matches in the sentence
    matches = re.findall(pattern, sentence)

    return matches

