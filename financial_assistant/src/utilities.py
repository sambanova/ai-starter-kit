import json
import logging
import os
import time
from typing import Any, Callable, Dict, TypeVar

import streamlit
import yaml
from langchain_classic.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# Generic function type
F = TypeVar('F', bound=Callable[..., Any])


def _get_config_info(config_path: str) -> Dict[str, str]:
    """
    Loads json config file.

    Args:
        path: The path to the config file.
        Defaults to CONFIG_PATH.
    Returns:
        A dictionary with the config information:
            - embedding_model_info:
                String containing embedding model
            - llm_info: Dictionary containing LLM parameters.
            - retrieval_info:
                Dictionary containing retrieval parameters.
            - web_crawling_params:
                Dictionary containing web crawling parameters.
            - extra_loaders:
                List containing extra loader to use when doing web crawling
                (only pdf available in base kit).
    """
    # Read config file
    with open(config_path, 'r') as yaml_file:
        config_file = yaml.safe_load(yaml_file)

    # Convert the config file to a dictionary
    config = dict(config_file)

    return config


# Configure the logger
def get_logger(logger_name: str = 'logger') -> logging.Logger:
    # Get or create a logger instance
    logger = logging.getLogger(logger_name)

    # Ensure that the logger has not been configured with handlers already
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        # Prevent log messages from being propagated to the root logger
        logger.propagate = False

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        general_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(general_formatter)

        logger.addHandler(console_handler)

    return logger


# Get the loggers
logger = get_logger()


def time_llm(func: F) -> Any:
    """
    Decorator to measure the execution time of a function and log the duration.

    Args:
        func: The function to be decorated.

    Returns:
        The wrapped function that returns the original function's return value.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        Wrapper function to execute the timing logic.

        Args:
            *args: Variable length argument list for the decorated function.
            **kwargs: Arbitrary keyword arguments for the decorated function.

        Returns:
            The original function's return value.
        """
        # Initialize time
        start_time = time.time()

        # Call the function
        result = func(*args, **kwargs)

        # End time
        end_time = time.time()

        # Compute duration
        duration = end_time - start_time

        # Log to file
        logger.info(f'{func.__name__} took {duration:.2f} seconds.')

        # Create a row for the csv file
        row = [func.__name__, duration]

        if os.path.exists(streamlit.session_state.time_llm_path):
            # Read the existing data from the JSON file
            with open(streamlit.session_state.time_llm_path, 'r') as file:
                data = json.load(file)
        else:
            # If the file does not exist, start with an empty list
            data = list()

        # Append the row to the list of rows
        data.append(row)

        # Save the new list of rows to a JSON file
        with open(streamlit.session_state.time_llm_path, 'w') as file:
            json.dump(data, file, indent=4)

        # Return only result
        return result

    return wrapper


def load_chat_prompt(path: str) -> ChatPromptTemplate:
    """Load chat prompt from yaml file"""

    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    config.pop('_type')

    template = config.pop('template')

    if not template:
        msg = "Can't load chat prompt without template"
        raise ValueError(msg)

    messages = []
    if isinstance(template, str):
        messages.append(('human', template))

    elif isinstance(template, list):
        for item in template:
            messages.append((item['role'], item['content']))

    return ChatPromptTemplate(messages=messages, **config)
