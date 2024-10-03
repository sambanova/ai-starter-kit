import netrc
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


def initialize_env_variables(prod_mode: bool = False, additional_env_vars: Optional[List[str]] = None) -> None:
    if additional_env_vars is None:
        additional_env_vars = []

    if not prod_mode:
        # In non-prod mode, prioritize environment variables
        st.session_state.SAMBANOVA_API_KEY = os.environ.get(
            'SAMBANOVA_API_KEY', st.session_state.get('SMABANOVA_API_KEY', '')
        )
        for var in additional_env_vars:
            st.session_state[var] = os.environ.get(var, st.session_state.get(var, ''))
    else:
        # In prod mode, only use session state
        if 'SAMBANOVA_API_KEY' not in st.session_state:
            st.session_state.SAMBANOVA_API_KEY = ''
        for var in additional_env_vars:
            if var not in st.session_state:
                st.session_state[var] = ''


def set_env_variables(api_key: str, additional_vars: Optional[Dict[str, Any]] = None, prod_mode: bool = False) -> None:
    st.session_state.SAMBANOVA_API_KEY = api_key
    if additional_vars:
        for key, value in additional_vars.items():
            st.session_state[key] = value
    if not prod_mode:
        # In non-prod mode, also set environment variables
        os.environ['SAMBANOVA_API_KEY'] = api_key
        if additional_vars:
            for key, value in additional_vars.items():
                os.environ[key] = value


def env_input_fields(additional_env_vars: Optional[List[str]] = None) -> Tuple[str, str]:
    if additional_env_vars is None:
        additional_env_vars = []

    api_key = st.text_input('Sambanova API Key', value=st.session_state.SAMBANOVA_API_KEY, type='password')

    additional_vars = {}
    for var in additional_env_vars:
        additional_vars[var] = st.text_input(f'{var}', value=st.session_state.get(var, ''), type='password')

    return api_key, additional_vars


def are_credentials_set(additional_env_vars: Optional[List[str]] = None) -> bool:
    if additional_env_vars is None:
        additional_env_vars = []

    base_creds_set = bool(st.session_state.SAMBANOVA_API_KEY)
    additional_creds_set = all(bool(st.session_state.get(var, '')) for var in additional_env_vars)

    return base_creds_set and additional_creds_set


def save_credentials(api_key: str, additional_vars: Optional[Dict[str, Any]] = None, prod_mode: bool = False) -> str:
    set_env_variables(api_key, additional_vars, prod_mode)
    return 'Credentials saved successfully!'


def get_wandb_key() -> Optional[Any]:
    # Check for WANDB_API_KEY in environment variables
    env_wandb_api_key = os.getenv('WANDB_API_KEY')

    # Check for WANDB_API_KEY in ~/.netrc
    try:
        netrc_path = os.path.expanduser('~/.netrc')
        netrc_data = netrc.netrc(netrc_path)
        netrc_wandb_api_key = netrc_data.authenticators('api.wandb.ai')
    except (FileNotFoundError, netrc.NetrcParseError):
        netrc_wandb_api_key = None

    # If both are set, handle the conflict
    if env_wandb_api_key and netrc_wandb_api_key:
        print('WANDB_API_KEY is set in both the environment and ~/.netrc. Prioritizing environment variable.')
        # Optionally, you can choose to remove one of them, here we remove the env variable
        del os.environ['WANDB_API_KEY']  # Remove from environment to prioritize ~/.netrc
        return netrc_wandb_api_key[2] if netrc_wandb_api_key else None  # Return the key from .netrc

    # Return the key from environment if available, otherwise from .netrc
    if env_wandb_api_key:
        return env_wandb_api_key
    elif netrc_wandb_api_key:
        return netrc_wandb_api_key[2] if netrc_wandb_api_key else None

    # If neither is set, return None
    return None
