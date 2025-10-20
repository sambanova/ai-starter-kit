import netrc
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st


def initialize_env_variables(
    prod_mode: bool = False, additional_env_vars: Optional[Union[List[str], Dict[str, Any]]] = None
) -> None:
    if additional_env_vars is None:
        additional_env_vars = []

    if not prod_mode:
        # In non-prod mode, prioritize environment variables
        st.session_state.SAMBANOVA_API_KEY = os.environ.get(
            'SAMBANOVA_API_KEY', st.session_state.get('SAMBANOVA_API_KEY', '')
        )
        
        if isinstance(additional_env_vars, List):
            for var in additional_env_vars:
                st.session_state[var] = os.environ.get(var, st.session_state.get(var, ''))
        elif isinstance(additional_env_vars, Dict):
            for key, val in additional_env_vars.items():
                st.session_state[key] = os.environ.get(key, st.session_state.get(key, val))
        else:
            raise ValueError('additional_env_vars must be a List or Dict')
    else:
        # In prod mode, only use session state
        if 'SAMBANOVA_API_KEY' not in st.session_state:
            st.session_state.SAMBANOVA_API_KEY = ''

        if isinstance(additional_env_vars, List):
            for var in additional_env_vars:
                if var not in st.session_state:
                    st.session_state[var] = ''
        elif isinstance(additional_env_vars, Dict):
            for key, val in additional_env_vars.items():
                if key not in st.session_state:
                    st.session_state[key] = val
        else:
            raise ValueError('additional_env_vars must be a List or Dict')


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


def env_input_fields(
    additional_env_vars: Union[List[str], Dict[str, Any]] = None
) -> Tuple[str, Any]:
    if additional_env_vars is None:
        additional_env_vars = []
    additional_vars = {}

    if isinstance(additional_env_vars, List):
        # If SAMBANOVA_BASE_URL in additional env vars, show it first and then api key
        if 'SAMBANOVA_BASE_URL' in additional_env_vars: 
            additional_vars['SAMBANOVA_BASE_URL'] = st.text_input(
                'SAMBANOVA BASE URL', value=st.session_state.get('SAMBANOVA_BASE_URL', ''), type='password'
            )
            api_key = st.text_input(
                'SAMBANOVA API KEY', value=st.session_state.get('SAMBANOVA_API_KEY', ''), type='password'
            )
        # If SAMBANOVA_BASE_URL not in additional env vars, show api key only
        else:
            api_key = st.text_input(
                'SAMBANOVA API KEY', value=st.session_state.get('SAMBANOVA_API_KEY', ''), type='password'
            )
        for var in additional_env_vars:
            # Skip SAMBANOVA_BASE_URL since it has been handled above
            if 'SAMBANOVA_BASE_URL' == var:
                continue
            additional_vars[var] = st.text_input(f'{var}', value=st.session_state.get(var, ''), type='password')

    elif isinstance(additional_env_vars, Dict):
        # If SAMBANOVA_BASE_URL in additional env vars, show it first and then  api key
        if 'SAMBANOVA_BASE_URL' in additional_env_vars:
            additional_vars['SAMBANOVA_BASE_URL'] = st.text_input(
                'SAMBANOVA BASE URL', value=st.session_state.get('SAMBANOVA_BASE_URL', ''), type='password'
            )
            api_key = st.text_input(
                'SAMBANOVA API KEY', value=st.session_state.get('SAMBANOVA_API_KEY', ''), type='password'
            )
        # If SAMBANOVA_BASE_URL not in additional env vars, show cloud api key only
        else:
            api_key = st.text_input(
                'SAMBANOVA API KEY', value=st.session_state.get('SAMBANOVA_API_KEY', ''), type='password'
            )
        for key, val in additional_env_vars.items():
            # Skip SAMBANOVA_BASE_URL since it has been handled above
            if 'SAMBANOVA_BASE_URL' == key:
                continue
            additional_vars[key] = st.text_input(f'{key}', value=st.session_state.get(key, val), type='password')

    return api_key, additional_vars


def are_credentials_set(additional_env_vars: Optional[Union[List[str], Dict[str, Any]]] = None) -> bool:
    if additional_env_vars is None:
        additional_env_vars = []

    base_creds_set = bool(st.session_state.SAMBANOVA_API_KEY)

    if isinstance(additional_env_vars, List):
        additional_creds_set = all(bool(st.session_state.get(var, '')) for var in additional_env_vars)
    elif isinstance(additional_env_vars, Dict):
        additional_creds_set = all(bool(st.session_state.get(key, '')) for key, _ in additional_env_vars.items())

    return base_creds_set and additional_creds_set


def save_credentials(api_key: str, additional_vars: Optional[Dict[str, Any]] = None, prod_mode: bool = False) -> str:
    set_env_variables(api_key, additional_vars, prod_mode)
    return 'Credentials saved successfully!'


import netrc
import os
from typing import Optional


def get_wandb_key() -> Optional[str]:
    """
    Retrieve the Weights & Biases API key from the environment or ~/.netrc,
    and remove WANDB_API_KEY from the environment to prevent conflicts with weave.

    Returns:
        The API key if found, otherwise None.
    """
    # Check for WANDB_API_KEY in environment variables
    env_wandb_api_key = os.environ.pop('WANDB_API_KEY', None)

    # Check for WANDB_API_KEY in ~/.netrc
    netrc_wandb_api_key = None
    try:
        netrc_path = os.path.expanduser('~/.netrc')
        netrc_data = netrc.netrc(netrc_path)
        auth = netrc_data.authenticators('api.wandb.ai')
        if auth and len(auth) == 3:
            netrc_wandb_api_key = auth[2]  # The password (API key) is the third element
    except (FileNotFoundError, netrc.NetrcParseError):
        pass

    # Return the API key from the environment variable if it was set
    if env_wandb_api_key:
        return env_wandb_api_key
    elif netrc_wandb_api_key:
        return netrc_wandb_api_key

    # If neither is set, return None
    return None
