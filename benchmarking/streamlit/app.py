import os
import sys
import uuid

import yaml

sys.path.append('../')
sys.path.append('./src')
sys.path.append('./streamlit')

import warnings
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv
from st_pages import Page, hide_pages, show_pages

from benchmarking.streamlit.streamlit_utils import APP_PAGES
from benchmarking.utils import CONFIG_PATH, SAMBANOVA_URL
from utils.events.mixpanel import MixpanelEvents
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))


@st.cache_data
def _init() -> None:
    load_dotenv('../.env', override=True)


with open(CONFIG_PATH) as file:
    st.session_state.config = yaml.safe_load(file)
    st.session_state.prod_mode = st.session_state.config['prod_mode']
    st.session_state.pages_to_show = st.session_state.config['pages_to_show']


def _initialize_session_variables() -> None:
    if 'prod_mode' not in st.session_state:
        st.session_state.prod_mode = None
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = None
    if 'st_session_id' not in st.session_state:
        st.session_state.st_session_id = str(uuid.uuid4())
    if 'mp_events' not in st.session_state:
        st.session_state.mp_events = MixpanelEvents(
            os.getenv('MIXPANEL_TOKEN'),
            st_session_id=st.session_state.st_session_id,
            kit_name='benchmarking',
            track=st.session_state.prod_mode,
        )
        st.session_state.mp_events.demo_launch()


def main() -> None:
    show_pages(
        [
            Page(APP_PAGES['setup']['file_path'], APP_PAGES['setup']['page_label']),
            Page(APP_PAGES['synthetic_eval']['file_path'], APP_PAGES['synthetic_eval']['page_label']),
            Page(APP_PAGES['real_workload_eval']['file_path'], APP_PAGES['real_workload_eval']['page_label']),
            Page(APP_PAGES['custom_eval']['file_path'], APP_PAGES['custom_eval']['page_label']),
            Page(APP_PAGES['chat_eval']['file_path'], APP_PAGES['chat_eval']['page_label']),
        ]
    )

    prod_mode = st.session_state.prod_mode

    if prod_mode:
        if not st.session_state.setup_complete:
            hide_pages(
                [
                    APP_PAGES['synthetic_eval']['page_label'],
                    APP_PAGES['real_workload_eval']['page_label'],
                    APP_PAGES['custom_eval']['page_label'],
                    APP_PAGES['chat_eval']['page_label'],
                ]
            )

            st.title('Setup')

            # Callout to get SambaNova API Key
            st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')

            # Mode selection
            st.session_state.mode = st.radio('Select Mode', ['SambaNova Cloud', 'SambaStudio'])

            additional_env_vars: Dict[str, Any] = {}
            additional_env_vars = {'SAMBANOVA_URL': SAMBANOVA_URL}

            if st.session_state.mode == 'SambaNova Cloud':
                st.session_state.llm_api = 'sncloud'
            else:  # SambaStudio
                st.session_state.llm_api = 'sambastudio'

            initialize_env_variables(prod_mode, additional_env_vars)

            if not are_credentials_set(additional_env_vars):
                api_key, additional_vars = env_input_fields(additional_env_vars, st.session_state.mode)

                if st.button('Save Credentials'):
                    if st.session_state.mode == 'SambaNova Cloud':
                        message = save_credentials(api_key, additional_vars, prod_mode)
                    else:  # SambaStudio
                        additional_vars['SAMBASTUDIO_API_KEY'] = api_key
                        message = save_credentials(api_key, additional_vars, prod_mode)
                    st.success(message)
                    st.session_state.setup_complete = True
                    st.session_state.mp_events.api_key_saved()
                    st.rerun()
            else:
                st.success('Credentials are set')
                if st.button('Clear Credentials'):
                    if st.session_state.mode == 'SambaNova Cloud':
                        save_credentials('', None, prod_mode)
                    else:
                        save_credentials('', {var: '' for var in additional_env_vars}, prod_mode)
                    st.session_state.setup_complete = False
                    st.rerun()
                if st.button('Continue to App'):
                    st.session_state.setup_complete = True
                    st.rerun()

        else:
            st.switch_page('pages/synthetic_performance_eval_st.py')
    else:
        st.switch_page('pages/synthetic_performance_eval_st.py')


if __name__ == '__main__':
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
    )

    _init()
    _initialize_session_variables()

    main()
