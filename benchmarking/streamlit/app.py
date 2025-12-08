import os
import sys
import uuid

import yaml

sys.path.append('../')
sys.path.append('./src')
sys.path.append('./streamlit')

import warnings

import streamlit as st
from dotenv import load_dotenv

from benchmarking.streamlit.streamlit_utils import APP_PAGES, find_pages_to_show, render_logo, set_font
from benchmarking.utils import CONFIG_PATH
from utils.events.mixpanel import MixpanelEvents

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


@st.cache_resource
def _get_mixpanel_client(token: str | None, st_session_id: str, prod_mode: bool) -> MixpanelEvents:
    """Cache the MixpanelEvents client so it persists across reruns in a session."""
    mp = MixpanelEvents(
        token,
        st_session_id=st_session_id,
        kit_name='benchmarking',
        track=prod_mode,
    )
    mp.demo_launch()
    return mp


def _initialize_session_variables() -> None:
    if 'prod_mode' not in st.session_state:
        st.session_state.prod_mode = None
    if 'st_session_id' not in st.session_state:
        st.session_state.st_session_id = str(uuid.uuid4())
    if 'mp_events' not in st.session_state:
        token = os.getenv('MIXPANEL_TOKEN')
        st.session_state.mp_events = _get_mixpanel_client(
            token,
            st.session_state.st_session_id,
            st.session_state.prod_mode,
        )


def main() -> None:
    render_logo()
    set_font()


if __name__ == '__main__':
    synthetic_page = st.Page(
        APP_PAGES['synthetic_eval']['file_path'],
        title=APP_PAGES['synthetic_eval']['page_label'],
        icon=APP_PAGES['synthetic_eval']['page_icon'],
    )
    real_page = st.Page(
        APP_PAGES['real_workload_eval']['file_path'],
        title=APP_PAGES['real_workload_eval']['page_label'],
        icon=APP_PAGES['real_workload_eval']['page_icon'],
    )
    custom_page = st.Page(
        APP_PAGES['custom_eval']['file_path'],
        title=APP_PAGES['custom_eval']['page_label'],
        icon=APP_PAGES['custom_eval']['page_icon'],
    )
    chat_page = st.Page(
        APP_PAGES['chat_eval']['file_path'],
        title=APP_PAGES['chat_eval']['page_label'],
        icon=APP_PAGES['chat_eval']['page_icon'],
    )

    if st.session_state.prod_mode:
        pg = st.navigation(find_pages_to_show())
    else:
        pg = st.navigation([synthetic_page, real_page, custom_page, chat_page])

    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
    )

    _init()
    _initialize_session_variables()

    main()
    pg.run()
