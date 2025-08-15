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
from st_pages import Page, show_pages

from benchmarking.streamlit.streamlit_utils import APP_PAGES, render_logo, set_font
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


def _initialize_session_variables() -> None:
    if 'prod_mode' not in st.session_state:
        st.session_state.prod_mode = None
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
    render_logo()
    set_font()
    show_pages(
        [
            Page(APP_PAGES['main']['file_path'], APP_PAGES['main']['page_label']),
            Page(APP_PAGES['synthetic_eval']['file_path'], APP_PAGES['synthetic_eval']['page_label']),
            Page(APP_PAGES['real_workload_eval']['file_path'], APP_PAGES['real_workload_eval']['page_label']),
            Page(APP_PAGES['custom_eval']['file_path'], APP_PAGES['custom_eval']['page_label']),
            Page(APP_PAGES['chat_eval']['file_path'], APP_PAGES['chat_eval']['page_label']),
        ]
    )

    st.switch_page('pages/synthetic_performance_eval_st.py')


if __name__ == '__main__':
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
    )

    _init()
    _initialize_session_variables()
    main()