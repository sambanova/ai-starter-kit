import sys

import yaml

sys.path.append('../')
sys.path.append('./src')
sys.path.append('./streamlit')

import warnings

import streamlit as st
from dotenv import load_dotenv
from st_pages import Page, show_pages

from benchmarking.streamlit.streamlit_utils import APP_PAGES, shared_session_variables_initialization
from benchmarking.utils import CONFIG_PATH

warnings.filterwarnings('ignore')


@st.cache_data
def _init() -> None:
    load_dotenv('../.env', override=True)


with open(CONFIG_PATH) as file:
    st.session_state.config = yaml.safe_load(file)
    st.session_state.prod_mode = st.session_state.config['prod_mode']
    st.session_state.pages_to_show = st.session_state.config['pages_to_show']

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

    st.switch_page('pages/synthetic_performance_eval_st.py')


if __name__ == '__main__':
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )

    _init()
    show_pages(
        [
            Page(APP_PAGES['synthetic_eval']['file_path'], APP_PAGES['synthetic_eval']['page_label']),
            Page(APP_PAGES['real_workload_eval']['file_path'], APP_PAGES['real_workload_eval']['page_label']),
            Page(APP_PAGES['custom_eval']['file_path'], APP_PAGES['custom_eval']['page_label']),
            Page(APP_PAGES['chat_eval']['file_path'], APP_PAGES['chat_eval']['page_label']),
        ]
    )

    shared_session_variables_initialization()
    st.switch_page('pages/synthetic_performance_eval_st.py')
