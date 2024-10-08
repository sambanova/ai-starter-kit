import sys
import yaml

sys.path.append('../')
sys.path.append('./src')
sys.path.append('./streamlit')

import warnings

import streamlit as st
from st_pages import Page, show_pages, hide_pages
from benchmarking.streamlit.streamlit_utils import APP_PAGES

from utils.visual.env_utils import initialize_env_variables, are_credentials_set, save_credentials

warnings.filterwarnings('ignore')

CONFIG_PATH = "./config.yaml"


with open(CONFIG_PATH) as file:
    st.session_state.config = yaml.safe_load(file)
    st.session_state.prod_mode = st.session_state.config['prod_mode']
    st.session_state.pages_to_show = st.session_state.config['pages_to_show']

def env_input_fields(mode, additional_env_vars=None):
    if additional_env_vars is None:
        additional_env_vars = []

    additional_vars = {}

    if mode == "SambaNova Cloud":
        api_key = st.text_input("SAMBANOVA CLOUD API KEY",
                                value=st.session_state.get(
                                    "SAMBANOVA_API_KEY", ""),
                                type="password")
    else:  # SambaStudio
        api_key = st.text_input("SAMBASTUDIO API KEY",
                                value=st.session_state.get(
                                    "SAMBASTUDIO_API_KEY", ""),
                                type="password")
        for var in additional_env_vars:
            if var == "SAMBASTUDIO_BASE_URI":
                additional_vars[var] = st.text_input(
                    f"{var}",
                    value="api/v2/predict/generic",
                    type="password")
            elif var != "SAMBASTUDIO_API_KEY":
                additional_vars[var] = st.text_input(
                    f"{var}",
                    value=st.session_state.get(var, ""),
                    type="password")

    return api_key, additional_vars



def main():
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )
    
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = None

    show_pages(
        [
            Page(APP_PAGES['setup']['file_path'], APP_PAGES['setup']['page_label']),
            Page(APP_PAGES['synthetic_eval']['file_path'], APP_PAGES['synthetic_eval']['page_label']),
            Page(APP_PAGES['custom_eval']['file_path'], APP_PAGES['custom_eval']['page_label']),
            Page(APP_PAGES['chat_eval']['file_path'], APP_PAGES['chat_eval']['page_label']),
        ]
    )
    
    prod_mode = st.session_state.prod_mode
    
    if prod_mode:
        if not st.session_state.setup_complete:
            hide_pages([APP_PAGES['synthetic_eval']['page_label'], APP_PAGES['custom_eval']['page_label'], APP_PAGES['chat_eval']['page_label']])
            
            st.title("Setup")

            #Callout to get SambaNova API Key
            st.markdown(
                "Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)"
            )

            # Mode selection
            st.session_state.mode = st.radio("Select Mode",
                                            ["SambaNova Cloud", "SambaStudio"])

            if st.session_state.mode == "SambaNova Cloud":
                additional_env_vars = []
                st.session_state.llm_api = "sncloud"
            else:  # SambaStudio
                additional_env_vars = [
                    "SAMBASTUDIO_BASE_URL", "SAMBASTUDIO_BASE_URI", "SAMBASTUDIO_PROJECT_ID",
                    "SAMBASTUDIO_ENDPOINT_ID", "SAMBASTUDIO_API_KEY"
                ]
                st.session_state.llm_api = "sambastudio"

            initialize_env_variables(prod_mode, additional_env_vars)

            if not are_credentials_set(additional_env_vars):
                api_key, additional_vars = env_input_fields(
                    st.session_state.mode, additional_env_vars)
                if st.button("Save Credentials"):
                    if st.session_state.mode == "SambaNova Cloud":
                        message = save_credentials(api_key, None, prod_mode)
                    else:  # SambaStudio
                        additional_vars["SAMBASTUDIO_API_KEY"] = api_key
                        message = save_credentials(api_key, additional_vars,
                                                prod_mode)
                    st.success(message)
                    st.session_state.setup_complete = True
                    st.rerun()
            else:
                st.success("Credentials are set")
                if st.button("Clear Credentials"):
                    if st.session_state.mode == "SambaNova Cloud":
                        save_credentials("", None, prod_mode)
                    else:
                        save_credentials("",
                                        {var: ""
                                        for var in additional_env_vars},
                                        prod_mode)
                    st.session_state.setup_complete = False
                    st.rerun()
                if st.button("Continue to App"):
                    st.session_state.setup_complete = True
                    st.rerun()

        else:
            st.switch_page("pages/synthetic_performance_eval_st.py")
    else:
        st.switch_page("pages/synthetic_performance_eval_st.py")

if __name__ == '__main__':
    main()
