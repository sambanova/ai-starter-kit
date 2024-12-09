from typing import List, Optional

import streamlit as st


def tabs(tabs: List[str] = [], default_active_tab: int = 0) -> Optional[str]:
    if not tabs:
        return None
    active_tab = st.radio('', tabs, index=default_active_tab)
    child = tabs.index(active_tab) + 1
    st.markdown(
        """  
            <style type="text/css">
            div[role=radiogroup] {
                border-bottom: 2px solid rgba(49, 51, 63, 0.1);
            }
            div[role=radiogroup] > label > div:first-of-type {
               display: none
            }
            div[role=radiogroup] {
                flex-direction: unset
            }
            div[role=radiogroup] label {
                padding-bottom: 0.5em;
                border-radius: 0;
                position: relative;
                top: 3px;
            }
            div[role=radiogroup] label .st-fc {
                padding-left: 0;
            }
            div[role=radiogroup] label:hover p {
                color: red;
            }
            div[role=radiogroup] label:nth-child("""
        + str(child)
        + """) {    
                border-bottom: 2px solid rgb(255, 75, 75);
            }     
            div[role=radiogroup] label:nth-child("""
        + str(child)
        + """) p {    
                color: rgb(255, 75, 75);
                padding-right: 0;
            }            
            </style>
        """,
        unsafe_allow_html=True,
    )
    return active_tab
