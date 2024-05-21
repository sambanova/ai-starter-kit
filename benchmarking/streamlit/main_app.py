import os
import sys
sys.path.append('../')
sys.path.append('../src')
sys.path.append('../src/llmperf')
sys.path.append('../src/ray_clients')

import streamlit as st
from st_pages import Page, show_pages, add_page_title
from dotenv import load_dotenv

load_dotenv('../.env', override=True)


st.set_page_config(
    page_title="AI Starter Kit",
    page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
)

show_pages(
    [
        Page("main_app.py", "Home", "🏠"),
        Page("pages/chat_performance_st.py", "Chat performance", "💬"),
        Page("pages/concurrent_performance_st.py", "Concurrent performance", "🤖"),
    ]
)

st.title(":orange[SambaNova]Performance evaluation")    
st.markdown("This app shows two ways to evaluate performance: \n- Concurrency \n- Chat \n\nPlease choose one of the two options in the side bar and begin interacting with them. \n\nHave fun!")
    