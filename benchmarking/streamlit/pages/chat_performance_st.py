import os
import sys
sys.path.append('../')

import time
import streamlit as st
import pandas as pd
import math
import numpy as np
import asyncio
import yaml
from src.performance_evaluation import benchmark


st.title(":orange[SambaNova]Performance evaluation")    
st.header("Chat")    
st.markdown("With this option, users have a way to know performance metrics per response. Have a nice conversation with the LLM and know more about our performance metrics.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})