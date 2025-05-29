import logging
import os  # for using env variables
import sys  # for appending more paths

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from typing import Any

import streamlit as st  # for gui elements, secrets management
from dotenv import load_dotenv  # for loading env variables

from prompt_engineering.src.llm_management import LLMManager

# load env variables
load_dotenv(os.path.join(repo_dir, '.env'))

logging.basicConfig(level=logging.INFO)
logging.info('URL: https://localhost:8501')


@st.cache_data
def call_api(llm_manager: LLMManager, prompt: str, llm_expert: str) -> Any:
    """Calls the API endpoint. Uses an input prompt and returns a completion of it.

    Args:
        llm_manager (LLMManager): llm manager object
        prompt (str): prompt text
        selected_model (str): selected model from Streamlit

    Returns:
        Completion of the input prompt
    """

    # Setting llm
    llm = llm_manager.set_llm(model_expert=llm_expert)

    # Get completion from llm
    completion_text = llm.invoke(prompt)
    return completion_text

def main() -> None:
    # Set up title
    st.set_page_config(
        page_title='Prompt Engineering - SambaNova Starter Kits',
        layout='centered',
        initial_sidebar_state='auto',
        menu_items={'Get help': 'https://github.com/sambanova/ai-starter-kit/issues/new'},
    )
    
    # set buttons style
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: #250E36;  /* Button background */
            color: #FFFFFF;             /* Button text color */
        }
        div.stButton > button:hover, div.stButton > button:focus  {
            background-color: #4E22EB;  /* Button background */
            color: #FFFFFF;             /* Button text color */
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Load Inter font from Google Fonts and apply globally
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">

        <style>
            /* Apply Exile font to all elements on the page */
            * {
                font-family: 'Inter', sans-serif !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # add title and icon
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(os.path.join(repo_dir, 'images', 'SambaNova-dark-logo-1.png'))
    st.markdown("""
        <style>
            .kit-title {
                text-align: center;
                color: #250E36 !important;
                font-size: 3.0em;
                font-weight: bold;
                margin-bottom: 0.5em;
            }
        </style>
        <div class="kit-title">Prompt Engineering</div>
    """, unsafe_allow_html=True)

    # Instantiate LLMManager class
    llm_manager = LLMManager()

    llm_info = llm_manager.llm_info
    model_info = llm_manager.model_info
    prompt_use_cases = llm_manager.prompt_use_cases

    # Set up model names
    model_names = [key for key, _ in model_info.items()]
    model_name_candidates = [
        model_name
        for model_name in model_names
        if model_name.lower() in llm_info['select_expert'].lower().replace('-', '')
    ]

    if len(model_name_candidates) > 0:
        llm_expert = llm_info['select_expert']
        selected_model_for_prompt = model_name_candidates[0]
    else:
        raise Exception("The llm expert specified doesn't match with the list of models provided in config.")

    # Set up model selection drop box
    col1, col2 = st.columns([1, 1])
    with col1:
        st.text_input('Model display', llm_expert, disabled=True)
        st.write(
            f""":violet[**Architecture:**]
            {model_info[selected_model_for_prompt]['Model Architecture']}  \n:violet[**Prompting Tips:**]
            {model_info[selected_model_for_prompt]['Architecture Prompting Implications']}"""
        )

    # Set up use case drop box
    with col2:
        selected_prompt_use_case = st.selectbox(
            'Use Case for Sample Prompt',
            prompt_use_cases,
            help="""
            \n:violet[**General Assistant:**] Provides comprehensive assistance on a wide range of topics, including
            answering questions, offering explanations, and giving advice. It's ideal for general knowledge, trivia,
            educational support, and everyday inquiries.
            \n:violet[**Document Search:**] Specializes in locating and briefing relevant information from large documents
            or databases. Useful for research, data analysis, and extracting key points from extensive text sources.
            \n:violet[**Product Selection:**] Assists in choosing products by comparing features, prices, and reviews.
            Ideal for shopping decisions, product comparisons, and understanding the pros and cons of different items.
            \n:violet[**Code Generation:**] Helps in writing, debugging, and explaining code. Useful for software 
            development, learning programming languages, and automating simple tasks through scripting.
            \n:violet[**Summarization:**] Outputs a summary based on a given context. Essential for condensing large
            volumes of text into concise representations, aiding efficient information retrieval and comprehension.
            \n:violet[**Question & Answering:**] Answers questions regarding different topics given in a previous context.
            Crucial for enabling users to directly obtain relevant information from textual data, facilitating efficient
            access to knowledge and aiding decision-making processes.
            \n:violet[**Query decomposition:**] Aids on simplyfying complex queries into small and precise sub-questions.
            Vital for breaking down complex queries into more manageable sub-tasks, facilitating more effective
            information retrieval and generation processes.
            """,
        )
        st.write(f":violet[**Meta Tag Format:**]  \n {model_info[selected_model_for_prompt]['Meta Tag Format']}")

    # Set up prompting area. Show prompt depending on the model selected and use case
    assert isinstance(
        selected_prompt_use_case, str
    ), f'`selected_prompt_use_case` must be a string. Got type {type(selected_prompt_use_case)}.'
    prompt_template = llm_manager.get_prompt_template(selected_model_for_prompt, selected_prompt_use_case)
    prompt = st.text_area(
        'Prompt',
        prompt_template,
        height=210,
    )

    # Process prompt and show the completion
    if st.button('Send'):
        response_content = ''
        # Call endpoint and show the response content
        if llm_info['api'] == 'sambastudio':
            response_content = call_api(llm_manager, prompt, llm_expert)
            st.write(response_content)
        elif llm_info['api'] == 'sncloud':
            response_content = call_api(llm_manager, prompt, llm_expert)
            st.write(response_content)
        else:
            st.error('Please select a valid API in your config file "sncloud" or "sambastudio" ')


if __name__ == '__main__':
    # run following method if you want to know how prompt yaml files were created.
    # create_prompt_yamls()

    main()
