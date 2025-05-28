# app.py

import os
import sys
import tempfile

import streamlit as st
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from bundle_jump_start.src.use_bundle_model import (
    get_expert,
    get_expert_val,
    run_bulk_routing_eval,
    run_e2e_vector_database,
    run_simple_llm_invoke,
)

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

# Load config
with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)


def save_config() -> None:
    with open(CONFIG_PATH, 'w') as yaml_file:
        yaml.dump(config, yaml_file)


def main() -> None:
    st.set_page_config(
        page_title='Bundle LLM Router AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
        layout='wide',
    )

    # Sidebar
    logo_path = os.path.join(repo_dir, 'images', 'SambaNova-dark-logo-1.png')
    st.sidebar.image(logo_path, width=200)

    page = st.sidebar.selectbox(
        'Choose a page',
        ['Config', 'Expert', 'Simple', 'E2E With Vector DB', 'Bulk Evaluation'],
    )

    st.title('Bundle LLM Router Interface')

    if page == 'Config':
        st.header('Configuration')

        st.subheader('Bundle Name Map')
        for key, value in config['bundle_name_map'].items():
            new_value = st.text_input(f'{key}', value)
            config['bundle_name_map'][key] = new_value

        st.subheader('Expert Prompt')
        config['expert_prompt'] = st.text_area('Expert Prompt', config['expert_prompt'], height=300)

        st.subheader('Supported Experts Map')
        for key, value in config['supported_experts_map'].items():
            new_value = st.text_input(f'{key}', value)
            config['supported_experts_map'][key] = new_value

        if st.button('Save Configuration'):
            save_config()
            st.success('Configuration saved successfully!')

    elif page == 'Expert':
        st.header('Expert Mode')
        query = st.text_input('Enter your query:')
        if st.button('Get Expert'):
            with st.spinner('Getting expert...'):
                expert_response = get_expert(query, use_wrapper=True)
                expert = get_expert_val(expert_response)
            st.markdown(f'**Expert:** {expert}')

    elif page == 'Simple':
        st.header('Simple Mode')
        query = st.text_input('Enter your query:')
        if st.button('Run Simple Mode'):
            with st.spinner('Processing query...'):
                expert, response = run_simple_llm_invoke(query)

            st.subheader('Chatbot Interface')
            st.markdown(f'**Expert:** {expert}')
            st.write('User: ' + query)
            st.text_area('AI:', value=response, height=200)

    elif page == 'E2E With Vector DB':
        st.header('E2E With Vector DB Mode')
        uploaded_file = st.file_uploader('Choose a file', type=['pdf', 'txt', 'md'])
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                with st.spinner('Loading file...'):
                    loader: PyPDFLoader | TextLoader | UnstructuredMarkdownLoader
                    if file_extension == 'pdf':
                        loader = PyPDFLoader(tmp_file_path)
                    elif file_extension == 'txt':
                        loader = TextLoader(tmp_file_path)
                    elif file_extension == 'md':
                        loader = UnstructuredMarkdownLoader(tmp_file_path)

                    docs = loader.load()
                st.success('File loaded successfully!')

                query = st.text_input('Enter your query:')
                if st.button('Run E2E'):
                    with st.spinner('Processing query...'):
                        expert, response = run_e2e_vector_database(query, docs)

                    st.subheader('Chatbot Response')
                    st.markdown(f'**Expert:** {expert}')
                    st.write('User: ' + query)
                    st.text_area('AI:', value=response, height=200)

            finally:
                os.unlink(tmp_file_path)

    elif page == 'Bulk Evaluation':
        st.header('Bulk Evaluation')

        # Use a form to prevent automatic rerun on every input change
        with st.form('bulk_evaluation_form'):
            dataset_path = st.text_input('Enter the path to your dataset jsonl:', key='dataset_path_input')
            num_examples = st.number_input(
                'Number of examples to evaluate (leave blank for all):',
                min_value=1,
                value=None,
                key='num_examples_input',
            )

            submit_button = st.form_submit_button('Run Bulk Evaluation')

        if submit_button:
            with st.spinner('Running bulk evaluation...'):
                results, accuracies, confusion_matrix = run_bulk_routing_eval(dataset_path, num_examples)

            st.subheader('Results')
            st.dataframe(results)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Accuracy by Category')
                fig, ax = plt.subplots()
                ax.bar(accuracies.keys(), accuracies.values())
                ax.set_ylim(0, 1)
                ax.set_ylabel('Accuracy')
                ax.set_xlabel('Category')
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with col2:
                st.subheader('Confusion Matrix')
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                st.pyplot(fig)


if __name__ == '__main__':
    main()
