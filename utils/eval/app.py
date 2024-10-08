import logging
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import pandas as pd
import streamlit as st
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def load_default_config(config_path: str = 'config.yaml') -> Any:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str = 'config.yaml') -> None:
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def update_config(config: Dict[str, Any], key: str, value: Any) -> None:
    keys = key.split('.')
    if len(keys) == 1:
        if isinstance(config, list):
            index = int(keys[0])
            config[index] = None if value == '' else value
        else:
            config[keys[0]] = None if value == '' else value
    else:
        if isinstance(config[keys[0]], list):
            index = int(keys[1])
            update_config(config[keys[0]][index], '.'.join(keys[2:]), value)
        else:
            update_config(config[keys[0]], '.'.join(keys[1:]), value)


def display_config(config: Dict[str, Any]) -> None:
    st.sidebar.subheader('Evaluation Dataset')
    st.sidebar.text_input('Path', value=config['eval_dataset']['path'], key='eval_dataset.path')
    st.sidebar.text_input(
        'Question Column',
        value=config['eval_dataset']['question_col'],
        key='eval_dataset.question_col',
    )
    st.sidebar.text_input(
        'Ground Truth Column',
        value=config['eval_dataset']['ground_truth_col'],
        key='eval_dataset.ground_truth_col',
    )
    st.sidebar.text_input(
        'Answer Column',
        value=config['eval_dataset']['answer_col'],
        key='eval_dataset.answer_col',
    )
    st.sidebar.text_input(
        'Context Column',
        value=config['eval_dataset']['context_col'],
        key='eval_dataset.context_col',
    )

    st.sidebar.subheader('LLMs')
    for i, llm in enumerate(config['llms']):
        st.sidebar.text_input(f'LLM {i+1} Name', value=llm['name'], key=f'llms.{i}.name')
        st.sidebar.text_input(
            f'LLM {i+1} Select Expert',
            value=llm['model_kwargs']['select_expert'],
            key=f'llms.{i}.model_kwargs.select_expert',
        )
        st.sidebar.checkbox(
            f'LLM {i+1} Process Prompt',
            value=llm['model_kwargs']['process_prompt'],
            key=f'llms.{i}.model_kwargs.process_prompt',
        )
        st.sidebar.number_input(
            f'LLM {i+1} Max Tokens to Generate',
            value=llm['model_kwargs']['max_tokens_to_generate'],
            key=f'llms.{i}.model_kwargs.max_tokens_to_generate',
        )

    st.sidebar.subheader('Evaluation LLMs')
    for i, llm in enumerate(config['eval_llms']):
        st.sidebar.text_input(f'Eval LLM {i+1} Name', value=llm['name'], key=f'eval_llms.{i}.name')
        st.sidebar.text_input(
            f'Eval LLM {i+1} Select Expert',
            value=llm['model_kwargs']['select_expert'],
            key=f'eval_llms.{i}.model_kwargs.select_expert',
        )
        st.sidebar.checkbox(
            f'Eval LLM {i+1} Process Prompt',
            value=llm['model_kwargs']['process_prompt'],
            key=f'eval_llms.{i}.model_kwargs.process_prompt',
        )
        st.sidebar.number_input(
            f'Eval LLM {i+1} Max Tokens to Generate',
            value=llm['model_kwargs']['max_tokens_to_generate'],
            key=f'eval_llms.{i}.model_kwargs.max_tokens_to_generate',
        )

    st.sidebar.subheader('Vector DB')
    st.sidebar.text_input('Location', value=config['vector_db']['location'], key='vector_db.location')

    st.sidebar.subheader('Embeddings')
    st.sidebar.text_input(
        'Model Name',
        value=config['embeddings']['model_name'],
        key='embeddings.model_name',
    )

    st.sidebar.subheader('Evaluation')
    st.sidebar.number_input(
        'Number of Samples',
        value=config['evaluation']['num_samples'],
        key='evaluation.num_samples',
    )
    st.sidebar.checkbox(
        'Log to Weights & Biases',
        value=config['evaluation']['log_wandb'],
        key='evaluation.log_wandb',
    )
    st.sidebar.text_input(
        'Project Name',
        value=config['evaluation']['project_name'],
        key='evaluation.project_name',
    )
    st.sidebar.text_input(
        'Evaluation Name',
        value=config['evaluation']['eval_name'],
        key='evaluation.eval_name',
    )
    st.sidebar.text_input(
        'Evaluation Methodology',
        value=config['evaluation']['methodology'],
        key='evaluation.methodology',
    )
    st.sidebar.checkbox(
        'Save Evaluation Table CSV',
        value=config['evaluation']['save_eval_table_csv'],
        key='evaluation.save_eval_table_csv',
    )

    st.sidebar.subheader('Pipeline')
    st.sidebar.text_input('Pipeline Class', value=config['pipeline']['class'], key='pipeline.class')
    st.sidebar.text_input(
        'Vector DB Location',
        value=config['pipeline']['kwargs']['vector_db_location'],
        key='pipeline.kwargs.vector_db_location',
    )


def main() -> None:
    st.set_page_config(page_title='RAG Evaluation', layout='wide')

    st.title('RAG Evaluation')

    # Configuration
    st.sidebar.header('Configuration')
    config_path = 'config.yaml'
    config = load_default_config(config_path)
    display_config(config)

    if st.sidebar.button('Save Configuration'):
        for key, value in st.session_state.items():
            update_config(config, key, value)
        save_config(config, config_path)
        st.sidebar.success('Configuration saved!')

    # Evaluation CSV
    st.header('Evaluation CSV')
    eval_csv_file = st.file_uploader('Upload evaluation CSV file', type=['csv'])

    if eval_csv_file is not None:
        eval_df = pd.read_csv(eval_csv_file)
        with st.expander('View Evaluation CSV'):
            st.dataframe(eval_df, height=200)

    # Generation Pipeline
    st.header('Generation Pipeline')
    use_generation = st.checkbox('Use Generation Pipeline')

    # Run Evaluation
    if st.button('Run Evaluation'):
        if eval_csv_file is None:
            st.error('Please upload an evaluation CSV file.')
        else:
            # Update the configuration based on user input
            for key, value in st.session_state.items():
                update_config(config, key, value)

            # Save the updated config to a temporary file
            with NamedTemporaryFile(delete=False, mode='w', suffix='.yaml') as tmp_config_file:
                save_config(config, tmp_config_file.name)
                tmp_config_path = tmp_config_file.name

            eval_csv_path = eval_csv_file.name

            # Save the uploaded CSV to a temporary file
            with NamedTemporaryFile(delete=False, mode='w', suffix='.csv') as tmp_eval_file:
                eval_df.to_csv(tmp_eval_file.name, index=False)
                tmp_eval_csv_path = tmp_eval_file.name

            # Construct the command
            cmd = [
                sys.executable,
                'evaluate.py',
                '--config',
                tmp_config_path,
                '--eval_csv',
                tmp_eval_csv_path,
            ]
            if use_generation:
                cmd.append('--generation')

            # Run the evaluation and display stdout and stderr in real-time
            result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = result.communicate()

            st.text(stdout.decode('utf-8'))
            st.text(stderr.decode('utf-8'))

            # Clean up temporary files
            os.remove(tmp_config_path)
            os.remove(tmp_eval_csv_path)


if __name__ == '__main__':
    main()
