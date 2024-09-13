import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Callable, Generator, Optional

import streamlit as st
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from function_calling.src.function_calling import FunctionCallingLlm  # type: ignore
from function_calling.src.tools import calculator, get_time, python_repl, query_db, rag, translate  # type: ignore
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

logging.basicConfig(level=logging.INFO)

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

# tool mapping of available tools
TOOLS = {
    'get_time': get_time,
    'calculator': calculator,
    'python_repl': python_repl,
    'query_db': query_db,
    'translate': translate,
    'rag': rag,
}


def load_config():
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


config = load_config()
prod_mode = config.get('prod_mode', False)
additional_env_vars = config.get('additional_env_vars', None)


@contextmanager
def st_capture(output_func: Callable[[str], None]) -> Generator:
    """
    context manager to catch stdout and send it to an output streamlit element

    Args:
        output_func (function to write terminal output in

    Yields:
        Generator:
    """
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string: str) -> int:
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write  # type: ignore
        yield


def set_fc_llm(tools: list) -> None:
    """
    Set the FunctionCallingLlm object with the selected tools

    Args:
        tools (list): list of tools to be used
    """
    set_tools = [TOOLS[name] for name in tools]
    st.session_state.fc = FunctionCallingLlm(set_tools)


def handle_userinput(user_question: Optional[str]) -> None:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question (str): The user's question or input.
    """
    global output
    if user_question:
        with st.spinner('Processing...'):
            with st_capture(output.code):  # type: ignore
                response = st.session_state.fc.function_call_llm(
                    query=user_question, max_it=st.session_state.max_iterations, debug=True
                )

        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response)

    for ques, ans in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
    ):
        with st.chat_message('user'):
            st.write(f'{ques}')

        with st.chat_message(
            'ai',
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            st.write(f'{ans}')


def setChatInputValue(chat_input_value: str) -> None:
    js = f"""
    <script>
        function insertText(dummy_var_to_force_repeat_execution) {{
            var chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
            nativeInputValueSetter.call(chatInput, "{chat_input_value}");
            var event = new Event('input', {{ bubbles: true}});
            chatInput.dispatchEvent(event);
        }}
        insertText(3);
    </script>
    """
    st.components.v1.html(js)


def main() -> None:
    global output
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )

    initialize_env_variables(prod_mode, additional_env_vars)

    if 'fc' not in st.session_state:
        st.session_state.fc = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'tools' not in st.session_state:
        st.session_state.tools = ['get_time', 'python_repl', 'query_db']
    if 'max_iterations' not in st.session_state:
        st.session_state.max_iterations = 5
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = True

    st.title(':orange[SambaNova] Function Calling Assistant')

    with st.sidebar:
        st.title('Setup')

        #Callout to get SambaNova API Key
        st.markdown(
            "Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)"
        )


        if not are_credentials_set(additional_env_vars):
            api_key, additional_vars = env_input_fields(additional_env_vars)
            if st.button('Save Credentials'):
                message = save_credentials(api_key, additional_vars, prod_mode)
                st.success(message)
                st.rerun()

        else:
            st.success('Credentials are set')
            if st.button('Clear Credentials'):
                save_credentials('', {var: '' for var in (additional_env_vars or [])}, prod_mode)
                st.rerun()

        if are_credentials_set(additional_env_vars):
            st.markdown('**1. Select the tools for function calling.**')
            st.session_state.tools = st.multiselect(
                'Available tools',
                ['get_time', 'calculator', 'python_repl', 'query_db', 'translate', 'rag'],
                ['get_time', 'python_repl', 'query_db'],
            )
            st.markdown('**2. Set the maximum number of iterations your want the model to run**')
            st.session_state.max_iterations = st.number_input('Max iterations', value=5, max_value=20)
            st.markdown('**Note:** The response cannot completed if the max number of iterations is too low')
            if st.button('Set'):
                with st.spinner('Processing'):
                    set_fc_llm(st.session_state.tools)
                    st.toast(f'Tool calling assistant set! Go ahead and ask some questions', icon='🎉')
                st.session_state.input_disabled = False

            st.markdown('**3. Ask the model**')

            with st.expander('**Execution scratchpad**', expanded=True):
                output = st.empty()  # type: ignore

            with st.expander('**Preset Example queries**', expanded=True):
                st.markdown('DB operations')
                if st.button('Create a summary table in the db'):
                    setChatInputValue(
                        'Create and save a table in the database that will show the top 10 albums with the highest sales in 2013 and in the USA. The table fields will be the name of the album, the name of the artist, the total amount of sales, and the number of copies sold.'
                    )
                if st.button('Get information of the created summary table'):
                    setChatInputValue('Give me a summary of the 2013 top albums table')
                if st.button('Create insightful plots of the summary table'):
                    setChatInputValue(
                        'Get the information of the 2013 top albums table in the DB, when you get the data then create some meaningful plots that summarize the information, and store them in PNG format'
                    )

            with st.expander('Additional settings', expanded=False):
                st.markdown('**Interaction options**')

                st.markdown('**Reset chat**')
                st.markdown('**Note:** Resetting the chat will clear all interactions history')
                if st.button('Reset conversation'):
                    st.session_state.chat_history = []
                    st.session_state.sources_history = []
                    st.toast('Interactions reset. The next response will clear the history on the screen')

    user_question = st.chat_input('Ask something', disabled=st.session_state.input_disabled, key='TheChatInput')
    handle_userinput(user_question)


if __name__ == '__main__':
    main()
