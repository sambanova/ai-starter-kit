import base64
import logging
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from threading import Thread
from typing import Any, Callable, Generator, List, Optional

import schedule
import streamlit as st
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from function_calling.src.function_calling import FunctionCallingLlm
from utils.events.mixpanel import MixpanelEvents
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

logging.basicConfig(level=logging.INFO)

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
PRESET_QUERIES_PATH = os.path.join(kit_dir, 'prompts', 'streamlit_preset_queries.yaml')
APP_DESCRIPTION_PATH = os.path.join(kit_dir, 'streamlit', 'app_description.yaml')

EXIT_TIME_DELTA = 30


def load_config() -> Any:
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def load_preset_queries() -> Any:
    with open(PRESET_QUERIES_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def load_app_description() -> Any:
    with open(APP_DESCRIPTION_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


config = load_config()
prod_mode = config.get('prod_mode', False)
llm_type = 'SambaStudio' if config.get('llm', {}).get('api') == 'sambastudio' else 'SambaNova Cloud'
st_tools = config.get('st_tools', {})
st_preset_queries = load_preset_queries()
st_description = load_app_description()

db_path = config['tools']['query_db']['db'].get('path')
additional_env_vars = config.get('additional_env_vars', None)


@contextmanager
def st_capture(output_func: Callable[[str], None]) -> Generator[StringIO, None, None]:
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
        yield stdout


def delete_temp_dir(temp_dir: str) -> None:
    """
    Delete the temporary directory and its contents.

    Args:
        temp_dir (str): The path of the temporary directory.
    """

    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logging.info(f'Temporary directory {temp_dir} deleted.')
        except:
            logging.info(f'Could not delete temporary directory {temp_dir}.')


def schedule_temp_dir_deletion(temp_dir: str, delay_minutes: int) -> None:
    """
    Schedule the deletion of the temporary directory after a delay.

    Args:
        temp_dir (str): The path of the temporary directory.
        delay_minutes (int): The delay in minutes after which the temporary directory should be deleted.
    """

    schedule.every(delay_minutes).minutes.do(delete_temp_dir, temp_dir).tag(temp_dir)

    def run_scheduler() -> None:
        while schedule.get_jobs(temp_dir):
            schedule.run_pending()
            time.sleep(1)

    # Run scheduler in a separate thread to be non-blocking
    Thread(target=run_scheduler, daemon=True).start()


def create_temp_db(out_path: str) -> None:
    """
    Create a temporary database at the specified path.

    Args:
        out_path (str): The path where the temporary database will be created.
    """
    logging.info(f'creating temp db in {out_path}')
    directory = os.path.dirname(out_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    shutil.copy2(os.path.join(kit_dir, db_path), out_path)


def set_fc_llm(tools: List[Any]) -> None:
    """
    Set the FunctionCallingLlm object with the selected tools

    Args:
        tools (list): list of tools to be used
    """
    if 'query_db' in tools:
        if prod_mode:
            create_temp_db(st.session_state.session_temp_db)
            schedule_temp_dir_deletion(os.path.dirname(st.session_state.session_temp_db), EXIT_TIME_DELTA)
            st.toast("""your session will be active for the next 30 minutes, after this time tmp db will be deleted""")

    st.session_state.fc = FunctionCallingLlm(
        tools,
        sambanova_api_key=st.session_state.get('SAMBANOVA_API_KEY'),
        session_temp_db=st.session_state.session_temp_db,
    )


def handle_userinput(user_question: Optional[str]) -> None:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question (str): The user's question or input.
    """

    # append user question and response to history
    if user_question:
        st.session_state.chat_history.append(user_question)
        # crete an execution scratchpad output

    # show overview message when chat history is empty
    if len(st.session_state.chat_history) == 0:
        with st.chat_message(
            'ai',
            avatar=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
        ):
            st.write(st_description.get('app_overview'))

    # show history in chat view
    for i in range(len(st.session_state.chat_history)):
        if i % 2 == 0:
            # show user input (even messages)
            with st.chat_message('user'):
                st.write(f'{st.session_state.chat_history[i]}')
            # show execution scratchpad if already stored
            if i // 2 < len(st.session_state.execution_scratchpad_history):
                with st.expander('**Execution Scratchpad**', expanded=False):
                    st.code(st.session_state.execution_scratchpad_history[i // 2])
        else:
            # show AI outputs (odd messages)
            with st.chat_message(
                'ai',
                avatar=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
            ):
                formatted_ans = st.session_state.chat_history[i].replace('$', '\$')
                st.write(f'{formatted_ans}')

    # generate response
    if user_question:
        with st.spinner('Processing...'):
            with st.expander('**Calling Tools to generate a response**', expanded=True):
                # create a new empty scratchpad expander
                execution_scratchpad_output = st.empty()
            # capture logs and show them in the current expanded execution scratchpad
            with st_capture(execution_scratchpad_output.code) as stdout:  # type: ignore
                response = st.session_state.fc.function_call_llm(
                    query=user_question, max_it=st.session_state.max_iterations
                )
                # Add scratchpad output to the scratchpad history
                st.session_state.execution_scratchpad_history.append(stdout.getvalue())
                # append response to history
                st.session_state.chat_history.append(response)
                # rerun the app to reflect the new chat history
                st.rerun()


def setChatInputValue(chat_input_value: str) -> None:
    js = f"""
    <script>
        function insertText(dummy_var_to_force_repeat_execution) {{
            var chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype,
                "value"
            ).set;
            nativeInputValueSetter.call(chatInput, "{chat_input_value}");
            var event = new Event('input', {{ bubbles: true}});
            chatInput.dispatchEvent(event);
        }}
        insertText(3);
    </script>
    """
    st.components.v1.html(js)


def set_fc_session_state_variables() -> None:
    if 'tools' not in st.session_state:
        st.session_state.tools = [k for k, v in st_tools.items() if v['default'] and v['enabled']]
    if 'max_iterations' not in st.session_state:
        st.session_state.max_iterations = 5
    if 'session_temp_db' not in st.session_state:
        if prod_mode:
            st.session_state.session_temp_db = os.path.join(
                kit_dir, 'data', 'tmp_' + st.session_state.st_session_id, 'temp_db.db'
            )
        else:
            st.session_state.session_temp_db = None
    if 'fc' not in st.session_state:
        st.session_state.fc = None
        set_fc_llm(st.session_state.tools)


def main() -> None:
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
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
    col1, col2, col3 = st.columns([4, 1, 4])
    with col2:
        st.image(os.path.join(repo_dir, 'images', 'fc_icon.png'))
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
        <div class="kit-title">Function Calling Assistant</div>
    """, unsafe_allow_html=True)

    initialize_env_variables(prod_mode, additional_env_vars)

    if 'st_session_id' not in st.session_state:
        st.session_state.st_session_id = str(uuid.uuid4())
    if 'mp_events' not in st.session_state:
        st.session_state.mp_events = MixpanelEvents(
            os.getenv('MIXPANEL_TOKEN'),
            st_session_id=st.session_state.st_session_id,
            kit_name='function_calling',
            track=prod_mode,
        )
        st.session_state.mp_events.demo_launch()
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'execution_scratchpad_history' not in st.session_state:
        st.session_state.execution_scratchpad_history = []

    with st.sidebar:
        
        # Inject HTML to display the logo in the sidebar at 70% width
        logo_path = os.path.join(repo_dir, 'images', 'SambaNova-dark-logo-1.png')
        with open(logo_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        st.sidebar.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded}" style="width:60%; display: block; max-width:100%;">
            </div>
        """, unsafe_allow_html=True)
        
        st.title('Setup')

        # Callout to get SambaNova API Key
        st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')

        if not are_credentials_set(additional_env_vars):
            api_key, additional_vars = env_input_fields(additional_env_vars, mode=llm_type)
            if st.button('Save Credentials'):
                message = save_credentials(api_key, additional_vars, prod_mode)
                st.success(message)
                st.session_state.mp_events.api_key_saved()
                st.rerun()

        else:
            st.success('Credentials are set')
            if st.button('Clear Credentials'):
                save_credentials('', {var: '' for var in (additional_env_vars or [])}, prod_mode)
                st.rerun()

        if are_credentials_set(additional_env_vars):
            set_fc_session_state_variables()
            st.markdown('**1. Select the tools for function calling.**')
            st.session_state.tools = st.multiselect(
                'Available tools',
                [k for k, v in st_tools.items() if v['enabled']],
                [k for k, v in st_tools.items() if v['default'] and v['enabled']],
            )

            if st.button('Set'):
                with st.spinner('Processing'):
                    set_fc_llm(st.session_state.tools)
                    st.toast(f'Tool calling assistant set! Go ahead and ask some questions', icon='🎉')
                st.session_state.input_disabled = False

            st.markdown('**2. Set the maximum number of iterations your want the model to run**')
            st.session_state.max_iterations = st.number_input('Max iterations', value=5, max_value=20)
            st.markdown('**Note:** The response cannot completed if the max number of iterations is too low')

            st.markdown('**3. Ask the model**')

            with st.expander('**Preset Example queries**', expanded=True):
                st.markdown('DB operations')
                for button_title, query in st_preset_queries.items():
                    if st.button(button_title):
                        setChatInputValue(query.strip())

            with st.expander('Additional settings', expanded=False):
                st.markdown('**Interaction options**')

                st.markdown('**Reset messages**')
                st.markdown('**Note:** Resetting the chat will clear all interactions history')
                if st.button('Reset messages history'):
                    st.session_state.chat_history = []
                    st.session_state.execution_scratchpad_history = []
                    st.toast('Interactions reset. The next response will clear the history on the screen')

    user_question = st.chat_input('Ask something', disabled=st.session_state.input_disabled, key='TheChatInput')
    if user_question is not None:
        st.session_state.mp_events.input_submitted('chat_input')
    handle_userinput(user_question)


if __name__ == '__main__':
    main()
