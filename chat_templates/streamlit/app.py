"""
Interactive Streamlit app to explore model chat templates, apply them to message inputs,
send prompts via the SambaNova Completions API, and parse model outputs.
"""

import base64
import json
import logging
import os
import sys
import uuid
from datetime import datetime  # noqa
from typing import Any, Optional, cast

import streamlit as st
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from chat_templates.src.chat_template import ChatTemplateManager
from utils.events.mixpanel import MixpanelEvents
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
APP_DESCRIPTION_PATH = os.path.join(kit_dir, 'streamlit', 'app_description.yaml')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@st.cache_data
def load_config() -> Any:
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


@st.cache_data
def load_app_description() -> Any:
    with open(APP_DESCRIPTION_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


# General style
def setup_ui_style(repo_dir: str) -> None:
    st.set_page_config(
        page_title='Custom Chat Templates Kit',
        page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
        layout='wide',
        initial_sidebar_state='expanded',
    )

    # Button color theme
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #250E36;  
            color: #FFFFFF;             
        }
        div.stButton > button:hover, div.stButton > button:focus  {
            background-color: #4E22EB;  
            color: #FFFFFF;             
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Global Inter font
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">
        <style>
            html, body, [class^="css"] :not(.material-icons) {
                font-family: 'Inter', sans-serif !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Center title
    with open(os.path.join(repo_dir, 'images', 'chat_template_icon.png'), 'rb') as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: center;">
            <div style="margin-right: 20px;">
                <img src="data:image/png;base64,{encoded_img}" width="100">
            </div>
            <div>
                <style>
                    .kit-title {{
                        text-align: center;
                        color: #250E36 !important;
                        font-size: 3em;
                        font-weight: bold;
                    }}
                </style>
                <div class="kit-title">Custom Chat Templates Starter Kit</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# chat template manager initialization
def initialize_manager() -> ChatTemplateManager:
    manager = ChatTemplateManager(
        hf_token=st.session_state['HUGGINGFACE_TOKEN'],
        sambanova_api_key=st.session_state['SAMBANOVA_API_KEY'],
        sambanova_api_base=st.session_state['SAMBANOVA_API_BASE'],
    )
    return manager


# Sidebar Setup
def sidebar_setup(
    prod_mode: bool, additional_variables: dict[str, Optional[str]], app_description: dict[str, Optional[str]]
) -> None:
    with st.sidebar:
        logo_path = os.path.join(repo_dir, 'images', 'SambaNova-dark-logo-1.png')
        with open(logo_path, 'rb') as img_file:
            import base64

            encoded = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded}" style="width:60%; display: block; max-width:100%;">
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.header('Credentials')

        # Callout to get SambaNova API Key
        st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')

        # Settings

        if not are_credentials_set(additional_variables):
            st.session_state.manager = None
            api_key, additional_vars = env_input_fields(additional_variables)
            if st.button('Save Credentials', key='save_credentials_sidebar'):
                message = save_credentials(api_key, additional_vars, prod_mode)
                st.session_state.mp_events.api_key_saved()
                st.success(message)
                st.rerun()
        else:
            st.success('Credentials are set')
            if st.button('Clear Credentials', key='clear_credentials'):
                save_credentials('', '', prod_mode)  # type: ignore
                st.session_state.manager = initialize_manager()
                st.rerun()
        if are_credentials_set(additional_variables):
            if st.session_state.manager is None:
                st.session_state.manager = initialize_manager()

            st.header('Setup')

            ## chat template settings
            with st.expander('#### 1. Set  model chat template to use', expanded=True):
                chat_template_source = st.radio(
                    'Chat template source',
                    ['Hugging Face model', 'Custom chat template'],
                    help="""Select whether to **load the chat template directly from a Hugging Face model’s tokenizer** 
                    *(which includes the model’s default Jinja chat formatting)* or to define **your own custom 
                    Jinja-based template** manually. \n\n**The chosen template determines how your messages and tools 
                    are converted into the final text prompt sent to the Completions API**""",
                )
                if 'hugging' in chat_template_source.lower():
                    hf_model = st.text_input(
                        'Hugging face model name',
                        placeholder='meta-llama/Llama-3.3-70B-Instruct',
                        help="""Enter the **Full Hugging Face model identifier** whose tokenizer defines the chat 
                        template you want to load (for example: 'meta-llama/Llama-3.3-70B-Instruct').
                        The app will automatically download the tokenizer and display its built-in Jinja chat template.
                        \n\n**Find the models identifier in** https://huggingface.co/models""",
                    )
                    if st.button('Get chat template', key='get_chat_template_button'):
                        if hf_model:
                            try:
                                with st.spinner('Getting chat template'):
                                    chat_template = st.session_state.manager.get_chat_template(hf_model)
                                st.session_state.model_name = hf_model
                                st.toast(f'✅ Chat template and custom variables for {hf_model} set')
                                with st.popover(f'Chat template for **{hf_model[:15]}** ... '):
                                    st.write(f'Chat template for **{hf_model}**')
                                    st.code(chat_template, 'django', height=300)
                            except Exception as e:
                                st.error(e)
                        else:
                            st.error('You must input a Hugging face model name')
                elif 'custom' in chat_template_source.lower():
                    custom_model = st.text_input(
                        'Chat template name',
                        placeholder='Custom-Llama-3-Instruct',
                        help="""Enter a model identifier (for example: 'Custom-Llama-3.1-8B-Instruct').""",
                    )
                    custom_template = st.text_area(
                        'Custom Jinja chat template',
                        height=150,
                        value=app_description.get('sample_chat_template'),
                        placeholder=app_description.get('sample_chat_template'),
                        help=(
                            """Paste or write your **Jinja2** chat template here.
                            You can use variables like 'bos_token', 'eos_token', 'date_string', or any you define bellow.
                            'Your chat template must contain and handle `messages`, `tools` and `add_generation_prompt` 
                            variables.\n
                            **Use the prefilled one as reference**
                            """
                        ),
                    )
                    sample_variables = cast(dict[str, Any], app_description.get('sample_extra_variables', {}))
                    sample_variables = {
                        k: (eval(v.strip('{}')) if isinstance(v, str) and v.startswith('{') and v.endswith('}') else v)
                        for k, v in sample_variables.items()
                    }
                    sample_variables_str = json.dumps(sample_variables, indent=4)
                    raw_custom_variables = st.text_area(
                        'Custom variables (JSON)',
                        height=140,
                        value=sample_variables_str,
                        placeholder=sample_variables_str,
                        help="""Define variables to inject into the Jinja template in valid json format.\n
                        **Use the prefilled one as reference**
                        """,
                    )
                    try:
                        custom_variables = json.loads(raw_custom_variables)
                    except json.JSONDecodeError:
                        st.error('Invalid JSON format')
                        custom_variables = {}
                    if st.button('Set chat template', key='set_chat_template_button'):
                        if custom_model and custom_template and custom_variables:
                            try:
                                with st.spinner('Setting custom chat template'):
                                    chat_template = st.session_state.manager.set_chat_template(
                                        custom_model, custom_template, custom_variables
                                    )
                                st.session_state.model_name = custom_model
                                st.toast(f'✅ Chat template and custom variables for {custom_model} set')
                                with st.popover(f'Chat template for **{custom_model[:15]}** ... '):
                                    st.write(f'Custom chat template for **{custom_model}**')
                                    st.code(custom_template, 'django', height=300)
                                    st.write('Custom variables')
                                    st.code(json.dumps(custom_variables, indent=2), 'json', height=200)
                            except Exception as e:
                                st.error(e)
                        else:
                            st.error('You must fill all the custom chat template parameters above')

            ## Invocation settings
            with st.expander('#### 2. Define the model to call completions API', expanded=True):
                api_model_name = st.text_input(
                    'API model name',
                    'Meta-Llama-3.3-70B-Instruct',
                    help="""Enter the name of the model to use for generating completions through the **SambaNova API**,
                    this can be any model available in your SambaNova account that supports the Completions endpoint""",
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    temperature = st.slider(
                        'Temperature',
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        help=(
                            'Controls randomness in generation. '
                            'Lower values make responses more deterministic; higher values make them more diverse.'
                        ),
                    )
                with col2:
                    top_p = st.slider(
                        'Top P',
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        help=(
                            """Applies nucleus sampling. 
                            The model considers only the smallest set of tokens whose cumulative probability"""
                        ),
                    )
                with col3:
                    max_tokens = st.number_input(
                        'Max tokens',
                        min_value=1024,
                        max_value=131072,
                        value=8192,
                        step=128,
                        help='Specifies the maximum number of tokens to generate in the response.',
                    )
                if st.button('set', key='set_api_model_button'):
                    if api_model_name:
                        st.session_state.api_model_name = api_model_name
                        st.session_state.api_generation_config = {
                            'temperature': temperature,
                            'top_p': top_p,
                            'max_tokens': max_tokens,
                        }
                        st.toast('✅ API Model configs set')
                    else:
                        st.error('API model name must be set')

            ## Parsing settings
            with st.expander('#### 3. Define the model output tools parser to use', expanded=True):
                available_parsers = [
                    'JSON tools parser (llama custom)',
                    'XML tools parser (deepseek custom)',
                    'Custom defined tools parser',
                ]
                parser_source = st.radio(
                    'Parser source',
                    available_parsers,
                    help="""Select a **parser** to convert the raw text output from the Completions API 
                    into a structured **assistant message object** with tool calls or plain text content.  
                    You can pick a predefined parser or define your own custom parsing logic in Python.""",
                )
                if 'custom defined' in parser_source.lower():
                    custom_parser_name = st.text_input(
                        'Custom parser name',
                        placeholder='Custom JSON parser (llama)',
                        help='Enter a unique identifier for your parser.',
                    )
                    parser_code = st.text_area(
                        'Custom parser code (Python)',
                        height=200,
                        value=app_description.get('sample_parser_ref_code'),
                        placeholder=app_description.get('sample_parser_ref_code'),
                        help="""Write valid Python code defining an auto contained function named `parse(response: str) -> list`.
                            The function should return a list of tool call dicts or an empty list. "**Use the prefilled one as reference**""",  # noqa
                    )

                    if st.button('Add parser', key='add_parser_button'):
                        if custom_parser_name and parser_code:
                            try:
                                with st.spinner(f"Registering parser '{custom_parser_name}'..."):
                                    st.session_state.manager.add_custom_tool_parser(custom_parser_name, parser_code)
                                st.session_state.parser_name = custom_parser_name
                                st.toast(f'✅ Custom parser: `{custom_parser_name}` registered')
                                with st.popover(f'**{custom_parser_name}** `src`'):
                                    st.code(parser_code, language='python', height=300)
                            except Exception as e:
                                st.error(e)
                        else:
                            st.error('Both parser name and code must be provided.')
                else:
                    if st.button('Set parser', key='set_parser_button'):
                        if 'llama' in parser_source.lower():
                            st.session_state.parser_name = 'llama_json_parser'
                            st.toast('✅ `JSON parser (llama custom)` registered')
                            with st.popover('**JSON parser (llama custom)** `src`'):
                                st.code(
                                    app_description.get('llama_json_parser_ref_code'), language='python', height=300
                                )
                        elif 'deepseek' in parser_source.lower():
                            st.session_state.parser_name = 'deepseek_xml_parser'
                            st.toast('✅ `XML parser (deepseek custom)` registered')
                            with st.popover('**XML parser (deepseek custom)** `src`'):
                                st.code(
                                    app_description.get('deepseek_xml_parser_ref_code'), language='python', height=300
                                )


def main_interaction_area(app_description: dict[str, str]) -> None:
    # Messages and Tools Input
    st.divider()
    st.subheader('Define Messages and Tools')
    messages_col, tools_col = st.columns([6, 4])
    with messages_col:
        st.markdown('**Messages**')
        for i, msg in enumerate(st.session_state.messages):
            role_col, content_col = st.columns([1, 5])
            with role_col:
                st.session_state.messages[i]['role'] = st.selectbox(
                    f'Role {i + 1}',
                    ['system', 'user', 'assistant', 'tool'],
                    index=['system', 'user', 'assistant', 'tool'].index(msg['role']),
                    label_visibility='collapsed',
                )
            with content_col:
                if st.session_state.messages[i]['role'] == 'assistant':
                    # if message is assistant add content and tool calls text area editors
                    st.session_state.messages[i]['content'] = st.text_area(
                        f'Message {i + 1} content',
                        msg['content'],
                        placeholder='content',
                        label_visibility='collapsed',
                        height=50,
                    )
                    st.session_state.messages[i]['tool_calls'] = st.text_area(
                        f'Message {i + 1} tool_calls',
                        json.dumps(msg.get('tool_calls')) if msg.get('tool_calls') else None,
                        placeholder='tool_calls',
                        label_visibility='collapsed',
                        height=50,
                    )
                    # remove empty tool calls string or convert it to valid json ix exist
                    if not st.session_state.messages[i]['tool_calls']:
                        st.session_state.messages[i].pop('tool_calls')
                    else:
                        st.session_state.messages[i]['tool_calls'] = json.loads(
                            st.session_state.messages[i]['tool_calls']
                        )

                else:
                    # if message is other  type only add content t area editor
                    st.session_state.messages[i]['content'] = st.text_area(
                        f'Message {i + 1} content',
                        msg['content'],
                        label_visibility='collapsed',
                        height=50,
                    )
        if st.button('Add new message', key='add_new_message_button', type='primary'):
            st.session_state.messages.append({'role': 'user', 'content': ''})
            st.rerun()
        if st.button('Reset messages', key='reset_messages_button', type='secondary'):
            st.session_state.messages = app_description['sample_messages']
            st.session_state.rendered_prompt = ''
            st.session_state.raw_response = ''
            st.session_state.parsed_output = {}
            st.toast('Messages reset')
            st.rerun()
    with tools_col:
        st.markdown('**Tools definition (JSON)**')
        if not st.session_state.tools_set:  # conditional to trigger text are reload when reset tools
            st.session_state.tools = st.text_area(
                'Tools JSON',
                value=st.session_state.tools,
                height=155,
                placeholder='[{"name": "get_weather", "description": "Get the weather"}]',
                label_visibility='collapsed',
            )
            st.session_state.tools_set = True
        else:
            st.session_state.tools = st.text_area(
                'Tools JSON',
                value=st.session_state.tools,
                height=155,
                placeholder='[{"name": "get_weather", "description": "Get the weather"}]',
                label_visibility='collapsed',
            )
        if st.button('Reset tools', key='reset_tools_button', type='secondary'):
            st.session_state.tools = json.dumps(app_description['sample_tools'], indent=4)
            st.session_state.tools_set = False
            st.session_state.rendered_prompt = ''
            st.session_state.raw_response = ''
            st.session_state.parsed_output = {}
            st.toast('Tools reset')
            st.rerun()

    st.divider()
    st.subheader('Apply chat template')
    # Generation prompt checkbox + Apply template
    apply_col, rendered_col = st.columns([1, 3])
    with apply_col:
        add_prompt = st.checkbox('Add assistant generation prompt', value=True)
        apply_button = st.button(
            f'Apply {("ˋ" + st.session_state.model_name + "ˋ") if st.session_state.model_name is not None else ""} chat template',  # noqa
            key='apply_chat_template_button',
            type='primary',
        )

        if apply_button:
            if st.session_state.model_name:
                try:
                    messages = st.session_state.messages
                    if st.session_state.tools:
                        try:
                            tools = json.loads(st.session_state.tools)
                        except Exception as e:
                            raise ValueError(f'Error parsing tools, must be valid list of json tools: {e}')
                    else:
                        tools = None
                    rendered = st.session_state.manager.apply_chat_template(
                        model_name=st.session_state.model_name,
                        messages=messages,
                        tools=tools,
                        add_generation_prompt=add_prompt,
                    )
                    st.session_state.rendered_prompt = rendered
                    st.toast('✅ Chat template applied successfully.')
                except Exception as e:
                    st.error(str(e))
            else:
                st.error('You must set the model chat template first')
    with rendered_col:
        if st.session_state.rendered_prompt:
            st.write('**Rendered prompt:**')
            st.code(st.session_state.rendered_prompt, height=150)

    st.divider()
    st.subheader('Invoke model')
    invoke_col, raw_result_col = st.columns([1, 3])
    with invoke_col:
        if st.button(
            f'Invoke {("ˋ" + st.session_state.api_model_name + "ˋ") if st.session_state.api_model_name is not None else ""}',  # noqa
            key='invoke_model_btn',
            type='primary',
        ):
            if st.session_state.api_model_name:
                if st.session_state.rendered_prompt:
                    with st.spinner('Calling API'):
                        st.session_state.raw_response = st.session_state.manager.completions_invoke(
                            prompt=st.session_state.rendered_prompt,
                            model=st.session_state.api_model_name,
                            **st.session_state.api_generation_config,
                        )
                else:
                    st.error('You must apply chat template first')
            else:
                st.error('You must Define the model to call completions API first')
    with raw_result_col:
        if st.session_state.raw_response:
            st.write('**Raw model output**')
            st.code(st.session_state.raw_response, height=100)

    st.divider()
    st.subheader('Parse model response')
    parse_col, parsed_out_col = st.columns([1, 3])
    with parse_col:
        if st.button(
            f'Parse raw output {("with ˋ" + st.session_state.parser_name + "ˋ parser") if st.session_state.parser_name is not None else ""}'  # noqa
        ):
            if st.session_state.parser_name:
                if st.session_state.raw_response:
                    with st.spinner('parsing'):
                        st.session_state.parsed_output = st.session_state.manager.parse_to_message(
                            response=st.session_state.raw_response, parser_name=st.session_state.parser_name
                        )
                        st.session_state.messages.append(st.session_state.parsed_output)
                        st.rerun()
                else:
                    st.error('You must invoke the model first')
            else:
                st.error('You must define the model output tools parser to use first')
    with parsed_out_col:
        if st.session_state.parsed_output:
            st.code(st.session_state.parsed_output, height=100)


def main() -> None:
    setup_ui_style(repo_dir)

    config = load_config()
    app_description = load_app_description()
    prod_mode = config.get('prod_mode', False)
    additional_variables = {'SAMBANOVA_API_BASE': 'https://api.sambanova.ai/v1', 'HUGGINGFACE_TOKEN': None}
    initialize_env_variables(prod_mode, additional_variables)

    if 'manager' not in st.session_state:
        st.session_state.manager = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    if 'api_model_name' not in st.session_state:
        st.session_state.api_model_name = None
    if 'api_generation_config' not in st.session_state:
        st.session_state.api_generation_config = None
    if 'parser_name' not in st.session_state:
        st.session_state.parser_name = None
    if 'st_session_id' not in st.session_state:
        st.session_state.st_session_id = str(uuid.uuid4())
    if 'mp_events' not in st.session_state:
        st.session_state.mp_events = MixpanelEvents(
            os.getenv('MIXPANEL_TOKEN'),
            st_session_id=st.session_state.st_session_id,
            kit_name='Custom Chat Template Kit',
            track=prod_mode,
        )
        st.session_state.mp_events.demo_launch()
    if 'messages' not in st.session_state:
        st.session_state.messages = app_description['sample_messages']
    if 'tools' not in st.session_state:
        st.session_state.tools = json.dumps(app_description['sample_tools'], indent=4)
    if 'tools_set' not in st.session_state:
        st.session_state.tools_set = None
    if 'rendered_prompt' not in st.session_state:
        st.session_state.rendered_prompt = ''
    if 'raw_response' not in st.session_state:
        st.session_state.raw_response = ''
    if 'parsed_output' not in st.session_state:
        st.session_state.parsed_output = {}

    sidebar_setup(prod_mode, additional_variables, app_description)

    if st.session_state.manager is not None:
        main_interaction_area(app_description)

    else:
        st.divider()
        overview = app_description.get('app_overview', '')
        overview_html = overview.replace('\n', '<br>')
        st.markdown(
            f"""
                <style>
                    .kit-description {{
                        text-align:left;
                        color: #250E36 !important;
                        font-size: 1.4em;
                        display: inline;
                    }}
                </style>
                <div class="kit-description">{overview_html} </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == '__main__':
    main()
