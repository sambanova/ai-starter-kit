"""
Streamlit web application for educational content generation.

This module provides a web interface for generating educational content using
the SambaNova AI models and CrewAI framework. It handles user input, content
generation, and result display.
"""

import os
import re
import sys
import time
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator

from dotenv import load_dotenv

if TYPE_CHECKING:
    import streamlit as st
else:
    import streamlit as st

# Quick fix: Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

try:
    from src.edu_flow.config import (
        DEFAULT_PROVIDER,
        LLM_CONFIG,
        PROVIDER_CONFIGS,
    )
    from src.edu_flow.main import EduFlow
except ImportError as e:
    print(f'Error importing modules: {e}')
    sys.exit(1)

# Load environment variables
load_dotenv()


@contextmanager
def st_capture(output_func: Callable[[str], None]) -> Generator[StringIO, None, None]:
    """
    Context manager for capturing stdout and redirecting to Streamlit.

    Args:
        output_func (Callable[[str], None]): Function to handle captured output.

    Yields:
        StringIO: String buffer containing captured output.
    """
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string: str) -> int:
            ret = old_write(string)
            # Each time something is written to stdout,
            # we send it to Streamlit via `output_func`.
            output_func(stdout.getvalue() + '\n#####\n')
            return ret

        stdout.write = new_write
        yield stdout


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'final_content' not in st.session_state:
        st.session_state.final_content = None
    if 'show_api_warning' not in st.session_state:
        st.session_state.show_api_warning = False


def clean_markdown_content(content: str) -> str:
    """
    Clean and format markdown content for better rendering.

    Args:
        content (str): Raw markdown content.

    Returns:
        str: Cleaned and formatted markdown content.
    """
    content = content.rstrip()
    # Ensure a blank line after headings for readability
    content = re.sub(r'(#+ .*?)\n', r'\1\n\n', content)
    return content


def run_edu_flow(
    topic: str,
    audience_level: str,
    provider: str,
    model: str,
) -> Any:
    """
    Run the educational content generation flow.

    Args:
        topic (str): The subject to generate content about.
        audience_level (str): Target audience expertise level.
        provider (str): AI provider identifier.
        model (str): Model identifier.

    Returns:
        Any: Generated content and metadata.

    Raises:
        ValueError: If required API key is missing.
    """
    provider_config = PROVIDER_CONFIGS[provider]
    api_key = os.getenv(provider_config['api_key_env'])
    if not api_key:
        raise ValueError(f"Missing API key for {provider_config['display_name']}")

    model_name = f"{provider_config['model_prefix']}{model}"

    LLM_CONFIG.clear()
    LLM_CONFIG.update(
        {
            'model': model_name,
            'api_key': api_key,
            'base_url': provider_config['base_url'],
        }
    )

    input_vars: Dict[str, str] = {
        'audience_level': audience_level,
        'topic': topic,
    }

    from src.edu_flow.llm_config import llm

    llm.model = model_name
    llm.api_key = api_key
    llm.base_url = provider_config['base_url']

    edu_flow = EduFlow()
    edu_flow.input_variables = input_vars
    return edu_flow.kickoff()


def main() -> None:
    """Main entry point for the Streamlit application."""
    init_session_state()

    # Page config
    st.set_page_config(
        page_title='SambaNova Research Agent', page_icon='🤖', layout='wide', initial_sidebar_state='collapsed'
    )

    # Custom CSS with improved styling
    st.markdown(
        """
        <style>
        /* Base layout padding */
        .main > div { padding-top: 1rem; }
        .block-container { padding-top: 1rem; }

        /* Logo container styling */
        .logo-container {
            display: flex;
            align-items: center;
            gap: 2rem;
            padding: 1rem;
            margin-left: 0;
        }
        .logo-container img {
            max-height: 100px;
            width: auto;
            object-fit: contain;
        }

        /* Bold all text input / select labels, remove extra margin */
        .stTextInput label, .stSelectbox label {
            font-weight: bold !important;
            margin-bottom: 0.35rem !important;
        }

        /* Button styling */
        .stButton > button {
            width: 100%;
            margin-top: 1.5rem;  /* roughly aligns with input fields if they are short */
        }

        /* We use .custom-panel to unify both agent and content panels at same height */
        .custom-panel {
            height: 600px;
            overflow-y: auto !important;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* The placeholder with same .custom-panel height */
        .custom-placeholder {
            display: flex;
            height: 100%;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #666;
        }

        /* Code block styling for real-time logs */
        pre {
            height: auto;
            max-height: 530px !important;  /* just a bit less than 600 so the scrollbar is fully visible */
            overflow-y: auto !important;
            white-space: pre-wrap !important;
            background-color: #f8f9fa;
            margin: 0;
            padding: 10px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
            line-height: 1.4;
            border-radius: 4px;
        }
        /* Scrollbar styling in the code block */
        pre::-webkit-scrollbar {
            width: 8px;
        }
        pre::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        pre::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        pre::-webkit-scrollbar-thumb:hover {
            background: #666;
        }

        /* Markdown styling in final content */
        .markdown-content {
            height: 100%;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
        }
        .markdown-content h1, .markdown-content h2, .markdown-content h3 {
            color: #1f3d7a;
            margin: 1rem 0;
            font-weight: bold;
        }

        /* Adjust container heights for smaller screens */
        @media (max-width: 1200px) {
            .custom-panel {
                height: 400px !important;
            }
            pre {
                max-height: 330px !important;
            }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Logos + Title
    st.markdown(
        """
        <div class="logo-container">
            <img src="https://s3.amazonaws.com/media.ventureloop.com/images/SambaNovaSystems_paint.png" 
                 alt="SambaNova Logo">
            <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Ulg1BjUIxIdmOw63J5gF1Q.png" 
                 alt="CrewAI Logo">
        </div>
        <h2 style='text-align: center; color: #1f3d7a; margin: 0.5rem 0;'>
            SambaNova & CrewAI: Research Agent Crew
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # -------------------- FORM --------------------
    with st.form('generation_form'):
        col1, col2, col3, col4, col5 = st.columns([5, 2, 2, 2, 2])

        with col1:
            st.write('**Research Topic**')
            topic = st.text_input(
                label='Enter research topic (hidden)',
                placeholder='E.g., Quantum Computing, Machine Learning, Climate Change',
                help='Enter the main subject for content generation',
                key='compact_topic',
                label_visibility='collapsed',
            )

        with col2:
            st.write('**Target Audience**')
            audience_level = st.selectbox(
                label='Select target audience (hidden)',
                options=['beginner', 'intermediate', 'advanced'],
                index=1,
                help="Select your audience's expertise level",
                key='compact_audience',
                label_visibility='collapsed',
            )

        with col3:
            st.write('**Model**')
            model = st.selectbox(
                label='Select model (hidden)',
                options=PROVIDER_CONFIGS[DEFAULT_PROVIDER]['models'],
                help='Select the model',
                index=6,
                key='compact_model',
                label_visibility='collapsed',
            )

        with col4:
            if not os.getenv(PROVIDER_CONFIGS[DEFAULT_PROVIDER]['api_key_env']):
                st.write('**API Key**')
                api_key = st.text_input(
                    label='Enter API key (hidden)',
                    type='password',
                    help='Enter your API key',
                    key='compact_api',
                    label_visibility='collapsed',
                )
                if api_key:
                    os.environ[PROVIDER_CONFIGS[DEFAULT_PROVIDER]['api_key_env']] = api_key

        with col5:
            st.write('&nbsp;')
            generate_button = st.form_submit_button(
                label='🚀 Generate',
                type='primary',
                disabled=st.session_state.running,
                help='Click to start generating content',
            )

    # -------------------- OUTPUT COLUMNS --------------------
    output_col1, output_col2 = st.columns(2)

    # Left panel: Agent Progress
    with output_col1:
        st.markdown("<h4 style='margin: 0 0 0.5rem;'>🔄 Agent Progress</h4>", unsafe_allow_html=True)
        # We keep an empty container that we'll fill with logs or a placeholder
        execution_output = st.empty()

        if not st.session_state.running and not st.session_state.final_content:
            # Show placeholder (same .custom-panel style)
            execution_output.markdown(
                """
                <div class="custom-panel">
                    <div class="custom-placeholder">
                        Agent progress will appear here...
                        <br/><br/>
                        Click 'Generate' to begin.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Right panel: Generated Content
    with output_col2:
        st.markdown("<h4 style='margin: 0 0 0.5rem;'>📑 Generated Content</h4>", unsafe_allow_html=True)
        content_output = st.empty()

        if st.session_state.final_content:
            cleaned_content = clean_markdown_content(st.session_state.final_content)
            content_output.markdown(
                f"""
                <div class="custom-panel">
                    <div class="markdown-content">
                        {cleaned_content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Placeholder in the content box
            content_output.markdown(
                """
                <div class="custom-panel">
                    <div class="markdown-content" style="text-align: center; color: #666; padding-top: 2rem;">
                        Your generated content will appear here...
                        <br><br>
                        Configure the parameters above and click 'Generate' to begin.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------- ACTIONS ON SUBMIT --------------------
    if generate_button:
        if not topic:
            st.error('❌ Please enter a research topic!')
            return

        if not os.getenv(PROVIDER_CONFIGS[DEFAULT_PROVIDER]['api_key_env']):
            st.error('❌ Please provide an API key!')
            return

        try:
            start_time = time.time()
            st.session_state.running = True
            st.session_state.final_content = None

            # Update the right panel with spinner while generating
            content_output.markdown(
                """
                <div class="custom-panel">
                    <div class="markdown-content">
                        <div style="display: flex; flex-direction: column; 
                                   align-items: center; justify-content: center; height: 100%;">
                            <div class="stSpinner">
                                <div class="st-spinner-border" role="status"></div>
                            </div>
                            <p style="margin-top: 15px; color: #666;">
                                🔄 Generating comprehensive research content...
                                <br/>
                                This may take a few minutes. Track progress in the left panel.
                            </p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show real-time logs in the left panel
            with st.spinner('🤖 Our research agents are working on your content...'):
                with st_capture(execution_output.code):
                    run_edu_flow(topic, audience_level, DEFAULT_PROVIDER, model)

            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            time_msg = f'⚡ Generated in {minutes}m {seconds}s' if minutes > 0 else f'⚡ Generated in {seconds}s'

            st.success('✨ Content generated successfully!')
            st.markdown(f"{time_msg} using SambaNova's lightning-fast inference engine")

            # If the output file was written, show it
            output_file = f'output/{topic}_{audience_level}.md'.replace(' ', '_')
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    final_md = f.read()
                    st.session_state.final_content = final_md
                    cleaned_content = clean_markdown_content(final_md)

                    content_output.markdown(
                        f"""
                        <div class="custom-panel">
                            <div class="markdown-content">
                                {cleaned_content}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Download button
                col_dl1, col_dl2 = st.columns([3, 1])
                with col_dl2:
                    st.download_button(
                        label='📥 Download as Markdown',
                        data=final_md,
                        file_name=f'{topic}_{audience_level}.md',
                        mime='text/markdown',
                        help='Download your generated content as a markdown file',
                    )

        except Exception as e:
            st.error(f'❌ An error occurred: {str(e)}')
            st.info('🔄 Please try again or contact support if the issue persists.')
        finally:
            st.session_state.running = False


if __name__ == '__main__':
    main()
