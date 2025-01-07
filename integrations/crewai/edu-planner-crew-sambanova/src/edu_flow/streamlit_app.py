import streamlit as st
import os
import sys
from dotenv import load_dotenv
from contextlib import contextmanager
from io import StringIO
from contextlib import redirect_stdout
from typing import Callable, Generator
import time
import re

# Quick fix: Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src.edu_flow.main import EduFlow
from src.edu_flow.config import (
    LLM_CONFIG,
    DEFAULT_PROVIDER,
    PROVIDER_CONFIGS,
)

# Load environment variables
load_dotenv()


@contextmanager
def st_capture(output_func: Callable[[str], None]) -> Generator[StringIO, None, None]:
    """Context manager to catch stdout and send it to streamlit output"""
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string: str) -> int:
            ret = old_write(string)
            output_func(stdout.getvalue() + '\n#####\n')
            return ret

        stdout.write = new_write
        yield stdout


def init_session_state():
    """Initialize session state variables"""
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'final_content' not in st.session_state:
        st.session_state.final_content = None
    if 'show_api_warning' not in st.session_state:
        st.session_state.show_api_warning = False


def clean_markdown_content(content: str) -> str:
    """Clean and format markdown content for better rendering"""
    # Remove extra newlines at the end of the content
    content = content.rstrip()
    # Ensure headers have proper spacing
    content = re.sub(r'(#+ .*?)\n', r'\1\n\n', content)
    return content


def run_edu_flow(topic, audience_level, provider, model):
    """Run the education flow with given parameters"""
    provider_config = PROVIDER_CONFIGS[provider]

    api_key = os.getenv(provider_config['api_key_env'])
    if not api_key:
        raise ValueError(f"Missing API key for {provider_config['display_name']}")

    model_name = f"{provider_config['model_prefix']}{model}"

    LLM_CONFIG.clear()
    LLM_CONFIG.update({'model': model_name, 'api_key': api_key, 'base_url': provider_config['base_url']})

    input_vars = {
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


def main():
    init_session_state()

    # Page config
    st.set_page_config(
        page_title='SambaNova Research Agent', page_icon='🤖', layout='wide', initial_sidebar_state='collapsed'
    )

    # Custom CSS with improved styling
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .logo-container {
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 2rem;
        }
        .stButton > button {
            width: 100%;
            margin-top: 1rem;
        }
        .output-container {
            height: 600px;
            overflow-y: auto !important;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 8px;
            background-color: #ffffff;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        pre {
            height: 600px !important;
            overflow-y: auto !important;
            white-space: pre-wrap !important;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin: 0;
            font-family: 'Monaco', 'Menlo', monospace;
        }
        .markdown-content {
            height: 100%;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
            padding: 0 10px;
        }
        .markdown-content h1 {
            font-size: 1.8em;
            font-weight: 600;
            margin: 0.5em 0;
            padding: 0;
            text-align: left;
        }
        .markdown-content h2 {
            font-size: 1.5em;
            font-weight: 600;
            margin: 0.5em 0;
            padding: 0;
        }
        .markdown-content h3 {
            font-size: 1.3em;
            font-weight: 600;
            margin: 0.5em 0;
            padding: 0;
        }
        .markdown-content p {
            margin: 0.8em 0;
            text-align: justify;
        }
        .markdown-content ul, .markdown-content ol {
            margin: 0.8em 0;
            padding-left: 2em;
        }
        .markdown-content li {
            margin: 0.3em 0;
        }
        .input-section {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 2rem;
        }
        .stSelectbox {
            margin-bottom: 1rem;
        }
        /* Responsive adjustments */
        @media (max-width: 1200px) {
            .output-container, pre {
                height: 400px !important;
            }
        }
        /* Improved scrollbar styling */
        .markdown-content::-webkit-scrollbar {
            width: 8px;
        }
        .markdown-content::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        .markdown-content::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        .markdown-content::-webkit-scrollbar-thumb:hover {
            background: #666;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Title with improved styling
    st.markdown(
        """
        <h1 style='text-align: center; color: #1f3d7a; margin-bottom: 2rem;'>
            🤖 SambaNova x CrewAI Research Agent 
        </h1>
        <p style='text-align: center; color: #666; margin-bottom: 3rem;'>
            Powered by SambaNova's state-of-the-art LLMs
        </p>
    """,
        unsafe_allow_html=True,
    )

    # Input section with improved layout
    with st.container():
        st.markdown(
            """
            <h3 style='color: #1f3d7a; margin-bottom: 1rem;'>
                📝 Configuration
            </h3>
        """,
            unsafe_allow_html=True,
        )

        with st.expander('🎯 Input Parameters', expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                topic = st.text_input(
                    'Research Topic',
                    placeholder='E.g., Quantum Computing, Machine Learning, Climate Change',
                    help='Enter the main subject for content generation',
                )

                audience_level = st.selectbox(
                    'Target Audience',
                    options=['beginner', 'intermediate', 'advanced'],
                    index=1,  # Default to intermediate
                    help="Select your audience's expertise level",
                )

            with col2:
                provider = st.selectbox(
                    'AI Provider',
                    options=list(PROVIDER_CONFIGS.keys()),
                    index=list(PROVIDER_CONFIGS.keys()).index(DEFAULT_PROVIDER),
                    help='Select the AI provider',
                )

                model = st.selectbox(
                    'Model',
                    options=PROVIDER_CONFIGS[provider]['models'],
                    help=f"Select the {PROVIDER_CONFIGS[provider]['display_name']} model",
                    index=0,
                )

    # API Key section with improved error handling
    provider_config = PROVIDER_CONFIGS[provider]
    if not os.getenv(provider_config['api_key_env']):
        with st.expander('🔑 API Configuration', expanded=True):
            st.warning(f"⚠️ {provider_config['display_name']} API Key required!")
            api_key = st.text_input(
                f"{provider_config['display_name']} API Key",
                type='password',
                help=f"Enter your {provider_config['display_name']} API key",
            )
            if api_key:
                os.environ[provider_config['api_key_env']] = api_key

    # Generate button moved up
    generate_button = st.button(
        '🚀 Generate Research Content',
        type='primary',
        disabled=st.session_state.running,
        help='Click to start generating content',
    )

    # Output section
    st.markdown(
        """
        <h3 style='color: #1f3d7a; margin: 2rem 0 1rem;'>
            📊 Output
        </h3>
    """,
        unsafe_allow_html=True,
    )

    output_col1, output_col2 = st.columns(2)

    with output_col1:
        st.markdown('#### 🔄 Agent Progress')
        execution_container = st.container()
        with execution_container:
            execution_output = st.empty()

    with output_col2:
        st.markdown('#### 📑 Generated Content')
        content_container = st.container()
        with content_container:
            content_output = st.empty()

            if st.session_state.final_content:
                cleaned_content = clean_markdown_content(st.session_state.final_content)
                content_output.markdown(
                    f"""
                    <div class="output-container">
                        <div class="markdown-content">
                            {cleaned_content}
                        </div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                content_output.markdown(
                    """
                    <div class="output-container">
                        <div class="markdown-content" style="text-align: center; color: #666; padding-top: 2rem;">
                            Your generated content will appear here...
                            <br><br>
                            Configure the parameters above and click 'Generate Research Content' to begin.
                        </div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

    if generate_button:
        if not topic:
            st.error('❌ Please enter a research topic!')
            return

        if not os.getenv(provider_config['api_key_env']):
            st.error(f"❌ Please provide a {provider_config['display_name']} API key!")
            return

        try:
            start_time = time.time()
            st.session_state.running = True

            st.session_state.final_content = None
            content_output.markdown(
                """
                <div class="output-container">
                    <div class="markdown-content">
                        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
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

            with st.spinner('🤖 Our research agents are working on your content...'):
                with execution_container:
                    with st_capture(execution_output.code):
                        run_edu_flow(topic, audience_level, provider, model)

            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            time_msg = f'⚡ Generated in {minutes}m {seconds}s' if minutes > 0 else f'⚡ Generated in {seconds}s'

            # Show success message
            st.success('✨ Content generated successfully!')
            st.markdown(f"{time_msg} using SambaNova's lightning-fast inference engine")

            output_file = f'output/{topic}_{audience_level}.md'.replace(' ', '_')
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    content = f.read()
                    st.session_state.final_content = content
                    cleaned_content = clean_markdown_content(content)
                    content_output.markdown(
                        f"""
                        <div class="output-container">
                            <div class="markdown-content">
                                {cleaned_content}
                            </div>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                col1, col2 = st.columns([3, 1])
                with col2:
                    st.download_button(
                        label='📥 Download as Markdown',
                        data=content,
                        file_name=f'{topic}_{audience_level}.md',
                        mime='text/markdown',
                        help='Download your generated content as a markdown file',
                        key='download_button',
                    )

        except Exception as e:
            st.error(f'❌ An error occurred: {str(e)}')
            st.info('🔄 Please try again or contact support if the issue persists.')
        finally:
            st.session_state.running = False


if __name__ == '__main__':
    main()
