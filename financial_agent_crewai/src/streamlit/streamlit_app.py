"""
Streamlit web application for financial content generation.

This module provides a web interface for generating financial content using
the SambaNova AI models and CrewAI framework. It handles user input, content
generation, and result display.
"""

import os
import re
import sys
import time
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Any, Generator

import streamlit
from dotenv import load_dotenv

from financial_agent_crewai.src.main import FinancialFlow

# Quick fix: Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)


# Load environment variables
load_dotenv()


def main() -> None:
    """Main entry point for the Streamlit application."""
    init_session_state()

    # Page config
    streamlit.set_page_config(
        page_title='SambaNova Financial Agent', page_icon='💸', layout='wide', initial_sidebar_state='collapsed'
    )

    # Custom CSS with improved styling
    streamlit.markdown(
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
            color: #ee7624;
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
            background: #ee7624;
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
            color: #ee7624;
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
    streamlit.markdown(
        """
        <div class="logo-container">
            <img src="https://s3.amazonaws.com/media.ventureloop.com/images/SambaNovaSystems_paint.png" 
                 alt="SambaNova Logo">
            <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Ulg1BjUIxIdmOw63J5gF1Q.png" 
                 alt="CrewAI Logo">
        </div>
        <h2 style='text-align: center; color: #ee7624; margin: 0.5rem 0;'>
            SambaNova Financial Agent
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # -------------------- FORM --------------------
    with streamlit.form('generation_form'):
        col1, col2, col3 = streamlit.columns([7, 3, 4])

        with col1:
            streamlit.write('**User query**')
            query = streamlit.text_input(
                label='Enter research topic (hidden)',
                placeholder='E.g., What is the research and development trend for Google in 2024?',
                help='Enter the main subject for content generation',
                key='compact_topic',
                label_visibility='collapsed',
            )

            streamlit.write('&nbsp;')
            generate_button = streamlit.form_submit_button(
                label='🚀 Generate',
                type='secondary',
                disabled=streamlit.session_state.running,
                help='Click to start generating content',
            )

        with col2:
            streamlit.write('**Sources**')
            # Checkboxes for user options
            generic_research_option = streamlit.checkbox('Generic Google Search', value=True)
            sec_edgar_option = streamlit.checkbox('SEC Edgar Filings', value=True)
            yfinance_news_option = streamlit.checkbox('Yahoo Finance News', value=True)
            yfinance_stocks_option = streamlit.checkbox('Yahoo Finance Stocks', value=True)

        with col3:
            # if not os.getenv('SAMBANOVA_API_KEY'):
            streamlit.write('**SAMBANOVA_API_KEY**')
            sambanova_api_key = streamlit.text_input(
                label='Enter API key (hidden)',
                type='password',
                help='Enter your API key',
                key='sambanova_api_key',
                label_visibility='collapsed',
            )
            if sambanova_api_key:
                os.environ['SAMBANOVA_API_KEY'] = sambanova_api_key

            # if not os.getenv('SAMBANOVA_API_KEY'):
            streamlit.write('**SERPER_API_KEY (Optional)**')
            serper_api_key = streamlit.text_input(
                label='Enter API key (hidden)',
                type='password',
                help='Enter your API key',
                key='serper_api_key',
                label_visibility='collapsed',
            )
            if serper_api_key:
                os.environ['SERPER_API_KEY'] = serper_api_key
            pass

    # -------------------- OUTPUT COLUMNS --------------------
    output_col1, output_col2 = streamlit.columns(2)

    # Left panel: Agent Progress
    with output_col1:
        streamlit.markdown("<h4 style='margin: 0 0 0.5rem;'>🔄 Agent Progress</h4>", unsafe_allow_html=True)
        # We keep an empty container that we'll fill with logs or a placeholder
        execution_output = streamlit.empty()

        if not streamlit.session_state.running and not streamlit.session_state.final_content:
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
        streamlit.markdown("<h4 style='margin: 0 0 0.5rem;'>📑 Financial Report</h4>", unsafe_allow_html=True)
        content_output = streamlit.empty()

        if streamlit.session_state.final_content:
            cleaned_content = clean_markdown_content(streamlit.session_state.final_content)
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
                    <div class="custom-placeholder">
                        Your Financial Report will appear here...
                        <br><br>
                        Click 'Generate' to begin.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------- ACTIONS ON SUBMIT --------------------
    if generate_button:
        if not query:
            streamlit.error('❌ Please enter a query.')
            return

        if not os.getenv('SAMBANOVA_API_KEY'):
            streamlit.error('❌ Please provide an API key.')
            return

        try:
            start_time = time.time()
            streamlit.session_state.running = True
            streamlit.session_state.final_content = None

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
                            <p style="margin-top: 15px; color: #ee7624;">
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
            with streamlit.spinner('💸 Our financial agents are working on your query...'):
                with st_capture(execution_output.code):
                    financial_flow = FinancialFlow(
                        query=query,
                        source_generic_search=generic_research_option,
                        source_sec_filings=sec_edgar_option,
                        source_yfinance_news=yfinance_news_option,
                        source_yfinance_stocks=yfinance_stocks_option,
                    )
                    financial_flow.kickoff()

            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            time_msg = f'⚡ Generated in {minutes}m {seconds}s' if minutes > 0 else f'⚡ Generated in {seconds}s'

            streamlit.success('✨ Content generated successfully!')
            streamlit.markdown(f"{time_msg} using SambaNova's lightning-fast inference engine")

            # If the output file was written, show it
            output_file = f'output/.md'.replace(' ', '_')
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    final_md = f.read()
                    streamlit.session_state.final_content = final_md
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
                col_dl1, col_dl2 = streamlit.columns([3, 1])
                with col_dl2:
                    streamlit.download_button(
                        label='📥 Download as Markdown',
                        data=final_md,
                        file_name=f'financial_report.md',
                        mime='text/markdown',
                        help='Download your generated content as a markdown file',
                    )

        except Exception as e:
            streamlit.error(f'❌ An error occurred: {str(e)}')
            streamlit.info('🔄 Please try again or contact support if the issue persists.')
        finally:
            streamlit.session_state.running = False


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'running' not in streamlit.session_state:
        streamlit.session_state.running = False
    if 'final_content' not in streamlit.session_state:
        streamlit.session_state.final_content = None
    if 'show_api_warning' not in streamlit.session_state:
        streamlit.session_state.show_api_warning = False


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


@contextmanager
def st_capture(output_func: Any) -> Generator[StringIO, None, None]:
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

        stdout.write = new_write  # type: ignore
        yield stdout


if __name__ == '__main__':
    main()
