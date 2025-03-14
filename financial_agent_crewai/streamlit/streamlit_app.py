"""
Streamlit web application for financial content generation.

This module provides a web interface for generating financial content using
the SambaNova AI models and CrewAI framework. It handles user input, content
generation, and result display.
"""

import base64
import os
import time
from typing import Any
from uuid import uuid4

import streamlit
from dotenv import load_dotenv

from financial_agent_crewai.src.financial_agent_flow.config import *
from financial_agent_crewai.src.financial_agent_flow.main import FinancialFlow
from financial_agent_crewai.utils.utilities import *

# Load environment variables
load_dotenv()

# Unset API KEYS
os.environ.pop('SAMBANOVA_API_KEY', None)
os.environ.pop('SERPER_API_KEY', None)


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
            justify-content: center;
            margin: 0.5rem 0;
        }
        .logo-container img {
            height: 2rem; /* Adjust the height to control the size of the logo */
            margin-right: 0.5rem;
        }
        .logo-container h2 {
            color: #ee7624;
            margin: 0;
            text-align: center;
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
            <img src="https://sambanova.ai/hubfs/logotype_sambanova_orange.png" 
                alt="SambaNova Logo">
            <h2>SambaNova Financial Agent</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------- CREDENTAILS --------------------
    with streamlit.expander('Credentials', icon='🔑'):
        with streamlit.form('credentials'):
            if (
                streamlit.session_state.get(['SAMBANOVA_API_KEY']) is None  # type: ignore
                and streamlit.session_state.get(['SERPER_API_KEY']) is None  # type: ignore
            ):
                sambanova_api_key = streamlit.session_state.get(['SAMBANOVA_API_KEY'])  # type: ignore
                if sambanova_api_key is None:
                    streamlit.write('**SAMBANOVA_API_KEY**')
                    sambanova_api_key = streamlit.text_input(
                        label='Enter API key (hidden)',
                        type='password',
                        help='Enter your API key',
                        key=f'sambanova_api_key',
                        label_visibility='collapsed',
                    )

                serper_api_key = streamlit.session_state.get(['SERPER_API_KEY'])  # type: ignore
                if serper_api_key is None:
                    streamlit.write('**SERPER_API_KEY (Optional)**')
                    serper_api_key = streamlit.text_input(
                        label='Enter API key (hidden)',
                        type='password',
                        help='Enter your API key',
                        key=f'serper_api_key',
                        label_visibility='collapsed',
                    )

                if streamlit.form_submit_button('Save Credentials'):
                    os.environ['SAMBANOVA_API_KEY'] = sambanova_api_key
                    os.environ['SERPER_API_KEY'] = serper_api_key
                    streamlit.session_state['SAMBANOVA_API_KEY'] = sambanova_api_key
                    streamlit.session_state['SERPER_API_KEY'] = serper_api_key
                    streamlit.session_state['SESSION_ID'] = str(uuid4())

    # -------------------- FORM --------------------
    with streamlit.form('generation_form'):
        streamlit.write('**User query**')
        query = streamlit.text_input(
            label='Enter research topic (hidden)',
            placeholder='E.g., What was the research and development spending trend for Google in 2024?',
            help='Enter the main subject for content generation',
            key='compact_topic',
            label_visibility='collapsed',
        )

        with streamlit.expander('Sources'):
            col21, col22 = streamlit.columns([0.5, 0.5], vertical_alignment='center')
            with col21:
                generic_research_option = streamlit.checkbox('Generic Google Search', value=False)
                sec_edgar_option = streamlit.checkbox('SEC Edgar Filings', value=False)
            with col22:
                yfinance_news_option = streamlit.checkbox('Yahoo Finance News', value=False)
                yfinance_stocks_option = streamlit.checkbox('Yahoo Finance Stocks', value=False)

        # Generate button
        generate_button = streamlit.form_submit_button(
            label='🚀 Generate',
            type='secondary',
            disabled=streamlit.session_state.running,
            help='Click to start generating content',
        )

    # -------------------- ACTIONS ON SUBMIT --------------------
    if generate_button:
        if not query:
            streamlit.error('❌ Please enter a query.')
            return

        if (
            streamlit.session_state.get('SAMBANOVA_API_KEY') is None
            or streamlit.session_state.get('SERPER_API_KEY') is None
        ):
            streamlit.error('❌ Please set your API keys.')
            return

        if (
            not generic_research_option
            and not sec_edgar_option
            and not yfinance_news_option
            and not yfinance_stocks_option
        ):
            streamlit.error('❌ Please provide data source.')
        else:
            try:
                start_time = time.time()
                streamlit.session_state.running = True
                streamlit.session_state.final_content = None

                if streamlit.session_state.running:
                    streamlit.markdown("<h4 style='margin: 0 0 0.5rem;'>🔄 Agent Progress</h4>", unsafe_allow_html=True)
                    # We keep an empty container that we'll fill with logs or a placeholder
                    execution_output = streamlit.empty()

                    # Show real-time logs
                    with streamlit.spinner('💸 Our financial agents are working on your query...'):
                        with st_capture(execution_output.json):
                            financial_flow = FinancialFlow(
                                query=query,
                                source_generic_search=generic_research_option,
                                source_sec_filings=sec_edgar_option,
                                source_yfinance_news=yfinance_news_option,
                                source_yfinance_stocks=yfinance_stocks_option,
                                sambanova_api_key=streamlit.session_state['SAMBANOVA_API_KEY'],
                                serper_api_key=streamlit.session_state['SERPER_API_KEY'],
                            )
                            financial_flow.kickoff()

                end_time = time.time()
                elapsed_time = end_time - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                time_msg = f'⚡ Generated in {minutes}m {seconds}s' if minutes > 0 else f'⚡ Generated in {seconds}s'

                output_file = CACHE_DIR / f'report.md'
                streamlit.session_state.running = False
                streamlit.session_state.final_content = output_file
                streamlit.success('✨ Content generated successfully!')
                streamlit.markdown(f"{time_msg} using SambaNova's lightning-fast inference engine")

                # If the output file was written, show it
                if os.path.exists(output_file):
                    output_col1, output_col2 = streamlit.columns([0.5, 0.5], gap='large')
                    with output_col1:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            report_md = f.read()
                            streamlit.session_state.final_content = report_md
                            # Clean the Markdown (base64 images, table classes) -> HTML
                            cleaned_html = clean_markdown_content(report_md)
                            content_output = streamlit.empty()
                            content_output.markdown(
                                f"""
                                <div class="custom-panel">
                                    <div class="markdown-content">
                                        {cleaned_html}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Download the Markdown report
                        download_section(
                            label='📥 Download Markdown',
                            data=report_md,
                            file_name=f'report.md',
                            mime='text/markdown',
                        )

                    with output_col2:
                        # Convert the cleaned HTML to a PDF
                        pdf_data = convert_html_to_pdf(cleaned_html)

                        # Encode the PDF to Base64
                        base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                        pdf_display = (
                            f'<embed src="data:application/pdf;base64,{base64_pdf}"'
                            ' width="700" height="400" type="application/pdf">'
                        )
                        streamlit.markdown(pdf_display, unsafe_allow_html=True)

                        # Download the PDF report
                        download_section(
                            label='📥 Download PDF',
                            data=pdf_data,
                            file_name='report.pdf',
                            mime='application/pdf',
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


@streamlit.fragment
def download_section(
    label: str = '📥 Download',
    data: Any = '',
    file_name: str = 'file.txt',
    mime: str = 'text/plain',
) -> None:
    """This function prevents streamlit from refreshing the page when the download button is clicked."""

    if streamlit.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
    ):
        pass


if __name__ == '__main__':
    main()
