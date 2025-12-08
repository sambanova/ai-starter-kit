import base64
import datetime
import os
import sys
import time

from dotenv import load_dotenv

# Main directories
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)
import streamlit

from financial_assistant.constants import *
from financial_assistant.streamlit.utilities_app import (
    clear_cache,
    create_temp_dir_with_subdirs,
    delete_all_subdirectories,
    display_directory_contents,
    initialize_session,
    schedule_temp_dir_deletion,
    set_css_styles,
    submit_sec_edgar_details,
)
from utils.visual.env_utils import are_credentials_set, save_credentials

if not prod_mode:
    load_dotenv(os.path.join(repo_dir, '.env'))

# Initialize session
initialize_session(streamlit.session_state, prod_mode)


# Set CSS styles
set_css_styles()

# Add SambaNova logo
streamlit.logo(image=os.path.join(repo_dir, 'images', 'dark-logo.png'))

# Title of the main page
columns = streamlit.columns([0.15, 0.85], vertical_alignment='top')
columns[0].image(os.path.join(repo_dir, 'images', 'dark-logo.png'))
columns[1].title('SambaNova Financial Assistant')

# Home page
if not are_credentials_set():
    streamlit.title('Financial Insights with LLMs')
    streamlit.write(INTRODUCTION_TEXT)

# Add sidebar
with streamlit.sidebar:
    # Inject HTML to display the logo in the sidebar at 70% width
    logo_path = os.path.join(repo_dir, 'images', 'dark-logo.png')
    with open(logo_path, 'rb') as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    streamlit.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{encoded}" style="width:60%; display: block; max-width:100%;">
        </div>
    """,
        unsafe_allow_html=True,
    )

    if not are_credentials_set():
        # Get the SambaNova API Key
        streamlit.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')
        api_key = streamlit.text_input('SAMBANOVA CLOUD API KEY')
        if streamlit.button('Save Credentials', key='save_credentials_sidebar'):
            message = save_credentials(api_key=api_key, prod_mode=prod_mode)
            streamlit.success(message)
            streamlit.session_state.mp_events.api_key_saved()
            streamlit.rerun()
        else:
            streamlit.stop()
    else:
        if prod_mode:
            streamlit.success('Credentials are set')
            if streamlit.button('Clear Credentials', key='clear-credentials'):
                save_credentials(api_key='', prod_mode=prod_mode)
                streamlit.success(r':orange[You have been logged out.]')
                time.sleep(2)
                streamlit.rerun()


from financial_assistant.src.utilities import get_logger
from financial_assistant.streamlit.app_financial_filings import include_financial_filings
from financial_assistant.streamlit.app_pdf_report import include_pdf_report
from financial_assistant.streamlit.app_stock_data import get_stock_data_analysis
from financial_assistant.streamlit.app_stock_database import get_stock_database
from financial_assistant.streamlit.app_yfinance_news import get_yfinance_news
from financial_assistant.streamlit.utilities_methods import stream_chat_history

logger = get_logger()


def main() -> None:
    with streamlit.sidebar:
        # Create the cache and its main subdirectories
        if are_credentials_set() and not os.path.exists(streamlit.session_state.cache_dir):
            # List the main cache subdirectories
            subdirectories = [
                streamlit.session_state.sources_dir,
                streamlit.session_state.pdf_sources_dir,
                streamlit.session_state.pdf_generation_dir,
            ]

            # In production mode
            create_temp_dir_with_subdirs(streamlit.session_state.cache_dir, subdirectories)

            if prod_mode:
                # In production, schedule deletion after EXIT_TIME_DELTA minutes
                try:
                    schedule_temp_dir_deletion(streamlit.session_state.cache_dir, delay_minutes=EXIT_TIME_DELTA)
                except:
                    logger.warning('Could not schedule deletion of cache directory.')

        # Custom button to exit the app in prod mode
        # This will clear the chat history, delete the cache and clear the SambaNova credentials
        if prod_mode:
            time_delta = datetime.datetime.now() - streamlit.session_state.launch_time
            if (
                streamlit.button('Exit App', help='This will delete the cache.')
                or time_delta.seconds / 30 > EXIT_TIME_DELTA
            ):
                # Crear the chat history
                streamlit.session_state.chat_history = list()
                # Delete the cache
                clear_cache(delete=True)
                # Clear the SambaNova credentials
                save_credentials(api_key='', prod_mode=prod_mode)

                streamlit.success(r':green[The chat history has been cleared.]')
                streamlit.success(r':green[The cache has been deleted.]')
                streamlit.success(r':orange[You have been logged out.]')
                time.sleep(2)
                streamlit.rerun()
                return

        if are_credentials_set():
            # Navigation menu
            streamlit.title('Navigation')
            menu = streamlit.radio(
                'Go to',
                [
                    'Home',
                    'Stock Data Analysis',
                    'Stock Database',
                    'Financial News Scraping',
                    'Financial Filings Analysis',
                    'Generate PDF Report',
                    'Print Chat History',
                ],
            )

            # Add saved files
            streamlit.title('Saved Files')

            # Custom button to clear all files
            if streamlit.button(
                label='Clear All Files',
                key='clear-files',
                help='This will delete all saved files.',
            ):
                try:
                    clear_cache(delete=False)
                    # List the main cache subdirectories
                    subdirectories = [
                        streamlit.session_state.sources_dir,
                        streamlit.session_state.pdf_sources_dir,
                        streamlit.session_state.pdf_generation_dir,
                    ]
                    delete_all_subdirectories(directory=streamlit.session_state.cache_dir, exclude=subdirectories)
                    # Clear chat history
                    streamlit.session_state.chat_history = list()
                    streamlit.sidebar.success('All files have been deleted.')
                except:
                    pass

            # Use Streamlit's session state to persist the current path
            if 'current_path' not in streamlit.session_state:
                streamlit.session_state.current_path = streamlit.session_state.cache_dir

            if os.path.exists(streamlit.session_state.cache_dir):
                # Input to allow user to go back to a parent directory, up to the cache, but not beyond the cache
                if (
                    streamlit.sidebar.button('⬅️ Back', key=f'back')
                    and streamlit.session_state.cache_dir in streamlit.session_state.current_path
                ):
                    streamlit.session_state.current_path = os.path.dirname(streamlit.session_state.current_path)

                    # Display the current directory contents
                    display_directory_contents(streamlit.session_state.current_path, streamlit.session_state.cache_dir)
                else:
                    # Display the current directory contents
                    display_directory_contents(streamlit.session_state.current_path, streamlit.session_state.cache_dir)

    if are_credentials_set():
        # Home page
        if menu == 'Home':
            streamlit.title('Financial Insights with LLMs')
            streamlit.write(INTRODUCTION_TEXT)

        # Stock Data Analysis page
        elif menu == 'Stock Data Analysis':
            get_stock_data_analysis()

        # Stock Database page
        elif menu == 'Stock Database':
            get_stock_database()

        # Financial News Scraping page
        elif menu == 'Financial News Scraping':
            get_yfinance_news()

        # Financial Filings Analysis page
        elif menu == 'Financial Filings Analysis':
            # Populate SEC-EDGAR credentials
            submit_sec_edgar_details()
            if os.getenv('SEC_API_ORGANIZATION') is not None and os.getenv('SEC_API_EMAIL') is not None:
                include_financial_filings()

        # Generate PDF Report page
        elif menu == 'Generate PDF Report':
            include_pdf_report()

        # Print Chat History page
        elif menu == 'Print Chat History':
            # Custom button to clear chat history
            if streamlit.button('Clear Chat History', help='This will delete the chat history.'):
                streamlit.session_state.chat_history = list()
                # Log message
                streamlit.success(f'Cleared chat history.')

            # Add button to stream chat history
            if streamlit.button('Print Chat History'):
                if len(streamlit.session_state.chat_history) == 0:
                    streamlit.warning('No chat history to show.')
                else:
                    stream_chat_history()


if __name__ == '__main__':
    main()
