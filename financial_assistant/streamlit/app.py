import datetime
import os
import sys
import time

import weave
import yaml

# Main directories
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

import streamlit
from streamlit_extras.stylable_container import stylable_container

from financial_assistant.src.tools import get_logger
from financial_assistant.streamlit.app_financial_filings import include_financial_filings
from financial_assistant.streamlit.app_pdf_report import include_pdf_report
from financial_assistant.streamlit.app_stock_data import get_stock_data_analysis
from financial_assistant.streamlit.app_stock_database import get_stock_database
from financial_assistant.streamlit.app_yfinance_news import get_yfinance_news
from financial_assistant.streamlit.constants import *
from financial_assistant.streamlit.utilities_app import (
    clear_cache,
    create_temp_dir_with_subdirs,
    delete_all_subdirectories,
    display_directory_contents,
    get_blue_button_style,
    initialize_session,
    schedule_temp_dir_deletion,
    set_css_styles,
    submit_sec_edgar_details,
)
from financial_assistant.streamlit.utilities_methods import stream_chat_history
from utils.visual.env_utils import are_credentials_set, env_input_fields, save_credentials

# Initialize Weave with your project name
if os.getenv('WANDB_API_KEY') is not None:
    weave.init('sambanova_financial_assistant')

# Load the config
with open(CONFIG_PATH, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
# Get the production flag
prod_mode = config['prod_mode']

logger = get_logger()


def main() -> None:
    # Initialize session
    initialize_session(streamlit.session_state, prod_mode)

    # Streamlit app setup
    streamlit.set_page_config(
        page_title='Finance App',
        page_icon=SAMBANOVA_LOGO,
        layout='wide',
    )

    # Set CSS styles
    set_css_styles()

    # Add SambaNova logo
    streamlit.logo(
        image=SAMBANOVA_LOGO,
        link=SAMBANOVA_LOGO,
        icon_image=SAMBANOVA_LOGO,
    )

    # Add sidebar
    with streamlit.sidebar:
        if not are_credentials_set():
            # Get the SambaNova API Key
            streamlit.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')
            url, api_key = env_input_fields()
            if streamlit.button('Save Credentials', key='save_credentials_sidebar'):
                message = save_credentials(url, api_key, prod_mode)  # type: ignore
                streamlit.success(message)
                streamlit.rerun()
        else:
            streamlit.success('Credentials are set')
            with stylable_container(
                key='blue-button',
                css_styles=get_blue_button_style(),
            ):
                if streamlit.button('Clear Credentials', key='clear_credentials'):
                    save_credentials('', '', prod_mode)  # type: ignore
                    streamlit.success(r':orange[You have been logged out.]')
                    time.sleep(2)
                    streamlit.rerun()

        # Create the cache and its main subdirectories
        if are_credentials_set() and not os.path.exists(streamlit.session_state.cache_dir):
            # List the main cache subdirectories
            subdirectories = [
                streamlit.session_state.source_dir,
                streamlit.session_state.pdf_sources_directory,
                streamlit.session_state.pdf_generation_directory,
            ]

            if prod_mode:
                # In production mode
                create_temp_dir_with_subdirs(streamlit.session_state.cache_dir, subdirectories)

                # In production, schedule deletion after EXIT_TIME_DELTA minutes
                try:
                    schedule_temp_dir_deletion(streamlit.session_state.cache_dir, delay_minutes=EXIT_TIME_DELTA)
                except:
                    logger.warning('Could not schedule deletion of cache directory.')

            else:
                # In development mode
                create_temp_dir_with_subdirs(streamlit.session_state.cache_dir, subdirectories)

        # Custom button to exit the app in prod mode
        # This will clear the chat history, delete the cache and clear the SambaNova credentials
        if prod_mode:
            with stylable_container(
                key='blue-button',
                css_styles=get_blue_button_style(),
            ):
                time_delta = datetime.datetime.now() - streamlit.session_state.launch_time
                if (
                    streamlit.button('Exit App', help='This will delete the cache!')
                    or time_delta.seconds / 30 > EXIT_TIME_DELTA
                ):
                    # Crear the chat history
                    streamlit.session_state.chat_history = list()
                    # Delete the cache
                    clear_cache(delete=True)
                    # Clear the SambaNova credentials
                    save_credentials('', '', prod_mode)  # type: ignore

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
            with stylable_container(
                key='blue-button',
                css_styles=get_blue_button_style(),
            ):
                if streamlit.button(
                    label='Clear All Files',
                    key='clear-button',
                    help='This will delete all saved files',
                ):
                    try:
                        clear_cache(delete=False)
                        # List the main cache subdirectories
                        subdirectories = [
                            streamlit.session_state.source_dir,
                            streamlit.session_state.pdf_sources_directory,
                            streamlit.session_state.pdf_generation_directory,
                        ]
                        delete_all_subdirectories(directory=streamlit.session_state.cache_dir, exclude=subdirectories)

                        streamlit.sidebar.success('All files have been deleted.')
                    except:
                        pass

            # Set the default path
            cache_dir = streamlit.session_state.cache_dir
            # Use Streamlit's session state to persist the current path
            if 'current_path' not in streamlit.session_state:
                streamlit.session_state.current_path = cache_dir

            if os.path.exists(cache_dir):
                # Input to allow user to go back to a parent directory, up to the cache, but not beyond the cache
                if (
                    streamlit.sidebar.button('⬅️ Back', key=f'back')
                    and cache_dir in streamlit.session_state.current_path
                ):
                    streamlit.session_state.current_path = os.path.dirname(streamlit.session_state.current_path)

                    # Display the current directory contents
                    display_directory_contents(streamlit.session_state.current_path, cache_dir)
                else:
                    # Display the current directory contents
                    display_directory_contents(streamlit.session_state.current_path, cache_dir)

    # Title of the main page
    columns = streamlit.columns([0.15, 0.85], vertical_alignment='top')
    columns[0].image(SAMBANOVA_LOGO, width=100)
    columns[1].title('SambaNova Financial Assistant')

    if are_credentials_set():
        # Home page
        if menu == 'Home':
            streamlit.title('Financial Insights with LLMs')

            streamlit.write(
                """
                    Welcome to SambaNova Financial Assistant.
                    This app demonstrates the capabilities of large language models (LLMs)
                    in extracting and analyzing financial data using function calling, web scraping,
                    and retrieval-augmented generation (RAG).
                    
                    Use the navigation menu to explore various features including:
                    
                    - **Stock Data Analysis**: Query and analyze stocks based on Yahoo Finance data.
                    - **Stock Database**: Create and query an SQL database based on Yahoo Finance data.
                    - **Financial News Scraping**: Scrape financial news articles from Yahoo Finance News.
                    - **Financial Filings Analysis**: Query and analyze financial filings based on SEC EDGAR data.
                    - **Generate PDF Report**: Generate a PDF report based on the saved answered queries
                        or on the whole chat history.
                    - **Print Chat History**: Print the whole chat history.
                """
            )

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
            if (
                streamlit.session_state.SEC_API_ORGANIZATION is not None
                and streamlit.session_state.SEC_API_EMAIL is not None
            ):
                include_financial_filings()

        # Generate PDF Report page
        elif menu == 'Generate PDF Report':
            include_pdf_report()

        # Print Chat History page
        elif menu == 'Print Chat History':
            # Custom button to clear chat history
            with stylable_container(
                key='blue-button',
                css_styles=get_blue_button_style(),
            ):
                if streamlit.button('Clear Chat History'):
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
