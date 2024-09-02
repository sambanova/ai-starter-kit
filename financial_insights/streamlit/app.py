import os
import sys

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

from financial_insights.src.utilities_retrieval import VectorstoreRegistry
from financial_insights.streamlit.app_financial_filings import include_financial_filings
from financial_insights.streamlit.app_pdf_report import include_pdf_report
from financial_insights.streamlit.app_stock_data import get_stock_data_analysis
from financial_insights.streamlit.app_stock_database import get_stock_database
from financial_insights.streamlit.app_yfinance_news import get_yfinance_news
from financial_insights.streamlit.constants import *
from financial_insights.streamlit.utilities_app import (
    clear_directory,
    display_directory_contents,
    get_blue_button_style,
    set_css_styles,
)
from financial_insights.streamlit.utilities_methods import stream_chat_history
from utils.visual.env_utils import are_credentials_set, env_input_fields, save_credentials

# Initialize Weave with your project name
if os.getenv('WANDB_API_KEY') is not None:
    weave.init('sambanova_financial_insights')


def main() -> None:
    # Create cache
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

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
    # Initialize credentials
    streamlit.session_state.FASTAPI_URL = os.getenv('FASTAPI_URL', '')
    streamlit.session_state.FASTAPI_API_KEY = os.getenv('FASTAPI_API_KEY', '')

    # Load the config
    with open(CONFIG_PATH, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    # Get the production flag
    prod_mode = config['prod_mode']

    with streamlit.sidebar:
        if not are_credentials_set():
            url, api_key = env_input_fields()
            if streamlit.button('Save Credentials', key='save_credentials_sidebar'):
                message = save_credentials(url, api_key, prod_mode)
                streamlit.success(message)
                streamlit.rerun()
        else:
            streamlit.success('Credentials are set')
            if streamlit.button('Clear Credentials', key='clear_credentials'):
                save_credentials('', '', prod_mode)
                streamlit.rerun()

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
                    clear_directory(CACHE_DIR)
                    clear_directory(SOURCE_DIR)
                    clear_directory(STOCK_QUERY_FIGURES_DIR)
                    clear_directory(HISTORY_FIGURES_DIR)
                    clear_directory(DB_QUERY_FIGURES_DIR)
                    clear_directory(PDF_SOURCES_DIRECTORY)
                    clear_directory(PDF_GENERATION_DIRECTORY)
                    streamlit.sidebar.success('All files have been deleted.')

            # Set the default path
            default_path = './financial_insights/streamlit/cache'
            # Use Streamlit's session state to persist the current path
            if 'current_path' not in streamlit.session_state:
                streamlit.session_state.current_path = default_path

            # Input to allow user to go back to a parent directory
            if streamlit.sidebar.button('⬅️ Back', key=f'back') and streamlit.session_state.current_path != default_path:
                streamlit.session_state.current_path = os.path.dirname(streamlit.session_state.current_path)

                # Display the current directory contents
                display_directory_contents(streamlit.session_state.current_path, default_path)
            else:
                # Display the current directory contents
                display_directory_contents(streamlit.session_state.current_path, default_path)

    # Initialize the chat history
    if 'chat_history' not in streamlit.session_state:
        streamlit.session_state.chat_history = list()
    # Initialize function calling
    if 'fc' not in streamlit.session_state:
        streamlit.session_state.fc = None
    # Initialize the registry only once and store it in Streamlit session state
    if 'vectorstore_registry' not in streamlit.session_state:
        streamlit.session_state.vectorstore_registry = VectorstoreRegistry()

    columns = streamlit.columns([0.15, 0.85], vertical_alignment='top')
    columns[0].image(SAMBANOVA_LOGO, width=100)
    columns[1].title('SambaNova Financial Assistant')

    if are_credentials_set():
        # Home page
        if menu == 'Home':
            streamlit.title('Financial Insights with LLMs')

            streamlit.write(
                """
                Welcome to the Financial Insights application.
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

        elif menu == 'Stock Database':
            get_stock_database()

        # Financial News Scraping page
        elif menu == 'Financial News Scraping':
            get_yfinance_news()

        # Financial Filings Analysis page
        elif menu == 'Financial Filings Analysis':
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
                    streamlit.write(f'Cleared chat history.')

            # Add button to stream chat history
            if streamlit.button('Print Chat History'):
                if len(streamlit.session_state.chat_history) == 0:
                    streamlit.write('No chat history to show.')
                else:
                    stream_chat_history()


if __name__ == '__main__':
    main()
