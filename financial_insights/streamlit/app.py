import logging
import os

import streamlit
from streamlit_extras.stylable_container import stylable_container

from financial_insights.streamlit.app_financial_filings import get_financial_filings
from financial_insights.streamlit.app_pdf_report import get_pdf_report
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

logging.basicConfig(level=logging.INFO)


def main() -> None:
    # clear_directory(CACHE_DIR + 'sources')
    global output

    # Streamlit app setup
    streamlit.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        layout='wide',
    )

    set_css_styles()

    with streamlit.sidebar:
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
                clear_directory(CACHE_DIR + 'stock_query_figures/')
                clear_directory(CACHE_DIR + 'history_figures/')
                clear_directory(CACHE_DIR + 'db_query_figures/')
                clear_directory(CACHE_DIR + 'pdf_generation/')
                streamlit.sidebar.success('All files have been deleted.')

        # Set the default path (you can change this to any desired default path)
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

    if 'fc' not in streamlit.session_state:
        streamlit.session_state.fc = None
    if 'chat_history' not in streamlit.session_state:
        streamlit.session_state.chat_history = list()
    if 'tools' not in streamlit.session_state:
        streamlit.session_state.tools = ['get_time', 'python_repl', 'query_db']
    if 'max_iterations' not in streamlit.session_state:
        streamlit.session_state.max_iterations = 5

    streamlit.title(':orange[SambaNova] Financial Insights Assistant')

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
        get_financial_filings()

    # Generate PDF Report page
    elif menu == 'Generate PDF Report':
        get_pdf_report()

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
