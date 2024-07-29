import logging
import os
import sys

import streamlit
from streamlit_extras.stylable_container import stylable_container

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)
from financial_insights.src.tools_filings import retrieve_filings
from financial_insights.src.tools_stocks import (get_financial_summary,
                                                 get_historical_price,
                                                 get_stock_info,
                                                 retrieve_symbol_list,
                                                 retrieve_symbol_quantity_list)
from financial_insights.src.tools_yahoo_news import scrape_yahoo_finance_news
from financial_insights.streamlit.app_custom_queries import get_custom_queries
from financial_insights.streamlit.app_financial_filings import \
    get_financial_filings
from financial_insights.streamlit.app_pdf_report import get_pdf_report
from financial_insights.streamlit.app_stock_data import get_stock_data_analysis
from financial_insights.streamlit.app_stock_database import get_stock_database
from financial_insights.streamlit.app_yfinance_news import get_yfinance_news
from financial_insights.streamlit.utilities_app import (
    clear_directory, get_custom_button_style, list_files_in_directory,
    set_css_styles)
from financial_insights.streamlit.utilities_methods import stream_chat_history

logging.basicConfig(level=logging.INFO)

TEMP_DIR = 'financial_insights/streamlit/cache/'


def main() -> None:
    clear_directory(TEMP_DIR + 'sources')
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
                'Query Stock Database',
                'Financial News Scraping',
                'Financial Filings Analysis',
                'Custom Queries',
                'Generate PDF Report',
                'Print Chat History',
            ],
        )

        streamlit.title('Saved Files')

        files = list_files_in_directory(TEMP_DIR)

        # Custom button to clear all files
        with stylable_container(
            key='blue-button',
            css_styles=get_custom_button_style(),
        ):
            if streamlit.button(
                label='Clear All Files',
                key='clear-button',
                help='This will delete all saved files',
            ):
                clear_directory(TEMP_DIR)
                streamlit.sidebar.success('All files have been deleted.')

        if files:
            for file in files:
                file_path = os.path.join(TEMP_DIR, file)
                with open(file_path, 'r') as f:
                    try:
                        file_content = f.read()
                        streamlit.sidebar.download_button(
                            label=f'{file}',
                            data=file_content,
                            file_name=file,
                            mime='text/plain',
                        )
                    except Exception as e:
                        logging.warning('Error reading file', str(e))
                    except FileNotFoundError as e:
                        logging.warning('File not found', str(e))
        else:
            streamlit.write('No files found')

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
            
            - Stock Data Analysis
            - Query Stock Database
            - Financial News Scraping
            - Financial Filings Analysis
            - Custom Queries
            - Generate PDF Report
            - Print Chat History
        """
        )

    # Stock Data Analysis page
    elif menu == 'Stock Data Analysis':
        get_stock_data_analysis()

    elif menu == 'Query Stock Database':
        get_stock_database()

    # Financial News Scraping page
    elif menu == 'Financial News Scraping':
        get_yfinance_news()

    # Financial Filings Analysis page
    elif menu == 'Financial Filings Analysis':
        get_financial_filings()

    # Custom Queries page
    elif menu == 'Custom Queries':
        get_custom_queries()

    # Generate PDF Report page
    elif menu == 'Generate PDF Report':
        get_pdf_report()

    # Print Chat History page
    elif menu == 'Print Chat History':
        # Custom button to clear chat history
        with stylable_container(
            key='blue-button',
            css_styles=get_custom_button_style(),
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
