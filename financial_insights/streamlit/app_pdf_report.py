import logging
import os
import shutil
import sys
from typing import Any, Dict

import streamlit

from financial_insights.src.function_calling import FunctionCallingLlm
from financial_insights.src.utilities_pdf_generation import generate_pdf, parse_documents, read_txt_files

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

logging.basicConfig(level=logging.INFO)
TEMP_DIR = 'financial_insights/streamlit/cache/'


def get_pdf_report() -> None:
    streamlit.markdown('<h2> Generate PDF Report </h2>', unsafe_allow_html=True)
    # Initialize session state for checkboxes if not already done
    if 'checkbox_include_stock' not in streamlit.session_state:
        streamlit.session_state['checkbox_include_stock'] = False
    if 'checkbox_include_database' not in streamlit.session_state:
        streamlit.session_state['checkbox_include_database'] = False
    if 'checkbox_inlude_yahoo_news' not in streamlit.session_state:
        streamlit.session_state['checkbox_include_yahoo_news'] = False
    if 'checkbox_include_filings' not in streamlit.session_state:
        streamlit.session_state['checkbox_include_filings'] = False
    if 'checkbox_generate_from_history' not in streamlit.session_state:
        streamlit.session_state['checkbox_generate_from_history'] = False

    def check_generate_from_history():
        if streamlit.session_state['checkbox_generate_from_history']:
            streamlit.session_state['checkbox_include_stock'] = False
            streamlit.session_state['checkbox_include_database'] = False
            streamlit.session_state['checkbox_include_yahoo_news'] = False
            streamlit.session_state['checkbox_include_filings'] = False

        if (
            streamlit.session_state['checkbox_include_stock']
            or streamlit.session_state['checkbox_include_database']
            or streamlit.session_state['checkbox_include_yahoo_news']
            or streamlit.session_state['checkbox_include_filings']
        ):
            streamlit.session_state['checkbox_generate_from_history'] = False

    include_stock = streamlit.checkbox('Include saved stock queries', key='checkbox_include_stock', value=False)
    include_database = streamlit.checkbox(
        'Include saved stock database queries', key='checkbox_include_database', value=False
    )
    inlude_yahoo_news = streamlit.checkbox(
        'Include saved Yahoo News queries', key='checkbox_include_yahoo_news', value=False
    )
    include_filings = streamlit.checkbox(
        'Include saved financial filings queries', key='checkbox_include_filings', value=False
    )
    generate_from_history = streamlit.checkbox(
        'Generate from the whole chat history',
        key='checkbox_generate_from_history',
        value=False,
        on_change=check_generate_from_history,
    )

    # Add title name (optional)
    title_name = streamlit.text_input('Title Name', 'Financial Report')
    if streamlit.button('Generate Report'):
        data_paths = dict()
        if include_stock:
            # Add data from Stock Data Analysis
            data_paths['stock_query'] = TEMP_DIR + 'stock_query.txt'
        if include_database:
            # Add data from Stock Database Analysis
            data_paths['stock_database'] = TEMP_DIR + 'db_query.txt'
        if inlude_yahoo_news:
            # Add data from Yahoo News Analysis
            data_paths['yfinance_news'] = TEMP_DIR + 'yfinance_news.txt'
        if include_filings:
            # Add data from Financial Filings Analysis
            data_paths['filings'] = TEMP_DIR + 'financials.txt'
        if generate_from_history:
            # Deselect all other options
            include_stock = False
            include_database = False
            inlude_yahoo_news = False
            include_filings = False

            # Add data from chat history
            data_paths['history'] = create_chat_history()

        # generate_pdf_report(data)
        handle_pdf_generation(title_name, data_paths)
        streamlit.write('PDF report generated successfully.')


def create_chat_history() -> Any:
    # Create a chat history text file from the user's previous conversations

    print('Break')

    return streamlit.session_state.chat_history


def handle_pdf_generation(title_name: str = 'Financial Report', data_paths: Dict[str, str] = dict()) -> None:
    streamlit.session_state.fc = FunctionCallingLlm()

    destination_directory = TEMP_DIR + 'pdf_generation/'  # Update this path
    output_file = destination_directory + 'financial_report.pdf'

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Check that at least one data source is available
    assert any([data_paths[key] for key in data_paths]), 'Select at least one data source.'

    # Assert that at least one data source exists as a file
    assert any([os.path.isfile(data_paths[key]) for key in data_paths]), 'No data source available.'

    for source_file in data_paths.values():
        # Create the full path for the destination file
        destination_file = os.path.join(destination_directory, os.path.basename(source_file))

        try:
            # Copy selected document to pdf generation directory
            shutil.copy(source_file, destination_file)

            logging.info(f'{source_file} has been copied to {destination_file}')
        except Exception as e:
            logging.error('Error while copying file', exc_info=True)

    documents = read_txt_files(destination_directory)
    report_content = parse_documents(documents)
    generate_pdf(report_content, output_file, title_name)
