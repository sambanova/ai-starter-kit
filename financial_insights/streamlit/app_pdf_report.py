import logging
import os
import shutil
from typing import Dict

import streamlit

from financial_insights.src.function_calling import FunctionCallingLlm
from financial_insights.src.utilities_pdf_generation import (generate_pdf,
                                                             parse_documents,
                                                             read_txt_files)
from financial_insights.streamlit.constants import *

logging.basicConfig(level=logging.INFO)


def get_pdf_report() -> None:
    streamlit.markdown('<h2> Generate PDF Report </h2>', unsafe_allow_html=True)
    # Initialize session state for checkboxes if not already done
    if 'checkbox_include_stock' not in streamlit.session_state:
        streamlit.session_state['checkbox_include_stock'] = False
    if 'checkbox_include_database' not in streamlit.session_state:
        streamlit.session_state['checkbox_include_database'] = False
    if 'checkbox_include_yahoo_news' not in streamlit.session_state:
        streamlit.session_state['checkbox_include_yahoo_news'] = False
    if 'checkbox_include_filings' not in streamlit.session_state:
        streamlit.session_state['checkbox_include_filings'] = False
    if 'checkbox_generate_from_history' not in streamlit.session_state:
        streamlit.session_state['checkbox_generate_from_history'] = False

    include_stock = streamlit.checkbox(
        'Include saved stock queries',
        key='checkbox_include_stock',
        value=False,
        on_change=check_inclusions,
    )
    include_database = streamlit.checkbox(
        'Include saved stock database queries',
        key='checkbox_include_database',
        value=False,
        on_change=check_inclusions,
    )
    inlude_yahoo_news = streamlit.checkbox(
        'Include saved Yahoo News queries',
        key='checkbox_include_yahoo_news',
        value=False,
        on_change=check_inclusions,
    )
    include_filings = streamlit.checkbox(
        'Include saved financial filings queries',
        key='checkbox_include_filings',
        value=False,
        on_change=check_inclusions,
    )
    streamlit.markdown('---')
    generate_from_history = streamlit.checkbox(
        'Generate from the whole chat history',
        key='checkbox_generate_from_history',
        value=False,
        on_change=check_generate_from_history,
    )

    data_paths = dict()
    if include_stock:
        # Add data from Stock Data Analysis
        data_paths['stock_query'] = STOCK_QUERY_PATH
    if include_database:
        # Add data from Stock Database Analysis
        data_paths['stock_database'] = DB_QUERY_PATH
    if inlude_yahoo_news:
        # Add data from Yahoo News Analysis
        data_paths['yfinance_news'] = YFINANCE_NEWS_PATH
    if include_filings:
        # Add data from Financial Filings Analysis
        data_paths['filings'] = FILINGS_PATH
    if generate_from_history:
        # Deselect all other options
        include_stock = False
        include_database = False
        inlude_yahoo_news = False
        include_filings = False

        # Add data from chat history
        data_paths['history'] = HISTORY_PATH

    # Add title name (optional)
    title_name = streamlit.text_input('Title Name', 'Financial Report')

    include_summary = streamlit.checkbox(
        'Include summary from each section', key='checkbox_summary', value=False, help='This will take longer!'
    )
    if include_summary:
        streamlit.write(':red[Warning: This will take longer!]')

    if streamlit.button('Generate Report'):
        with streamlit.expander('**Execution scratchpad**', expanded=True):
            report_name = title_name.lower().replace(' ', '_') + '.pdf'

            # generate_pdf_report(data)
            handle_pdf_generation(title_name, report_name, data_paths, include_summary)

            with open(PDF_GENERATION_DIRECTORY + report_name, encoding='utf8', errors='ignore') as f:
                file_content = f.read()
                streamlit.download_button(
                    label=report_name,
                    data=file_content,
                    file_name=report_name,
                    mime='text/plain',
                )
        streamlit.write('PDF report generated successfully.')


def handle_pdf_generation(
    title_name: str = 'Financial Report',
    report_name: str = 'financial_report',
    data_paths: Dict[str, str] = dict(),
    include_summary: bool = False,
) -> None:
    streamlit.session_state.fc = FunctionCallingLlm()

    output_file = PDF_GENERATION_DIRECTORY + report_name

    # Ensure the destination directory exists
    os.makedirs(PDF_GENERATION_DIRECTORY, exist_ok=True)

    # Check that at least one data source is available
    assert any([data_paths[key] for key in data_paths]), 'Select at least one data source.'

    # Assert that at least one data source exists as a file
    assert any([os.path.isfile(data_paths[key]) for key in data_paths]), 'No data source available.'

    for source_file in data_paths.values():
        # Create the full path for the destination file
        destination_file = os.path.join(PDF_GENERATION_DIRECTORY, os.path.basename(source_file))

        try:
            # Copy selected document to pdf generation directory
            shutil.copy(source_file, destination_file)

            logging.info(f'{source_file} has been copied to {destination_file}')
        except Exception as e:
            logging.error('Error while copying file', exc_info=True)

    documents = read_txt_files(PDF_GENERATION_DIRECTORY)
    report_content = parse_documents(documents)
    generate_pdf(report_content, output_file, title_name, include_summary)


def check_generate_from_history() -> None:
    if streamlit.session_state['checkbox_generate_from_history']:
        streamlit.session_state['checkbox_include_stock'] = False
        streamlit.session_state['checkbox_include_database'] = False
        streamlit.session_state['checkbox_include_yahoo_news'] = False
        streamlit.session_state['checkbox_include_filings'] = False


def check_inclusions() -> None:
    if (
        streamlit.session_state['checkbox_include_stock']
        or streamlit.session_state['checkbox_include_database']
        or streamlit.session_state['checkbox_include_yahoo_news']
        or streamlit.session_state['checkbox_include_filings']
    ):
        streamlit.session_state['checkbox_generate_from_history'] = False
