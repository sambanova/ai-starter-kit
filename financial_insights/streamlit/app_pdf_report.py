import logging
import os
import shutil
from typing import Any, Dict, Union

import streamlit
from streamlit.runtime.uploaded_file_manager import UploadedFile

from financial_insights.src.function_calling import FunctionCallingLlm
from financial_insights.src.tools_pdf_generation import generate_pdf, parse_documents, read_txt_files
from financial_insights.streamlit.constants import *
from financial_insights.streamlit.utilities_app import clear_directory, save_output_callback
from financial_insights.streamlit.utilities_methods import handle_userinput, set_fc_llm

logging.basicConfig(level=logging.INFO)


def get_pdf_report() -> None:
    # Clean the directory
    if os.path.exists(PDF_GENERATION_DIRECTORY):
        clear_directory(PDF_GENERATION_DIRECTORY)

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

    # Include summary for each section
    include_summary = streamlit.checkbox(
        'Include summary from each section', key='checkbox_summary', value=False, help='This will take longer!'
    )
    if include_summary:
        streamlit.write(':red[Warning: This will take longer!]')

    # Generate the report
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

    # Initialize session state for checkboxes if not already done
    if 'checkbox_use_generated_pdf' not in streamlit.session_state:
        streamlit.session_state['checkbox_use_generated_pdf'] = False
    if 'checkbox_upload_your_pdf' not in streamlit.session_state:
        streamlit.session_state['checkbox_upload_your_pdf'] = False

    use_generated_pdf = streamlit.checkbox(
        'Use generated PDF',
        key='checkbox_use_generated_pdf',
        value=False,
        on_change=check_use_generated_pdf,
    )

    upload_your_pdf = streamlit.checkbox(
        'Upload your PDF',
        key='checkbox_upload_your_pdf',
        value=False,
        on_change=check_upload_your_pdf,
    )

    if streamlit.button('Use report for RAG'):
        user_request = streamlit.text_input(
            'Enter the info that you want to retrieve for given companies.',
            key='pdf-rag',
        )

        if use_generated_pdf or upload_your_pdf:
            if use_generated_pdf and not upload_your_pdf:
                answer = handle_pdf_rag(user_request, report_name)
            elif upload_your_pdf and not use_generated_pdf:
                # Add a PDF document for RAG (optional)
                your_pdf_file = streamlit.file_uploader('Upload a PDF document for RAG (optional):', type='pdf')
                if your_pdf_file is not None:
                    answer = handle_pdf_rag(user_request, your_pdf_file)

            if streamlit.button(
                'Save Answer',
                on_click=save_output_callback,
                args=(answer, PDF_RAG_PATH),
            ):
                pass


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
    return


def check_use_generated_pdf() -> None:
    if streamlit.session_state['checkbox_use_generated_pdf']:
        streamlit.session_state['checkbox_upload_your_pdf'] = False


def check_upload_your_pdf() -> None:
    if streamlit.session_state['checkbox_upload_your_pdf']:
        streamlit.session_state['checkbox_use_generated_pdf'] = False


def handle_pdf_rag(user_question: str, report_name: Union[str, UploadedFile]) -> Any:
    streamlit.session_state.tools = ['pdf_rag']
    set_fc_llm(streamlit.session_state.tools)

    user_request = 'Please use RAG from the provided PDF file to answer the following question: ' + user_question
    response = handle_userinput(user_question, user_request)
    return response
