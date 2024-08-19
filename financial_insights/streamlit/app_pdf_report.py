import logging
import os
import shutil
from typing import Any, Dict, List

import streamlit
from streamlit.runtime.uploaded_file_manager import UploadedFile

from financial_insights.src.function_calling import FunctionCalling
from financial_insights.src.tools_pdf_generation import generate_pdf, parse_documents, read_txt_files
from financial_insights.streamlit.constants import *
from financial_insights.streamlit.utilities_app import clear_directory, save_output_callback
from financial_insights.streamlit.utilities_methods import handle_userinput, set_fc_llm

logging.basicConfig(level=logging.INFO)


def get_pdf_report() -> None:
    streamlit.session_state['report_name'] = None
    # Initialize session state for selected files
    if 'selected_files' not in streamlit.session_state:
        streamlit.session_state.selected_files = []
    if 'uploaded_files' not in streamlit.session_state:
        streamlit.session_state.uploaded_files = []

    streamlit.markdown('<h2> Generate PDF Report </h2>', unsafe_allow_html=True)
    # Initialize session state for checkboxes if not already done
    if 'checkbox_include_stock' not in streamlit.session_state:
        streamlit.session_state.checkbox_include_stock = False
    if 'checkbox_include_database' not in streamlit.session_state:
        streamlit.session_state.checkbox_include_database = False
    if 'checkbox_include_yahoo_news' not in streamlit.session_state:
        streamlit.session_state.checkbox_include_yahoo_news = False
    if 'checkbox_include_filings' not in streamlit.session_state:
        streamlit.session_state.checkbox_include_filings = False
    if 'checkbox_include_pdf_rag' not in streamlit.session_state:
        streamlit.session_state.checkbox_include_pdf_rag = False
    if 'checkbox_generate_from_history' not in streamlit.session_state:
        streamlit.session_state.checkbox_generate_from_history = False

    # Display files horizontally using Streamlit columns
    cols = streamlit.columns(2)

    with cols[0]:
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
    with cols[1]:
        include_pdf_rag = streamlit.checkbox(
            'Include saved PDF report queries',
            key='checkbox_include_pdf_rag',
            value=False,
            on_change=check_inclusions,
        )
        # Add a space
        streamlit.text('\n' * 3)
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

    streamlit.markdown('<h2> Use PDF Report for RAG </h2>', unsafe_allow_html=True)

    # Initialize session state for checkboxes if not already done
    if 'checkbox_use_generated_pdf' not in streamlit.session_state:
        streamlit.session_state.checkbox_use_generated_pdf = False
    if 'checkbox_upload_your_pdf' not in streamlit.session_state:
        streamlit.session_state.checkbox_upload_your_pdf = False

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

    if use_generated_pdf or upload_your_pdf:
        if use_generated_pdf and not upload_your_pdf:
            # Get list of files in the directory
            files = [
                f
                for f in os.listdir(PDF_GENERATION_DIRECTORY)
                if os.path.isfile(os.path.join(PDF_GENERATION_DIRECTORY, f)) and f.endswith('.pdf')
            ]

            # Display files horizontally using Streamlit columns
            cols = streamlit.columns(len(files))

            for idx, file in enumerate(files):
                with cols[idx]:
                    if streamlit.button(
                        file,
                        key=f'file-{idx}',
                        on_click=handle_click_selected_file,
                        args=(file,),
                        disabled=True if file in streamlit.session_state.selected_files else False,
                    ):
                        pass
            streamlit.write(f'Selected files: {', '.join(streamlit.session_state.selected_files)}')

        elif upload_your_pdf and not use_generated_pdf:
            # Add a PDF document for RAG (optional)
            uploaded_results = streamlit.file_uploader(
                'Upload PDF reports for RAG (optional):',
                type='pdf',
                accept_multiple_files=True,
            )
            if isinstance(uploaded_results, UploadedFile):
                uploaded_files = [uploaded_results]
            else:
                uploaded_files = uploaded_results
            if (
                isinstance(uploaded_files, list)
                and len(uploaded_files) > 0
                and all(isinstance(file, UploadedFile) for file in uploaded_files)
                and all(isinstance(file.name, str) for file in uploaded_files)
            ):
                # Store uploaded files
                streamlit.session_state.uploaded_files = [file.name for file in uploaded_files]
                for file in uploaded_files:  #  type: ignore
                    assert isinstance(file, UploadedFile), f'{file} is not instance of UploadedFile.'
                    with open(os.path.join(PDF_GENERATION_DIRECTORY, file.name), 'wb') as f:
                        f.write(file.getbuffer())

        user_request = streamlit.text_input(
            'Enter the info that you want to retrieve for given companies.',
            key='pdf-rag',
        )

        if streamlit.button('Use report for RAG'):
            if use_generated_pdf and not upload_your_pdf:
                # Check file selection
                assert len(streamlit.session_state.selected_files) > 0, 'No file has been selected.'

                # Display the list of selected files
                selected_files = streamlit.session_state.selected_files
                streamlit.write('Selected Files:', selected_files)
                if len(selected_files) > 0:
                    answer = handle_pdf_rag(user_request, selected_files)

                    content = user_request + '\n\n' + answer
                    if streamlit.button(
                        'Save Answer',
                        on_click=save_output_callback,
                        args=(content, PDF_RAG_PATH),
                    ):
                        pass

            elif upload_your_pdf and not use_generated_pdf:
                # Check file upload
                assert len(streamlit.session_state.uploaded_files), 'No file has been uploaded.'

                # Display the list of selected files
                uploaded_file_names = streamlit.session_state.uploaded_files

                if (
                    isinstance(uploaded_file_names, list)
                    and len(uploaded_file_names) > 0
                    and all(isinstance(item, str) for item in uploaded_file_names)
                ):
                    streamlit.write('Uploaded Files:', uploaded_file_names)
                    answer = handle_pdf_rag(user_request, uploaded_file_names)

                    content = user_request + '\n\n' + answer
                    if streamlit.button(
                        'Save Answer',
                        on_click=save_output_callback,
                        args=(content, PDF_RAG_PATH),
                    ):
                        pass


def handle_pdf_generation(
    title_name: str = 'Financial Report',
    report_name: str = 'financial_report',
    data_paths: Dict[str, str] = dict(),
    include_summary: bool = False,
) -> None:
    # Clear the PDF_GENERATION directory
    clear_directory(PDF_GENERATION_DIRECTORY)

    streamlit.session_state.fc = FunctionCalling()

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
    if streamlit.session_state.checkbox_generate_from_history:
        streamlit.session_state.checkbox_include_stock = False
        streamlit.session_state.checkbox_include_database = False
        streamlit.session_state.checkbox_include_yahoo_news = False
        streamlit.session_state.checkbox_include_filings = False
        streamlit.session_state.checkbox_include_pdf_rag = False


def check_inclusions() -> None:
    if (
        streamlit.session_state.checkbox_include_stock
        or streamlit.session_state.checkbox_include_database
        or streamlit.session_state.checkbox_include_yahoo_news
        or streamlit.session_state.checkbox_include_filings
        or streamlit.session_state.checkbox_include_pdf_rag
    ):
        streamlit.session_state.checkbox_generate_from_history = False
    return


def check_use_generated_pdf() -> None:
    if streamlit.session_state.checkbox_use_generated_pdf:
        streamlit.session_state.checkbox_upload_your_pdf = False


def check_upload_your_pdf() -> None:
    if streamlit.session_state.checkbox_upload_your_pdf:
        streamlit.session_state.checkbox_use_generated_pdf = False


# Function to handle button click
def handle_click_selected_file(file: str) -> None:
    streamlit.session_state.selected_files.append(file)
    streamlit.session_state.selected_files = list(set(streamlit.session_state.selected_files))


def handle_pdf_rag(user_question: str, report_names: List[str]) -> Any:
    streamlit.session_state.tools = ['pdf_rag']
    set_fc_llm(streamlit.session_state.tools)

    user_request = (
        'Please use RAG from the provided PDF file to answer the following question: '
        + user_question
        + f'\nReport names: {', '.join(report_names)}.'
    )
    response = handle_userinput(user_question, user_request)
    return response
