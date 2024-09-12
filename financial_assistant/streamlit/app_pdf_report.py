import os
import shutil
from base64 import b64encode
from typing import Any, Dict, List, Optional

import streamlit
from streamlit.runtime.uploaded_file_manager import UploadedFile

from financial_assistant.src.llm import SambaNovaLLM
from financial_assistant.src.tools import get_logger
from financial_assistant.src.tools_pdf_generation import generate_pdf, parse_documents, read_txt_files
from financial_assistant.streamlit.constants import *
from financial_assistant.streamlit.utilities_app import clear_directory, save_output_callback
from financial_assistant.streamlit.utilities_methods import attach_tools, handle_userinput

logger = get_logger()


def include_pdf_report() -> None:
    """Include the app for the generation and the usage of the PDF report."""

    # Initialize session state for the selected uploaded files
    if 'selected_files' not in streamlit.session_state:
        streamlit.session_state.selected_files = []
    if 'uploaded_files' not in streamlit.session_state:
        streamlit.session_state.uploaded_files = []

    # PDF generation
    streamlit.markdown('<h2> Generate PDF Report </h2>', unsafe_allow_html=True)

    # Display files horizontally using Streamlit columns
    cols = streamlit.columns(2)

    with cols[0]:
        include_stock = streamlit.checkbox(
            'Include saved stock queries',
            key='checkbox_include_stock',
            value=True,
            on_change=check_inclusions,
        )
        include_database = streamlit.checkbox(
            'Include saved stock database queries',
            key='checkbox_include_database',
            value=True,
            on_change=check_inclusions,
        )
        inlude_yahoo_news = streamlit.checkbox(
            'Include saved Yahoo News queries',
            key='checkbox_include_yahoo_news',
            value=True,
            on_change=check_inclusions,
        )
        include_filings = streamlit.checkbox(
            'Include saved financial filings queries',
            key='checkbox_include_filings',
            value=True,
            on_change=check_inclusions,
        )

    with cols[1]:
        include_pdf_rag = streamlit.checkbox(
            'Include saved PDF report queries',
            key='checkbox_include_pdf_rag',
            value=False,
            on_change=check_inclusions,
        )
        # Add a vertical space
        streamlit.text('\n' * 3)
        generate_from_history = streamlit.checkbox(
            'Generate from the whole chat history',
            key='checkbox_generate_from_history',
            value=False,
            on_change=check_generate_from_history,
        )

    # Initialize session state for checkboxes if not already done
    if 'checkbox_include_stock' not in streamlit.session_state:
        streamlit.session_state.checkbox_include_stock = include_stock
    if 'checkbox_include_database' not in streamlit.session_state:
        streamlit.session_state.checkbox_include_database = include_database
    if 'checkbox_include_yahoo_news' not in streamlit.session_state:
        streamlit.session_state.checkbox_include_yahoo_news = inlude_yahoo_news
    if 'checkbox_include_filings' not in streamlit.session_state:
        streamlit.session_state.checkbox_include_filings = include_filings
    if 'checkbox_include_pdf_rag' not in streamlit.session_state:
        streamlit.session_state.checkbox_include_pdf_rag = include_pdf_rag
    if 'checkbox_generate_from_history' not in streamlit.session_state:
        streamlit.session_state.checkbox_generate_from_history = generate_from_history

    data_paths = dict()
    if include_stock:
        # Add data from Stock Data Analysis
        data_paths['stock_query'] = streamlit.session_state.stock_query_path
    if include_database:
        # Add data from Stock Database Analysis
        data_paths['stock_database'] = streamlit.session_state.db_query_path
    if inlude_yahoo_news:
        # Add data from Yahoo News Analysis
        data_paths['yfinance_news'] = streamlit.session_state.yfinance_news_path
    if include_filings:
        # Add data from Financial Filings Analysis
        data_paths['filings'] = streamlit.session_state.filings_path
    if generate_from_history:
        # Deselect all other options
        include_stock = False
        include_database = False
        inlude_yahoo_news = False
        include_filings = False

        # Add data from chat history
        data_paths['history'] = streamlit.session_state.history_path

    # Add title name (optional)
    title_name = streamlit.text_input(label='Title Name', value='Financial Report')

    # Include summary for each section
    include_summary = streamlit.checkbox(
        'Include summary from each section',
        key='checkbox_summary',
        value=True,
        help='This will take longer!',
    )
    if include_summary:
        streamlit.write(r':red[Warning: This will take longer!]')

    # Generate the report
    if streamlit.button('Generate Report'):
        with streamlit.expander('**Execution scratchpad**', expanded=True):
            report_name = title_name.lower().replace(' ', '_') + '.pdf'

            # generate_pdf_report(data)
            pdf_handler = handle_pdf_generation(title_name, report_name, data_paths, include_summary)

            # Embed PDF to display it:
            if pdf_handler is not None:
                base64_pdf = b64encode(pdf_handler).decode('utf-8')
                pdf_display = (
                    f'<embed src="data:application/pdf;base64,{base64_pdf}"'
                    ' width="700" height="400" type="application/pdf">'
                )
                streamlit.markdown(pdf_display, unsafe_allow_html=True)
                # Add download button
                streamlit.download_button(
                    label='Download Report',
                    data=pdf_handler,
                    file_name=report_name,
                    mime='application/pdf',
                )
                streamlit.write('PDF report generated successfully.')

    # Use PDF report for RAG
    streamlit.markdown('<h2> Use PDF Report for RAG </h2>', unsafe_allow_html=True)

    # Use the previously generated PDF reports
    use_generated_pdf = streamlit.checkbox(
        'Use generated PDF',
        key='checkbox_use_generated_pdf',
        value=True,
        on_change=check_use_generated_pdf,
    )

    # Upload your own PDF reports
    upload_your_pdf = streamlit.checkbox(
        'Upload your PDF',
        key='checkbox_upload_your_pdf',
        value=False,
        on_change=check_upload_your_pdf,
    )

    # Initialize session state for checkboxes if not already done
    if 'checkbox_use_generated_pdf' not in streamlit.session_state:
        streamlit.session_state.checkbox_use_generated_pdf = use_generated_pdf
    if 'checkbox_upload_your_pdf' not in streamlit.session_state:
        streamlit.session_state.checkbox_upload_your_pdf = upload_your_pdf

    if use_generated_pdf or upload_your_pdf:
        if use_generated_pdf and not upload_your_pdf:
            # Get list of files in the directory
            files = [
                f
                for f in os.listdir(streamlit.session_state.pdf_generation_directory)
                if os.path.isfile(os.path.join(streamlit.session_state.pdf_generation_directory, f))
                and f.endswith('.pdf')
            ]

            if len(files) > 0:
                # Display files horizontally using Streamlit columns
                cols = streamlit.columns(len(files))

                for idx, file in enumerate(files):
                    with cols[idx]:
                        if streamlit.button(
                            label=f':green[{file}]' if file in streamlit.session_state.selected_files else file,
                            key=f'file-{idx}',
                            type='primary',
                            on_click=handle_click_selected_file,
                            args=(file,),
                        ):
                            pass
                streamlit.write(f"Selected files: :green[{', '.join(streamlit.session_state.selected_files)}]")

        elif upload_your_pdf and not use_generated_pdf:
            # Add a PDF document for RAG
            uploaded_results = streamlit.file_uploader(
                'Upload PDF reports for RAG:',
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
                for file in uploaded_files:
                    if not isinstance(file, UploadedFile):
                        streamlit.error(f'{file} is not instance of UploadedFile.')
                    with open(os.path.join(streamlit.session_state.pdf_generation_directory, file.name), 'wb') as f:
                        f.write(file.getbuffer())

        # The user request
        user_request = streamlit.text_input(
            label=f'Ask a question about your financial report. :sparkles: :violet[{DEFAULT_PDF_RAG_QUERY}]',
            key='pdf-rag',
            placeholder='E.g. ' + DEFAULT_PDF_RAG_QUERY,
        )

        # Use PDF reports for RAG
        if streamlit.button('Use report for RAG'):
            if len(user_request) == 0:
                streamlit.error('Please enter your query.')
            else:
                if use_generated_pdf and not upload_your_pdf:
                    # Check file selection
                    if len(streamlit.session_state.selected_files) == 0:
                        streamlit.error('No file has been selected.')

                    # Retrieve the list of selected files
                    selected_files = streamlit.session_state.selected_files

                    # Display the list of selected files
                    streamlit.write('Selected Files:', selected_files)
                    if len(selected_files) > 0:
                        # Retrieve the answer
                        answer = handle_pdf_rag(user_request, selected_files)

                        # Compose the query and answer string
                        content = user_request + '\n\n' + answer

                        # Save the query and answer to the PDF RAG text file
                        if streamlit.button(
                            'Save Answer',
                            on_click=save_output_callback,
                            args=(content, streamlit.session_state.pdf_rag_path),
                        ):
                            pass

                elif upload_your_pdf and not use_generated_pdf:
                    # Check file upload
                    if len(streamlit.session_state.uploaded_files) == 0:
                        streamlit.error('No file has been uploaded.')

                    # Retrieve the list of uploaded files
                    uploaded_file_names = streamlit.session_state.uploaded_files

                    if (
                        isinstance(uploaded_file_names, list)
                        and len(uploaded_file_names) > 0
                        and all(isinstance(item, str) for item in uploaded_file_names)
                    ):
                        # Display the list of uploaded files
                        streamlit.write('Uploaded Files:', uploaded_file_names)

                        # Retrieve the answer
                        answer = handle_pdf_rag(user_request, uploaded_file_names)

                        # Compose the query and answer string
                        content = user_request + '\n\n' + answer

                        # Save the query and answer to the PDF RAG text file
                        if streamlit.button(
                            'Save Answer',
                            on_click=save_output_callback,
                            args=(content, streamlit.session_state.pdf_rag_path),
                        ):
                            pass


def handle_pdf_generation(
    title_name: str = 'Financial Report',
    report_name: str = 'financial_report',
    data_paths: Dict[str, str] = dict(),
    include_summary: bool = False,
) -> Optional[bytes]:
    """
    Generate a PDF report using the provided data paths.

    Args:
        title_name: The title of the report.
            Default is `Financial Report`.
        report_name: The name of the report file.
            Default is `financial_report`.
        data_paths: A dictionary of data paths to be used for the PDF generation.
        include_summary: Whether to include a summary for each section,
            and an abstract and a general summary at the beginning of the document.
            Default is False.

    Raises:
        Exception: If there are no values for selected or uploaded sources in the `data_paths` dictionary.
        Exception: If there are no file in `data_paths` from which to generate a PDF.
    """
    # Initialize the function calling object
    streamlit.session_state.llm = SambaNovaLLM()

    # Clean the sources directory if it exists
    if os.path.exists(streamlit.session_state.pdf_sources_directory):
        clear_directory(streamlit.session_state.pdf_sources_directory)

    # Derive the output file name
    output_file = streamlit.session_state.pdf_generation_directory + report_name

    # Check that at least one data source is available
    if not any([data_paths[key] for key in data_paths]):
        streamlit.error('Select at least one data source.')
        return None

    # Assert that at least one data source exists as a file
    if not any([os.path.isfile(data_paths[key]) for key in data_paths]):
        streamlit.error('No data source available.')
        return None

    for source_file in data_paths.values():
        # Create the full path for the destination file
        destination_file = os.path.join(streamlit.session_state.pdf_sources_directory, os.path.basename(source_file))

        try:
            # Copy selected document to the pdf generation directory
            shutil.copy(source_file, destination_file)

            logger.info(f'{source_file} has been copied to {destination_file}')
        except Exception as e:
            logger.error('Error while copying file', exc_info=True)

    # Extract the documents from the selected files
    documents = read_txt_files(streamlit.session_state.pdf_sources_directory)

    # Parse the documents into a list of tuples of text and figure paths
    report_content = parse_documents(documents)

    # Generate the PDF report
    pdf_handler = generate_pdf(report_content, output_file, title_name, include_summary)

    return pdf_handler


def check_generate_from_history() -> None:
    """Check the mutually exclusive checkboxes and reset the default values."""
    if streamlit.session_state.checkbox_generate_from_history:
        streamlit.session_state.checkbox_include_stock = False
        streamlit.session_state.checkbox_include_database = False
        streamlit.session_state.checkbox_include_yahoo_news = False
        streamlit.session_state.checkbox_include_filings = False
        streamlit.session_state.checkbox_include_pdf_rag = False


def check_inclusions() -> None:
    """Check the mutually exclusive checkboxes and reset the default values."""
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
    """Check the mutually exclusive checkboxes and reset the default values."""
    if streamlit.session_state.checkbox_use_generated_pdf:
        streamlit.session_state.checkbox_upload_your_pdf = False


def check_upload_your_pdf() -> None:
    """Check the mutually exclusive checkboxes and reset the default values."""
    if streamlit.session_state.checkbox_upload_your_pdf:
        streamlit.session_state.checkbox_use_generated_pdf = False


# Function to handle button click
def handle_click_selected_file(file: str) -> None:
    """Add selected file to the list of selected files."""

    if file not in streamlit.session_state.selected_files:
        # Add file to the list of selected files
        streamlit.session_state.selected_files.append(file)
    else:
        # Remove file from the list of selected files
        streamlit.session_state.selected_files = [x for x in streamlit.session_state.selected_files if x != file]
    streamlit.session_state.selected_files = list(set(streamlit.session_state.selected_files))


def handle_pdf_rag(user_question: str, report_names: List[str]) -> Any:
    """
    Handle the user request for PDF RAG.

    Args:
        user_question: The user's question to be answered.
        report_names: The list of report names.

    Returns:
        The LLM response using RAG from the selected or uploaded files.

    Raises;
        TypeError: If the LLM response does not conform to the return type.
    """
    # Declare the permitted tools for function calling
    streamlit.session_state.tools = ['pdf_rag']

    # Attach the tools for the LLM to use
    attach_tools(streamlit.session_state.tools)

    # Compose the user request
    user_request = f"""
        Please use RAG (Retrieval-Augmented Generation) from the provided PDF files to answer the following question:
        {user_question}

        Report names: {', '.join(report_names)}.
    """

    # Call the LLM on the user request with the attached tools
    response = handle_userinput(user_question, user_request)

    # Check the final answer of the LLM
    assert isinstance(response, str), TypeError(f'Invalid response: {response}.')

    return response
