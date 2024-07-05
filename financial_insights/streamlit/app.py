import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
from typing import Callable, Generator, Optional
import yfinance
import streamlit
from datetime import date
import pandas
import plotly.graph_objects as go
from typing import List
import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from financial_insights.src.function_calling import FunctionCallingLlm  # type: ignore
from function_calling.src.tools import calculator, get_time, python_repl, query_db, rag, translate  # type: ignore
from financial_insights.src.tools import get_stock_info, get_historical_price, plot_price_over_time

logging.basicConfig(level=logging.INFO)

# tool mapping of available tools
TOOLS = {
    'get_time': get_time,
    'calculator': calculator,
    'python_repl': python_repl,
    'query_db': query_db,
    'translate': translate,
    'rag': rag,
    'get_stock_info': get_stock_info,
    'get_historical_price': get_historical_price,
}


@contextmanager
def st_capture(output_func: Callable[[str], None]) -> Generator:
    """
    context manager to catch stdout and send it to an output streamlit element

    Args:
        output_func (function to write terminal output in

    Yields:
        Generator:
    """
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string: str) -> int:
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write  # type: ignore
        yield


def set_fc_llm(tools: list) -> None:
    """
    Set the FunctionCallingLlm object with the selected tools

    Args:
        tools (list): list of tools to be used
    """
    set_tools = [TOOLS[name] for name in tools]
    streamlit.session_state.fc = FunctionCallingLlm(set_tools)


def handle_userinput(user_question: Optional[str]) -> None:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        user_question (str): The user's question or input.
    """
    global output
    if user_question:
        with streamlit.spinner('Processing...'):
            with st_capture(output.code):  # type: ignore
                response = streamlit.session_state.fc.function_call_llm(
                    query=user_question, max_it=streamlit.session_state.max_iterations, debug=True
                )

        streamlit.session_state.chat_history.append(user_question)
        streamlit.session_state.chat_history.append(response)

    for ques, ans in zip(
        streamlit.session_state.chat_history[::2],
        streamlit.session_state.chat_history[1::2],
    ):
        with streamlit.chat_message('user'):
            streamlit.write(f'{ques}')

        with streamlit.chat_message(
            'ai',
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            streamlit.write(f'{ans}')


from typing import Any


def handle_stock_data_analysis(ticker_list: str, start_date: datetime.date, end_date: datetime.date) -> None:
    """
    Handle user input and generate a response, also update chat UI in streamlit app

    Args:
        symbol (str): The user's question or input.
    """
    symbol_list = ticker_list.split(',')
    global output
    if len(symbol_list) > 0:
        with streamlit.spinner('Processing...'):
            with st_capture(output.code):  # type: ignore
                response = get_historical_price.invoke(
                    input={'symbol_list': symbol_list, 'start_date': start_date, 'end_date': end_date}
                )

        streamlit.session_state.chat_history.append(symbol_list)
        streamlit.session_state.chat_history.append(response)

    for ques, ans in zip(
        streamlit.session_state.chat_history[::2],
        streamlit.session_state.chat_history[1::2],
    ):
        with streamlit.chat_message('user'):
            streamlit.write(f'{ques}')

        with streamlit.chat_message(
            'ai',
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            streamlit.write(f'{ans}')
    return None


def main() -> None:
    global output

    # Streamlit app setup
    streamlit.set_page_config(
        page_title='AI Starter Kit', page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png', layout='wide'
    )

    streamlit.markdown(
        """
    <style>
    /* General body styling */

    html, body {
        font-size: 1,
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #e0e0e0;
        background-color: #1e1e1e;
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        margin-bottom: 1em;
    }

    /* Paragraph and text styling */
    p, label {
        font-size: 1;
        line-height: 1.6;
        margin-bottom: 0.5em;
        color: #e0e0e0;
    }

    /* Button styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 0.75em 1.5em;
        font-size: 1;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }

    /* Radio button styling */
    .stRadio > label {
        font-size: 1;
    }
    .stRadio > div > div > label {
        font-size: 1;
        padding: 0.25em 0.75em;
        cursor: pointer;
        color: #e0e0e0;
    }
    .stRadio > div > div {
        margin-bottom: 0.5em;
    }

    /* Input field styling */
    input[type="text"], input[type="date"], select {
        width: 100%;
        padding: 0.75em;
        margin: 0.5em 0 1em 0;
        display: inline-block;
        border: 1px solid #ccc;
        border-radius: 4px;
        box-sizing: border-box;
        font-size: 1.1em;
        background-color: #2c2c2c;
        color: #e0e0e0;
    }

    /* Checkbox styling */
    .stCheckbox > label {
        font-size: 1.1em;
    }

    /* Container styling */
    .main {
        padding: 2em;
        background: #2c2c2c;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 2em;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e1e1e;
    }
    .css-1d391kg .css-1v3fvcr, .css-1d391kg .css-1l5dyp6 {
        color: #e0e0e0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Navigation menu
    streamlit.sidebar.title('Navigation')
    menu = streamlit.sidebar.radio(
        'Go to', ['Home', 'Financial Filings Analysis', 'Stock Data Analysis', 'Custom Queries', 'Generate PDF Report']
    )

    if 'fc' not in streamlit.session_state:
        streamlit.session_state.fc = None
    if 'chat_history' not in streamlit.session_state:
        streamlit.session_state.chat_history = []
    if 'tools' not in streamlit.session_state:
        streamlit.session_state.tools = ['get_time', 'python_repl', 'query_db']
    if 'max_iterations' not in streamlit.session_state:
        streamlit.session_state.max_iterations = 5

    streamlit.title(':orange[SambaNova] Financial Insights Assistant')

    # Home page
    if menu == 'Home':
        streamlit.title('Financial Insights with LLMs')
        streamlit.write("""
            Welcome to the Financial Insights application. This app demonstrates the capabilities of large language models (LLMs) in extracting and analyzing financial data using function calling, web scraping, and retrieval-augmented generation (RAG).
            
            Use the navigation menu to explore various features including:
            - Financial Filings Analysis
            - Stock Data Analysis
            - Custom Queries
            - Generate PDF Report
        """)

    # Financial Filings Analysis page
    elif menu == 'Financial Filings Analysis':
        with streamlit.expander('**Execution scratchpad**', expanded=True):
            output = streamlit.empty()  # type: ignore
            streamlit.title('Financial Filings Analysis')
            ticker = streamlit.text_input('Enter Ticker Symbols, separated by commas:')
            filing_type = streamlit.selectbox('Select Filing Type:', ['10-K', '10-Q'])
            if streamlit.button('Analyze Filings'):
                filings = scrape_sec_filings(ticker, filing_type)
                summarized_filings = process_documents(filings)
                streamlit.write(summarized_filings)

    # Stock Data Analysis page
    elif menu == 'Stock Data Analysis':
        output = streamlit.empty()  # type: ignore
        streamlit.title('Stock Data Analysis')
        ticker_list = streamlit.text_input('Enter Ticker Symbols, separated by commas:')
        start_date = streamlit.date_input('Start Date')
        end_date = streamlit.date_input('End Date')
        if streamlit.button('Analyze Stock Data'):
            set_fc_llm(streamlit.session_state.tools)
            data = handle_stock_data_analysis(ticker_list, start_date, end_date)

            # streamlit.line_chart(stock_data['Close'])
            # Add more stock analysis as needed

    # Custom Queries page
    elif menu == 'Custom Queries':
        streamlit.title('Custom Queries')

        # Container for the entire section
        with streamlit.container():
            streamlit.header('Data Source Selection')
            data_source = streamlit.radio('Select Data Source:', ['yfinance', 'SEC EDGAR'])

        # Container for optional sections
        with streamlit.container():
            streamlit.header('Optional Additions')

            # Add a PDF document for RAG (optional)
            pdf_file = streamlit.file_uploader('Upload a PDF document for RAG (optional):', type='pdf')

            # Select another website for web scraping (optional)
            webscrape_url = streamlit.text_input('Enter another website URL for web scraping (optional):')

            # Add another custom database as a CSV file (optional)
            csv_file = streamlit.file_uploader('Upload a CSV file for additional database (optional):', type='csv')

        # Query input section
        streamlit.header('Query Input')

        streamlit.markdown('**Set the maximum number of iterations your want the model to run**')
        streamlit.session_state.max_iterations = streamlit.number_input('Max iterations', value=5, max_value=20)
        streamlit.markdown('**Note:** The response cannot completed if the max number of iterations is too low')

        query = streamlit.text_area("Enter your query related to a company's financials:")

        with streamlit.expander('**Execution scratchpad**', expanded=True):
            output = streamlit.empty()  # type: ignore

            if streamlit.button('Submit Query'):
                documents = []

                # Handle data source selection
                if data_source == 'yfinance':
                    # Example function to retrieve data from yfinance

                    documents = retrieve_documents(query)
                else:
                    # Example function to retrieve data from SEC EDGAR
                    documents = scrape_sec_filings(query, '10-K')  # Adjust as needed

                # Handle PDF document for RAG
                if pdf_file is not None:
                    documents.extend(retrieve_from_pdf(pdf_file))

                # Handle additional web scraping
                if webscrape_url:
                    additional_docs = scrape_yahoo_news(webscrape_url)  # Replace with appropriate function
                    documents.extend(additional_docs)

                # Handle custom database CSV file
                if csv_file is not None:
                    import pandas as pd

                    csv_data = pd.read_csv(csv_file)
                    documents.extend(csv_data.to_dict(orient='records'))  # Adjust processing as needed

                streamlit.session_state.tools = streamlit.multiselect(
                    'Available tools',
                    ['get_time', 'calculator', 'python_repl', 'query_db', 'translate', 'rag'],
                    ['get_time', 'python_repl', 'query_db'],
                )
                set_fc_llm(streamlit.session_state.tools)
                handle_userinput(query)
                documents = retrieve_documents(query)
                response = process_documents(documents)
                streamlit.write(response)

    # Generate PDF Report page
    elif menu == 'Generate PDF Report':
        streamlit.title('Generate PDF Report')
        include_filings = streamlit.checkbox('Include Financial Filings Analysis')
        include_stock_data = streamlit.checkbox('Include Stock Data Analysis')
        include_custom_queries = streamlit.checkbox('Include Custom Queries')
        if streamlit.button('Generate Report'):
            data = []
            if include_filings:
                # Add data from Financial Filings Analysis
                data.append('Financial Filings Analysis Data')
            if include_stock_data:
                # Add data from Stock Data Analysis
                data.append('Stock Data Analysis Data')
            if include_custom_queries:
                # Add data from Custom Queries
                data.append('Custom Queries Data')
            generate_pdf_report(data)
            streamlit.write('PDF report generated successfully.')

    # Additional Streamlit app code...


# user_question = streamlit.chat_input('Ask something')

# with streamlit.sidebar:
#     streamlit.title('Setup')
#     streamlit.markdown('**1. Select the tools for function calling.**')
#     streamlit.session_state.tools = streamlit.multiselect(
#         'Available tools',
#         ['get_time', 'calculator', 'python_repl', 'query_db', 'translate', 'rag'],
#         ['get_time', 'python_repl', 'query_db'],
#     )
#     streamlit.markdown('**2. Set the maximum number of iterations your want the model to run**')
#     streamlit.session_state.max_iterations = streamlit.number_input('Max iterations', value=5, max_value=20)
#     streamlit.markdown('**Note:** The response cannot completed if the max number of iterations is too low')
#     if streamlit.button('Set'):
#         with streamlit.spinner('Processing'):
#             set_fc_llm(streamlit.session_state.tools)
#             streamlit.toast(f'Tool calling assistant set! Go ahead and ask some questions', icon='🎉')

#     streamlit.markdown('**3. Ask questions about your data!**')

#     with streamlit.expander('**Execution scratchpad**', expanded=True):
#         output = streamlit.empty()  # type: ignore

#     with streamlit.expander('Additional settings', expanded=False):
#         streamlit.markdown('**Interaction options**')

#         streamlit.markdown('**Reset chat**')
#         streamlit.markdown('**Note:** Resetting the chat will clear all interactions history')
#         if streamlit.button('Reset conversation'):
#             streamlit.session_state.chat_history = []
#             streamlit.session_state.sources_history = []
#             streamlit.toast('Interactions reset. The next response will clear the history on the screen')

# handle_userinput(user_question)


if __name__ == '__main__':
    main()
