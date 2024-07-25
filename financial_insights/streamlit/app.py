import datetime
import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from datetime import date
from io import StringIO
from time import sleep
from typing import Any, Callable, Generator, List, Optional, Tuple, Type, Union

import pandas
import plotly
import plotly.graph_objects as go
import streamlit
import streamlit.components.v1 as components
import yfinance
from langchain.prompts import load_prompt
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import StructuredTool, Tool
from streamlit_extras.stylable_container import stylable_container

from financial_insights.src.function_calling import ConversationalResponse

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from financial_insights.src.function_calling import \
    FunctionCallingLlm  # type: ignore
from financial_insights.src.tools_filings import retrieve_filings
from financial_insights.src.tools_stocks import (get_financial_summary,
                                                 get_historical_price,
                                                 get_stock_info,
                                                 plot_price_over_time,
                                                 retrieve_symbol_list,
                                                 retrieve_symbol_quantity_list)
from financial_insights.src.tools_yahoo_news import scrape_yahoo_finance_news
from financial_insights.streamlit.utilities import (
    clear_directory, handle_financial_filings, handle_financial_summary,
    handle_stock_data_analysis, handle_stock_info, handle_userinput,
    handle_yfinance_news, list_files_in_directory,
    save_dataframe_figure_callback, save_dict_answer_callback,
    save_string_answer_callback, set_fc_llm, st_capture)
from function_calling.src.tools import (calculator, get_time,  # type: ignore
                                        python_repl, query_db, rag, translate)

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
    'retrieve_symbol_list': retrieve_symbol_list,
    'retrieve_symbol_quantity_list': retrieve_symbol_quantity_list,
    'scrape_yahoo_finance_news': scrape_yahoo_finance_news,
    'get_financial_summary': get_financial_summary,
    'retrieve_filings': retrieve_filings,
}

TEMP_DIR = 'financial_insights/streamlit/cache/'

# Inject custom CSS


def main() -> None:
    clear_directory(TEMP_DIR + 'sources')
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
        background-color: green;
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

    with streamlit.sidebar:
        # Navigation menu
        streamlit.title('Navigation')
        menu = streamlit.radio(
            'Go to',
            [
                'Home',
                'Stock Data Analysis',
                'Financial News Scraping',
                'Financial Filings Analysis',
                'Custom Queries',
                'Generate PDF Report',
            ],
        )

        streamlit.title('Saved Files')

        files = list_files_in_directory(TEMP_DIR)

        # Custom button to clear all files
        with stylable_container(
            key='orange-button',
            css_styles="""
            button {
                background-color: blue;
                color: black;
                padding: 0.75em 1.5em;
                font-size: 1;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }""",
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
                        streamlit.sidebar.download_button(label=f'{file}', data=file_content, file_name=file, mime='text/plain')
                    except Exception as e:
                        logging.warning('Error reading file', str(e))
                    except FileNotFoundError as e:
                        logging.warning('File not found', str(e))
        else:
            streamlit.write('No files found')

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
            Welcome to the Financial Insights application.
            This app demonstrates the capabilities of large language models (LLMs)
            in extracting and analyzing financial data using function calling, web scraping,
            and retrieval-augmented generation (RAG).
            
            Use the navigation menu to explore various features including:
            - Financial Filings Analysis
            - Stock Data Analysis
            - Financial News Scraping
            - Custom Queries
            - Generate PDF Report
        """)

    # Stock Data Analysis page
    elif menu == 'Stock Data Analysis':
        streamlit.markdown('<h2> Stock Data Analysis </h2>', unsafe_allow_html=True)
        streamlit.markdown(
            '<a href="https://pypi.org/project/yfinance/" target="_blank" '
            'style="color:cornflowerblue;text-decoration:underline;"><h3>via Yahoo! Finance API</h3></a>',
            unsafe_allow_html=True,
        )
        streamlit.markdown('<h3> Info retrieval </h3>', unsafe_allow_html=True)

        output = streamlit.empty()  # type: ignore

        user_request = streamlit.text_input(
            'Enter the info that you want to retrieve for given companies', key='ticker_symbol'
        )
        if streamlit.button('Retrieve stock info'):
            with streamlit.expander('**Execution scratchpad**', expanded=True):
                response_string = handle_stock_info(user_request)

                save_path = 'stock_info.txt'
                content = user_request + '\n\n' + response_string + '\n\n\n'
                if streamlit.button(
                    'Save Answer', on_click=save_string_answer_callback, args=(content, save_path)
                ):
                    pass

        if streamlit.button(label='Get financial summary'):
            with streamlit.expander('**Execution scratchpad**', expanded=True):
                response_dict = handle_financial_summary(user_request)
                save_path = 'summary_' + '_'.join(list(response_dict.keys()))
                if streamlit.button(
                    'Save Analysis', on_click=save_dict_answer_callback, args=(response_dict, save_path)
                ):
                    pass

        streamlit.markdown('<br><br>', unsafe_allow_html=True)
        streamlit.markdown('<h3> Stock data analysis </h3>', unsafe_allow_html=True)
        output = streamlit.empty()  # type: ignore
        ticker_list = streamlit.text_input(
            'Enter the quantities that you want to plot for given companies\n'
            'Suggested values: Open, High, Low, Close, Volume, Dividends, Stock Splits.'
        )
        start_date = streamlit.date_input('Start Date')
        end_date = streamlit.date_input('End Date')

        # Analyze stock data
        if streamlit.button('Analyze Stock Data'):
            with streamlit.expander('**Execution scratchpad**', expanded=True):
                data, fig = handle_stock_data_analysis(ticker_list, start_date, end_date)  # type: ignore

                if streamlit.button(
                    'Save Analysis', on_click=save_dataframe_figure_callback, args=(ticker_list, data, fig)
                ):
                    pass

    # Stock Data Analysis page
    elif menu == 'Financial News Scraping':
        streamlit.markdown('<h2> Financial news scraping </h2>', unsafe_allow_html=True)
        streamlit.markdown(
            '<a href="https://uk.finance.yahoo.com/" target="_blank" '
            'style="color:cornflowerblue;text-decoration:underline;"><h3>via Yahoo! Finance News</h3></a>',
            unsafe_allow_html=True,
        )
        output = streamlit.empty()  # type: ignore

        user_request = streamlit.text_input(
            'Enter the yfinance news that you want to retrieve for given companies', key='yahoo_news'
        )

        # Retrieve news
        if streamlit.button('Retrieve news'):
            with streamlit.expander('**Execution scratchpad**', expanded=True):
                if user_request is not None:
                    answer, url_list = handle_yfinance_news(user_request)
                else:
                    raise ValueError('No input provided')

            if answer is not None:
                content = user_request + '\n\n' + answer + '\n\n' + '\n'.join(url_list) + '\n\n\n'
                if streamlit.button(
                    'Save Answer', on_click=save_string_answer_callback, args=(content, 'yfinance_news.txt')
                ):
                    pass

    # Financial Filings Analysis page
    elif menu == 'Financial Filings Analysis':
        streamlit.markdown('<h2> Financial Filings Analysis </h2>', unsafe_allow_html=True)
        streamlit.markdown(
            '<a href="https://www.sec.gov/edgar/search/" target="_blank" '
            'style="color:cornflowerblue;text-decoration:underline;"><h3>via SEC EDGAR</h3></a>',
            unsafe_allow_html=True,
        )
        user_request = streamlit.text_input(
            'Enter your query:', key='stock_info'
        )
        company_name = streamlit.text_input('Company name (optional if in the query already)')
        # Define the range of years
        start_year = 2020
        end_year = 2024
        years = list(range(start_year, end_year + 1))
        # Set the default year (e.g., 2023)
        default_year = 2023
        default_index = years.index(default_year)
        # Create the selectbox with the default year
        selected_year = streamlit.selectbox('Select a year:', years, index=default_index)
        
        filing_type = streamlit.selectbox('Select Filing Type:', ['10-K', '10-Q'], index=0)
        if filing_type == '10-Q':
            filing_quarter = streamlit.selectbox('Select Quarter:', [1, 2, 3, 4])
        else:
            filing_quarter = None
        if streamlit.button('Analyze Filing'):
            with streamlit.expander('**Execution scratchpad**', expanded=True):
                answer, filename = handle_financial_filings(user_request, company_name, filing_type, filing_quarter, selected_year)
                
                content = user_request + '\n\n' + answer + '\n\n\n'
                if streamlit.button(
                    'Save Answer', on_click=save_string_answer_callback, args=(content, filename + '.txt')
                ):
                    pass

    # Custom Queries page
    elif menu == 'Custom Queries':
        streamlit.markdown('<h2> Custom Queries </h2>', unsafe_allow_html=True)
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
                pass
                # documents = []

                # Handle data source selection
                if data_source == 'yfinance':
                    pass
                    # Example function to retrieve data from yfinance

                #     documents = retrieve_documents(query)
                # else:
                #     # Example function to retrieve data from SEC EDGAR
                #     documents = scrape_sec_filings(query, '10-K')  # Adjust as needed

                # # Handle PDF document for RAG
                # if pdf_file is not None:
                #     documents.extend(retrieve_from_pdf(pdf_file))

                # # Handle additional web scraping
                # if webscrape_url:
                #     additional_docs = scrape_yahoo_news(webscrape_url)  # Replace with appropriate function
                #     documents.extend(additional_docs)

                # # Handle custom database CSV file
                # if csv_file is not None:
                #     import pandas as pd

                #     csv_data = pd.read_csv(csv_file)
                #     documents.extend(csv_data.to_dict(orient='records'))  # Adjust processing as needed

                # streamlit.session_state.tools = streamlit.multiselect(
                #     'Available tools',
                #     ['get_time', 'calculator', 'python_repl', 'query_db', 'translate', 'rag'],
                # )
                # streamlit.session_state.tools = ['get_stock_info', 'get_historical_price']
                # set_fc_llm(streamlit.session_state.tools)
                # handle_userinput(query)
                # documents = retrieve_documents(query)
                # response = process_documents(documents)
                # streamlit.write(response)

    # Generate PDF Report page
    elif menu == 'Generate PDF Report':
        streamlit.markdown('<h2> Generate PDF Report </h2>', unsafe_allow_html=True)
        include_stock_data = streamlit.checkbox('Include Stock Data')
        inlude_yahoo_news = streamlit.checkbox('Include Yahoo News')
        include_filings = streamlit.checkbox('Include Financial Filings')
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
            # generate_pdf_report(data)
            streamlit.write('PDF report generated successfully.')


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
