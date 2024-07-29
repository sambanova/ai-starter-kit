import os
import sys

import streamlit

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)


def get_custom_queries() -> None:
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
        output = streamlit.empty()

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
