import logging
import os
import sys

import streamlit
from streamlit_extras.stylable_container import stylable_container

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

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
from financial_insights.streamlit.app_yfinance_news import get_yfinance_news
from financial_insights.streamlit.utilities_app import (
    clear_directory, list_files_in_directory)
from financial_insights.streamlit.utilities_methods import stream_chat_history
from function_calling.src.tools import (calculator, get_time, python_repl,
                                        query_db, rag, translate)

logging.basicConfig(level=logging.INFO)

# tool mapping of available tools
TOOLS = {
    "get_time": get_time,
    "calculator": calculator,
    "python_repl": python_repl,
    "query_db": query_db,
    "translate": translate,
    "rag": rag,
    "get_stock_info": get_stock_info,
    "get_historical_price": get_historical_price,
    "retrieve_symbol_list": retrieve_symbol_list,
    "retrieve_symbol_quantity_list": retrieve_symbol_quantity_list,
    "scrape_yahoo_finance_news": scrape_yahoo_finance_news,
    "get_financial_summary": get_financial_summary,
    "retrieve_filings": retrieve_filings,
}

TEMP_DIR = "financial_insights/streamlit/cache/"


def main() -> None:
    clear_directory(TEMP_DIR + "sources")
    global output

    # Streamlit app setup
    streamlit.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
        layout="wide",
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
        streamlit.title("Navigation")
        menu = streamlit.radio(
            "Go to",
            [
                "Home",
                "Stock Data Analysis",
                "Financial News Scraping",
                "Financial Filings Analysis",
                "Custom Queries",
                "Generate PDF Report",
                "Print Chat History",
            ],
        )

        streamlit.title("Saved Files")

        files = list_files_in_directory(TEMP_DIR)

        # Custom button to clear all files
        with stylable_container(
            key="blue-button",
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
                label="Clear All Files",
                key="clear-button",
                help="This will delete all saved files",
            ):
                clear_directory(TEMP_DIR)
                streamlit.sidebar.success("All files have been deleted.")

        if files:
            for file in files:
                file_path = os.path.join(TEMP_DIR, file)
                with open(file_path, "r") as f:
                    try:
                        file_content = f.read()
                        streamlit.sidebar.download_button(
                            label=f"{file}",
                            data=file_content,
                            file_name=file,
                            mime="text/plain",
                        )
                    except Exception as e:
                        logging.warning("Error reading file", str(e))
                    except FileNotFoundError as e:
                        logging.warning("File not found", str(e))
        else:
            streamlit.write("No files found")

    if "fc" not in streamlit.session_state:
        streamlit.session_state.fc = None
    if "chat_history" not in streamlit.session_state:
        streamlit.session_state.chat_history = list()
    if "tools" not in streamlit.session_state:
        streamlit.session_state.tools = ["get_time", "python_repl", "query_db"]
    if "max_iterations" not in streamlit.session_state:
        streamlit.session_state.max_iterations = 5

    streamlit.title(":orange[SambaNova] Financial Insights Assistant")

    # Home page
    if menu == "Home":
        streamlit.title("Financial Insights with LLMs")
        streamlit.write(
            """
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
            - Print chat history
        """
        )

    # Stock Data Analysis page
    elif menu == "Stock Data Analysis":
        get_stock_data_analysis()

    # Financial News Scraping page
    elif menu == "Financial News Scraping":
        get_yfinance_news()

    # Financial Filings Analysis page
    elif menu == "Financial Filings Analysis":
        get_financial_filings()

    # Custom Queries page
    elif menu == "Custom Queries":
        get_custom_queries()

    # Generate PDF Report page
    elif menu == "Generate PDF Report":
        get_pdf_report()

    # Print Chat History page
    elif menu == "Print Chat History":
        # Custom button to clear chat history
        with stylable_container(
            key="blue-button",
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
            if streamlit.button("Clear Chat History"):
                streamlit.session_state.chat_history = list()
                # Log message
                streamlit.write(f"Cleared chat history.")

        # Add button to stream chat history
        if streamlit.button("Print Chat History"):
            if len(streamlit.session_state.chat_history) == 0:
                streamlit.write("No chat history to show.")
            else:
                stream_chat_history()



if __name__ == "__main__":
    main()
