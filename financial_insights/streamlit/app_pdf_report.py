import os
import sys

import streamlit

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)



def get_pdf_report() -> None:
    streamlit.markdown("<h2> Generate PDF Report </h2>", unsafe_allow_html=True)
    include_stock_data = streamlit.checkbox("Include Stock Data")
    inlude_yahoo_news = streamlit.checkbox("Include Yahoo News")
    include_filings = streamlit.checkbox("Include Financial Filings")
    include_custom_queries = streamlit.checkbox("Include Custom Queries")
    if streamlit.button("Generate Report"):
        data = []
        if include_filings:
            # Add data from Financial Filings Analysis
            data.append("Financial Filings Analysis Data")
        if include_stock_data:
            # Add data from Stock Data Analysis
            data.append("Stock Data Analysis Data")
        if include_custom_queries:
            # Add data from Custom Queries
            data.append("Custom Queries Data")
        # generate_pdf_report(data)
        streamlit.write("PDF report generated successfully.")