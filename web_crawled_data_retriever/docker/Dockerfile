FROM python:3.10-slim

WORKDIR /app

# Copy all necessary files
COPY web_crawled_data_retriever /app/web_crawled_data_retriever
COPY utils/ /app/utils
COPY vectordb /app/vectordb
COPY .env /app/.env

RUN pip3 install -r /app/web_crawled_data_retriever/requirements.txt

WORKDIR /app/web_crawled_data_retriever
CMD ["streamlit", "run", "streamlit/app.py", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]
