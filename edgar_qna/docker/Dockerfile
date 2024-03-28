FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install build-essential -y

# Copy all necessary files
COPY edgar_qna/src /app/edgar_qna/src
COPY edgar_qna/api /app/edgar_qna/api
COPY edgar_qna/prompts /app/edgar_qna/prompts
COPY vectordb/ /app/vectordb
COPY utils/ /app/utils
COPY edgar_qna/requirements.txt /app/edgar_qna/requirements.txt
COPY .env /app/.env

RUN pip3 install -r /app/edgar_qna/requirements.txt
RUN python -c 'from langchain_community.embeddings import HuggingFaceInstructEmbeddings; HuggingFaceInstructEmbeddings()'

WORKDIR /app/edgar_qna
CMD ["streamlit", "run", "app_comparative_chat.py", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]
