FROM python:3.10-slim

WORKDIR /app

RUN apt update && apt install build-essential -y

COPY edgar_qna /app/edgar_qna
COPY vectordb/ /app/vectordb
COPY utils/ /app/utils
COPY .env /app/.env

RUN pip3 install -r /app/edgar_qna/requirements.txt
RUN python -c 'from langchain_community.embeddings import HuggingFaceInstructEmbeddings; HuggingFaceInstructEmbeddings()'

WORKDIR /app/edgar_qna
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "streamlit/app_comparative_chat.py", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]