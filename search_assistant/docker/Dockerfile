FROM python:3.10-slim

WORKDIR /app

RUN apt update
RUN apt install tesseract-ocr -y
RUN apt install poppler-utils -y

# Copy all necessary files
COPY search_assistant /app/search_assistant
COPY utils/ /app/utils
COPY vectordb /app/vectordb
COPY .env /app/.env

RUN pip3 install -r /app/search_assistant/requirements.txt
#RUN python -c 'from langchain_community.embeddings import HuggingFaceInstructEmbeddings; HuggingFaceInstructEmbeddings()'


WORKDIR /app/search_assistant
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "streamlit/app.py", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]