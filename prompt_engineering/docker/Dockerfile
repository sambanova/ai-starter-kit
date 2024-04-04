FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install build-essential -y

COPY prompt_engineering /app/prompt_engineering
COPY vectordb/ /app/vectordb
COPY utils/ /app/utils
COPY .env /app/.env

RUN pip3 install -r /app/prompt_engineering/requirements.txt
#RUN python -c 'from langchain_community.embeddings import HuggingFaceInstructEmbeddings; HuggingFaceInstructEmbeddings()'

WORKDIR /app/prompt_engineering
CMD ["streamlit", "run", "streamlit/app.py", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]
