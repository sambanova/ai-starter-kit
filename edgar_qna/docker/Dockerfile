FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install build-essential -y

# Copy all necessary files
COPY edgar_qna/src /app/edgar_qna/src
COPY edgar_qna/api /app/edgar_qna/api
COPY edgar_qna/prompts /app/edgar_qna/prompts
COPY vectordb/ /app/vectordb
COPY utils/ /app/utils
COPY edgar_qna/data/vectordbs/tsla /app/edgar_qna/data/vectordbs/tsla
COPY edgar_qna/requirements.txt /app/edgar_qna/requirements.txt
COPY export.env /app/export.env

RUN pip3 install -r /app/edgar_qna/requirements.txt
RUN python -c 'from langchain_community.embeddings import HuggingFaceInstructEmbeddings; HuggingFaceInstructEmbeddings()'

WORKDIR /app/edgar_qna/api
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
