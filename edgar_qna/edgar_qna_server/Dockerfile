# app/Dockerfile

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install build-essential -y

COPY src/models/ src/models
COPY edgar_qna/edgar_qna_server edgar_qna_server

WORKDIR /app/edgar_qna_server
RUN pip3 install -r requirements.txt
RUN python -c 'from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings; HuggingFaceInstructEmbeddings()'
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader averaged_perceptron_tagger

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
