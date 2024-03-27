FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install build-essential -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy all necessary files
COPY enterprise_knowledge_retriever /app/enterprise_knowledge_retriever
COPY utils/ /app/utils
COPY data_extraction/src /app/data_extraction/src
COPY vectordb /app/vectordb
COPY .env /app/.env

RUN apt install tesseract-ocr -y
RUN pip3 install -r /app/enterprise_knowledge_retriever/requirements.txt
RUN python -c 'from langchain_community.embeddings import HuggingFaceInstructEmbeddings; HuggingFaceInstructEmbeddings()'

WORKDIR /app/enterprise_knowledge_retriever
CMD ["streamlit", "run", "streamlit/app.py", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]
