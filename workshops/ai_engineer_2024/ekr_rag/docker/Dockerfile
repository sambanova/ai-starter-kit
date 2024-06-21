FROM python:3.10-slim

WORKDIR /app

RUN apt update && apt-get install build-essential -y
RUN apt install ffmpeg libsm6 libxext6  -y
RUN apt install pandoc -y
RUN apt install tesseract-ocr -y
RUN apt install poppler-utils -y

# Copy all necessary files
COPY enterprise_knowledge_retriever /app/enterprise_knowledge_retriever
COPY utils/ /app/utils
COPY data_extraction/src /app/data_extraction/src
COPY vectordb /app/vectordb
COPY .env /app/.env

RUN pip3 install -r /app/enterprise_knowledge_retriever/requirements.txt
#RUN python -c 'from langchain_community.embeddings import HuggingFaceInstructEmbeddings; HuggingFaceInstructEmbeddings()'

WORKDIR /app/enterprise_knowledge_retriever
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "streamlit/app.py", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]
