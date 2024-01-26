# Use the base image from downloads.unstructured.io/unstructured-io/unstructured with the latest tag
#FROM --platform=linux/amd64 downloads.unstructured.io/unstructured-io/unstructured:latest
FROM python:3.9
RUN python3 -m pip install -U pip
RUN apt update \
    && apt install ffmpeg libsm6 libxext6  -y \
    && apt install pandoc -y \
    && apt install tesseract-ocr -y \
    && apt install poppler-utils -y \
    && apt autoremove \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* 

# Set the working directory
WORKDIR /app/data_extraction
# Copy the AI Starterkit into the container at /app
COPY . /app
RUN pip install -r requirements.txt
RUN python -c 'from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings; HuggingFaceInstructEmbeddings()'
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN pip install notebook

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--allow-root" ]