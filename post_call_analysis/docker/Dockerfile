FROM python:3.10-slim

ARG SNSDK=snsdk-XX.X.X+rcX-py3-none-any.whl
ARG SNAPI=SambaStudio_API_Client-XX.X.X+rcX-py3-none-any.whl
ARG SAMBASTUDIO_KEY=abcd12345

WORKDIR /app

RUN apt update && apt-get install build-essential -y
RUN apt install ffmpeg libsm6 libxext6  -y

# Copy snsdk and snapi .whl files
COPY ${SNSDK} /app/${SNSDK}
COPY ${SNAPI} /app/${SNAPI}

# Copy all necessary files for runing the starter kit
COPY post_call_analysis /app/post_call_analysis
COPY utils/ /app/utils
COPY vectordb /app/vectordb
COPY .env /app/.env
RUN mkdir /root/.snapi
#COPY secret.txt /root/.snapi/secret.txt

# set SNAPI and SNSDK
RUN echo ${SAMBASTUDIO_KEY} > ~/.snapi/secret.txt 
RUN pip3 install -U pip
RUN pip3 install ${SNSDK}
RUN pip3 install ${SNAPI}

# set postcall analysis
RUN pip3 install -r /app/post_call_analysis/requirements.txt
# RUN python -c 'from langchain_community.embeddings import HuggingFaceInstructEmbeddings; HuggingFaceInstructEmbeddings()'

WORKDIR /app/post_call_analysis
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "streamlit/app.py", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]
