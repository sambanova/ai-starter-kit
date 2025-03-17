FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip

RUN apt-get clean && \
    apt-get update && \
    apt-get install -y python3-dev build-essential && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git config --global --add safe.directory /app

COPY ./e2e_fine_tuning/backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install e2e_fine_tuning/backend/sambanova_packages/snsdk-24.11.1+rc4-py3-none-any.whl e2e_fine_tuning/backend/sambanova_packages/SambaStudio_API_Client-24.11.1+rc4-py3-none-any.whl

EXPOSE 8000

CMD ["uvicorn", "e2e_fine_tuning.backend.src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
