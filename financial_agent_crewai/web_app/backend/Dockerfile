FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip

RUN apt-get update && \
    apt-get install -y python3-dev build-essential python3-pip python3-cffi python3-brotli libpango-1.0-0 libpangoft2-1.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./web_app/backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./web_app/backend ./src

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]