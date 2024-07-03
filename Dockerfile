# Use an official Python runtime as a parent image
FROM python:3.11.3-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    poppler-utils \
    tesseract-ocr \
    qpdf \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Expose the ports for the parsing service and Streamlit
EXPOSE 8005 8501

# Set the entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command
CMD ["/bin/bash"]