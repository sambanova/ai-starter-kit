# Use an official Python runtime as a parent image
FROM python:3.11.5-bookworm as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    tesseract-ocr \
    qpdf \
    make \
    ffmpeg \
    libsm6 \
    libxext6 \
    --fix-missing \
    && rm -rf /var/lib/apt/lists/*

# Set working directory in the container
WORKDIR /app

# Copy only the requirements files first
COPY base-requirements.txt tests/requirements.txt ./

# Upgrade pip and install project dependencies
# Use BuildKit's cache mount to speed up pip installs
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip uv && \
    uv pip install --system -r base-requirements.txt

# Final stage
FROM python:3.11.5-slim-bookworm

# Copy installed packages from builder stage
COPY --from=builder /usr/local /usr/local

# Install runtime system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    poppler-utils \
    tesseract-ocr \
    qpdf \
    make \
    ffmpeg \
    libsm6 \
    libxext6 \
    --fix-missing \
    && rm -rf /var/lib/apt/lists/*

# Set working directory in the container
WORKDIR /app

# Copy the application code (excluding .env file)
COPY . .
RUN rm -f .env

# Add build argument for parsing service setup
ARG SETUP_PARSING_SERVICE=no

# Set up parsing service conditionally
RUN if [ "$SETUP_PARSING_SERVICE" = "yes" ]; then \
        cd utils/parsing/unstructured-api && \
        make install; \
    fi

# Copy the startup script and make it executable
COPY docker-startup.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-startup.sh

# Expose the ports for the parsing service and Streamlit
EXPOSE 8005 8501

# Set the startup script as the entrypoint
ENTRYPOINT ["/usr/local/bin/docker-startup.sh"]

# Default command
CMD ["make", "run"]