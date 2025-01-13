#!/bin/bash

# Build Docker image
docker build -t document_comparison -f docker/Dockerfile ..

# Run Docker container
docker run -p 8501:8501 -v $(pwd)/data:/app/document_comparison/data document_comparison
