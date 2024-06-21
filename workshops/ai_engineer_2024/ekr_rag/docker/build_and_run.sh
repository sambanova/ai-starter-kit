#!/bin/bash

# Build Docker image
docker build -t enterprise_knowledge_retriever -f docker/Dockerfile ..

# Run Docker container
docker run -p 8501:8501 -v $(pwd)/data:/app/enterprise_knowledge_retriever/data enterprise_knowledge_retriever
