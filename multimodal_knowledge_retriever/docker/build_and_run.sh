#!/bin/bash

# Build Docker image
docker build -t multimodal_knowledge_retriever -f docker/Dockerfile ..

# Run Docker container
docker run -p 8501:8501 -v $(pwd)/data:/app/multimodal_knowledge_retriever/data multimodal_knowledge_retriever
