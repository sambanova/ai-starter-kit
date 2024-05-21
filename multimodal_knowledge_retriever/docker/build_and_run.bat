@echo off

REM Build Docker image
docker build -t mutimodal_knowledge_retriever -f docker\Dockerfile ..

REM Run Docker container
docker run -p 8501:8501 -v %cd%\data:/app/mutimodal_knowledge_retriever/data mutimodal_knowledge_retriever
