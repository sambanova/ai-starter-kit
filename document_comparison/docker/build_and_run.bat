@echo off

REM Build Docker image
docker build -t document_comparison -f docker\Dockerfile ..

REM Run Docker container
docker run -p 8501:8501 -v %cd%\data:/app/document_comparison/data document_comparison
