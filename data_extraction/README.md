<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

SambaNova AI Starter Kits
====================

# Data Extraction Examples

- [Data Extraction Examples](#data-extraction-examples)
    - [Overview](#overview)
    - [Getting started](#getting-started)
        - [Deploy in vitual environment](#option-1-run-through-local-virtual-environment)
        - [Deploy in Docker container](#option-2-run-via-docker)
    - [File Loaders](#file-loaders)
        - [CSV Documents](#csv-documents)
        - [XLS/XLSX Documents](#xlsxlsx-documents)
        - [DOC/DOCX Documents](#docdocx-documents)
        - [RTF Documents](#rtf-documents)
        - [Markdown Documents](#markdown-documents)
        - [HTML Documents](#html-documents)
        - [PDF Documents](#pdf-documents)
    - [Included Files](#included-files)

## Overview
This kit include a series of Notebooks that demonstrates various methods for extracting text from documents in different input formats. including Markdown, PDF, CSV, RTF, DOCX, XLS, HTML

## Getting started

### Deploy the starter kit

#### Option 1: Run through local virtual environment

> **Important:** With this option some funcionalities requires to install some pakges directly in your system
>- [pandoc](https://pandoc.org/installing.html) (for local rtf files loading)
>- [tesseract-ocr](https://tesseract-ocr.github.io/tessdoc/Installation.html) (for PDF ocr and table extraction)
>- [poppler-utils](https://pdf2image.readthedocs.io/en/latest/installation.html) (for PDF ocr and table extraction)

1. Clone repo.
```
git clone https://github.sambanovasystems.com/SambaNova/ai-starter-kit.git
```
2.1 Install requirements: It is recommended to use virtualenv or conda environment for installation.
```
cd ai-starter-kit
python3 -m venv data_extract_env
source data_extract_env/bin/activate
cd data_extraction
pip install -r requirements.txt
```
2.2 Install requirements for paddle utility: ,It is recommended to use virtualenv or conda environment for installation.
>Use this in case you want to use **Paddle OCR** recipe for [PDF OCR and table extraction](pdf_extraction_ocr_tables.ipynb) you shold use the requirementsPaddle file instead
```
cd ai-starter-kit
python3 -m venv data_extract_env
source data_extract_env/bin/activate
cd data_extraction
pip install -r requirementsPaddle.txt
```
3. Some text extraction examples use Unstructured lib. Please register at [Unstructured.io](https://unstructured.io/#get-api-key) to get a free API Key. then create an enviroment file to store the APIkey and URL provided.
```
echo 'UNSTRUCTURED_API_KEY="your_API_key_here"\nUNSTRUCTURED_API_KEY="your_API_url_here"' > .env
```

#### Option 2: Run via Docker
>With this option all funcionalities and notebook are ready to use 

>You need to have the Docker engine installed [Docker installation](https://docs.docker.com/engine/install/)

1. Clone repo.
```
git clone https://github.sambanovasystems.com/SambaNova/ai-starter-kit.git
```
2. Some text extraction examples use Unstructured lib. Please register at [Unstructured.io](https://unstructured.io/#get-api-key) to get a free API Key. then create an enviroment file to store the APIkey and URL provided.
```
echo 'UNSTRUCTURED_API_KEY="your_API_key_here"\nUNSTRUCTURED_API_KEY="your_API_url_here"' > .env
```
3.1 Run data extraction docker container
```
sudo docker-compose up data_extraction_service 
```
3.2 Run data extraction docker container for Paddle utility
3.1 Run data extraction docker container
```
sudo docker-compose up data_extraction_service 
```
3.2 Run data extraction docker container for Paddle utility
>Use this in case you want to use **Paddle OCR** recipe for [PDF OCR and table extraction](pdf_extraction_ocr_tables.ipynb) you shold use the startPaddle script instead
```
sudo docker-compose up data_extraction_paddle_service  
```


### File loaders 

You will find several data extraction recipes and pipelines in the [notebooks](notebooks) folder as follows:

#### CSV Documents

- [csv_extraction.ipynb](notebooks/csv_extraction.ipynb): This notebook provides examples of text extraction from CSV files using different packages. Depending on your specific use case, some packages may perform better than others.

#### XLS/XLSX Documents

- [xls_extraction.ipynb](notebooks/xls_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 2 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### DOC/DOCX Documents

- [docx_extraction.ipynb](notebooks/docx_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 3 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### RTF Documents

- [rtf_extraction.ipynb](notebooks/rtf_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 4 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### Markdown Documents

- [markdown_extraction.ipynb](notebooks/markdown_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 5 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### HTML Documents

- [web_extraction.ipynb](notebooks/web_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 6 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### PDF Documents

- [pdf_extraction.ipynb](notebooks/pdf_extraction.ipynb): This notebook provides examples of text extraction from PDF documents using different packages including different OCR and non-OCR packages. Depending on your specific use case, some packages may perform better than others.

- [retrieval_from_pdf_tables.ipynb](notebooks/retrieval_from_pdf_tables.ipynb):  This notebook provides an example of a simple RAG retiever and an example of a multivector RAG retriever for pdf with tables retrieval. For SambaNova model endpoint usage refer to the [ai-starter-kit docs](../README.md) 

- [qa_qc_util.ipynb](notebooks/qa_qc_util.ipynb): This notebook offers a simple utility for visualizing text boxes extracted using the Fitz package. This visualization can be particularly helpful when dealing with complex multi-column PDF documents, and in the debugging process.


### Included files
- [data](data): Contains sample data for running the notebooks, and is used as storage for intermediate steps for recipes.

- [src](src): contains the source code for some functionalities used in the notebooks.

- [docker](docker): contains Dockerfile for data extraction starter kit