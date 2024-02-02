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
        - [Multidocument](#multidocument)
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
echo 'UNSTRUCTURED_API_KEY="your_API_key_here"\nUNSTRUCTURED_API_KEY="your_API_url_here"' > export.env
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
echo 'UNSTRUCTURED_API_KEY="your_API_key_here"\nUNSTRUCTURED_API_KEY="your_API_url_here"' > export.env
```
3.1 Build data extraction AI starter kit docker image
```
cd ai-starter-kit
sudo docker build -t data_extraction:v1 -f data_extraction/docker/Dockerfile . 
```
3.2 Build data extraction AI starter kit docker image for Paddle utility
>Use this in case you want to use **Paddle OCR** recipe for [PDF OCR and table extraction](pdf_extraction_ocr_tables.ipynb) you shold use the PaddleDockerfile file instead
```
cd ai-starter-kit
sudo docker build -t data_extraction_paddle:v1 -f data_extraction/docker/PaddleDockerfile . 
```
4.1 Run data extraction docker container
```
sudo ./data_extraction/docker/start.sh    
```
4.2 Run data extraction docker container for Paddle utility
>Use this in case you want to use **Paddle OCR** recipe for [PDF OCR and table extraction](pdf_extraction_ocr_tables.ipynb) you shold use the startPaddle script instead
```
sudo ./data_extraction/docker/startPaddle.sh    
```


### File loaders 

#### CSV Documents

- [csv_extraction.ipynb](csv_extraction.ipynb): This notebook provides examples of text extraction from CSV files using different packages. Depending on your specific use case, some packages may perform better than others.

- [unstructured_extraction.ipynb](unstructured_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 1 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### XLS/XLSX Documents

- [unstructured_extraction.ipynb](unstructured_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 2 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### DOC/DOCX Documents

- [unstructured_extraction.ipynb](unstructured_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 3 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### RTF Documents

- [unstructured_extraction.ipynb](unstructured_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 4 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### Markdown Documents

- [unstructured_extraction.ipynb](unstructured_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 5 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### HTML Documents

- [unstructured_extraction.ipynb](unstructured_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 6 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### PDF Documents

- [pdf_extraction_non_OCR.ipynb](pdf_extraction_non_ocr.ipynb): This notebook provides examples of text extraction from PDF documents using different packages. Depending on your specific use case, some packages may perform better than others.

- [pdf_extraction_ocr_tables.ipynb](pdf_extraction_ocr_tables.ipynb): This notebook provides examples of text and tables extraction from PDF documents using different OCR packages. Depending on your specific use case, some packages may perform better than others. It also provides an example of a simple RAG retiever an an example of a multivector RAG retriever. For SambaNova model endpoint usage refer [here](../README.md) 

- [qa_qc_util.ipynb](qa_qc_util.ipynb): This notebook offers a simple utility for visualizing text boxes extracted using the PyMuPDF or Fitz package. This visualization can be particularly helpful when dealing with complex multi-column PDF documents, aiding in the debugging process.

- [unstructured_extraction.ipynb](unstructured_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 7 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### Multidocument 

- [multidocs_extraction.ipynb](multidocs_extraction.ipynb): This notebook provides examples of text extraction from multiple docs using Unstructured.io as file loader. The input format could be a mixed of formats.

### Included files
- [sample_data](sample_data): Contains sample data for running the notebooks.

- [src](src): contains the source code for some functionalities used in the notebooks.

- [docker](docker): contains Dockerfile and run script for data extraction starter kit