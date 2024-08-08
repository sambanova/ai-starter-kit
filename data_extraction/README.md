<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

SambaNova AI Starter Kits
====================

# Data Extraction Examples

<!-- TOC -->
- [Data Extraction Examples](#data-extraction-examples)
    - [Deploy the starter kit](#deploy-the-starter-kit)
        - [Option 1: Run through local virtual environment](#option-1-run-through-local-virtual-environment)
        - [Option 2: Run via Docker](#option-2-run-via-docker)
    - [File loaders](#file-loaders)
        - [CSV Documents](#csv-documents)
        - [XLS/XLSX Documents](#xlsxlsx-documents)
        - [DOC/DOCX Documents](#docdocx-documents)
        - [RTF Documents](#rtf-documents)
        - [Markdown Documents](#markdown-documents)
        - [HTML Documents](#html-documents)
        - [PDF Documents](#pdf-documents)
    - [Included files](#included-files)

<!-- /TOC -->

This kit include a series of Notebooks that demonstrates various methods for extracting text from documents in different input formats. including Markdown, PDF, CSV, RTF, DOCX, XLS, HTML

## Deploy the starter kit

### Option 1: Run through local virtual environment

> **Important:** With this option you have to install some packages directly in your system:
>- [pandoc](https://pandoc.org/installing.html) (for local rtf files loading)
>- [tesseract-ocr](https://tesseract-ocr.github.io/tessdoc/Installation.html) (for PDF ocr and table extraction)
>- [poppler-utils](https://pdf2image.readthedocs.io/en/latest/installation.html) (for PDF ocr and table extraction)

1. Clone the repo.
```
git clone https://github.sambanovasystems.com/SambaNova/ai-starter-kit.git
```
2. (Recommended) Set up a `venv` or `conda` environment for installation.
```
cd ai-starter-kit
python3 -m venv data_extract_env
source data_extract_env/bin/activate
cd data_extraction
pip install -r requirements.txt
```
3. Install files required for the paddle utility: We recommend that you use virtualenv or conda environment for installation.
>Use this in case you want to use **Paddle OCR** recipe for [PDF OCR and table extraction](notebooks/pdf_extraction.ipynb) you should use the requirementsPaddle file instead.
```
cd ai-starter-kit
python3 -m venv data_extract_env
source data_extract_env/bin/activate
cd data_extraction
pip install -r requirementsPaddle.txt
```
4. Some text extraction examples use the `Unstructured` library. Register at [Unstructured.io](https://unstructured.io/#get-api-key) to get a free API key and create an enviroment file to store the API key and URL:
```
echo 'UNSTRUCTURED_API_KEY="your_API_key_here"\nUNSTRUCTURED_API_KEY="your_API_url_here"' > .env
```
- Or start the parsing service, add parsing service url and API key (can be any value): 
```
make start-parsing-service
```
```
echo 'UNSTRUCTURED_API_KEY="your_API_key_here"\nUNSTRUCTURED_API_KEY="http://localhost:8005/general/v0/general"' > .env
```

### Option 2: Run via Docker

With this option, all functionality and Jupyter notebooks are ready to use. 

1. Ensure that you have the Docker engine installed [Docker installation](https://docs.docker.com/engine/install/).

2. Clone the repo.
```
git clone https://github.sambanovasystems.com/SambaNova/ai-starter-kit.git
```
3. Some text extraction examples use the `Unstructured` library. Register at [Unstructured.io](https://unstructured.io/#get-api-key) to get a free API key and create an enviroment file to store the API key and URL:
```
echo 'UNSTRUCTURED_API_KEY="your_API_key_here"\nUNSTRUCTURED_API_KEY="your_API_url_here"' > .env
```
- Or start the parsing service, add parsing service url and API key (can be any value): 
```
make start-parsing-service
```
```
echo 'UNSTRUCTURED_API_KEY="your_API_key_here"\nUNSTRUCTURED_API_KEY="http://host.docker.internal:8005/general/v0/general"' > .env
```

4. Run the data extraction Docker container:
```
sudo docker-compose up data_extraction_service 
```

5. Run data extraction docker container for Paddle utility.
>Use this in case you want to use **Paddle OCR** recipe for [PDF OCR and table extraction](notebooks/pdf_extraction.ipynb), use the `startPaddle` script instead

```
sudo docker-compose up data_extraction_paddle_service  
```


## File loaders 

The [notebooks](notebooks) folder has several data extraction recipes and pipelines: 

### CSV Documents

- [csv_extraction.ipynb](notebooks/csv_extraction.ipynb): Examples of text extraction from CSV files using different packages. Depending on your use case, some packages may perform better than others.

### XLS/XLSX Documents

- [xls_extraction.ipynb](notebooks/xls_extraction.ipynb): Examples of text extraction from files in different input formats using the `Unstructured` library. Section 2 includes two examples, one using the `Unstructured` API and the other using the local unstructured loader.

### DOC/DOCX Documents

- [docx_extraction.ipynb](notebooks/docx_extraction.ipynb): Examples of text extraction from files in different input formats using the `Unstructured` library. Section 3 includes two examples, one using the `Unstructured` API and the other using the local unstructured loader.

### RTF Documents

- [rtf_extraction.ipynb](notebooks/rtf_extraction.ipynb): Examples of text extraction from files in different input formats using the `Unstructured` library. Section 4 includes two examples, one using the `Unstructured` API and the other using the local unstructured loader.

### Markdown Documents

- [markdown_extraction.ipynb](notebooks/markdown_extraction.ipynb): Examples of text extraction from files in different input formats using the `Unstructured` library. Section 5 includes two examples, one using the `Unstructured` API and the other using the local unstructured loader.

### HTML Documents

- [web_extraction.ipynb](notebooks/web_extraction.ipynb): Examples of text extraction from files in different input format using the `Unstructured` library. Section 6 includes two loading examples, one using the `Unstructured` API and the other using the local unstructured loader.

### PDF Documents

- [pdf_extraction.ipynb](notebooks/pdf_extraction.ipynb): Examples of text extraction from PDF documents using different packages including different OCR and non-OCR packages. Depending on your specific use case, some packages may perform better than others.

- [retrieval_from_pdf_tables.ipynb](notebooks/retrieval_from_pdf_tables.ipynb): Example of a simple RAG retiever and an example of a multivector RAG retriever for PDF with tables retrieval. For SambaNova model endpoint usage, refer to the [top-level ai-starter-kit README](../README.md) 

- [qa_qc_util.ipynb](notebooks/qa_qc_util.ipynb): Simple utility for visualizing text boxes extracted using the Fitz package. This visualization can be particularly helpful when dealing with complex multi-column PDF documents, and in the debugging process.


## Included files
- [data](data): Sample data for running the notebooks. Used as storage for intermediate steps.

- [src](src): Source code for some functionalities used in the notebooks.

- [docker](docker): Docker file for the data extraction starter kit.
