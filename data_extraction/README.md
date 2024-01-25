SambaNova AI Starter Kits
====================

# Data Extraction Examples

- [Data Extraction Examples](#data-extraction-examples)
    - [Overview](#overview)
    - [Getting started](#getting-started)
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

1. Clone repo.
```
git clone https://github.sambanovasystems.com/SambaNova/ai-starter-kit.git
```
2. Install requirements: It is recommended to use virtualenv or conda environment for installation.
```
python3 -m venv data_extract_env
source data_extract_env/bin/activate
cd data_extraction
pip install -r requirements.txt
```
3. Some text extraction examples use Unstructured lib. Please register at [Unstructured.io](https://unstructured.io/#get-api-key) to get a free API Key. then create an enviroment file to store the APIkey and URL provided.
```
echo 'UNSTRUCTURED_API_KEY="your_API_key_here"\nUNSTRUCTURED_API_KEY="your_API_url_here"' > export.env
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

- [qa_qc_util.ipynb](qa_qc_util.ipynb): This notebook offers a simple utility for visualizing text boxes extracted using the PyMuPDF or Fitz package. This visualization can be particularly helpful when dealing with complex multi-column PDF documents, aiding in the debugging process.

- [unstructured_extraction.ipynb](unstructured_extraction.ipynb): This notebook provides examples of text extraction from files in different input format using Unstructured lib. Section 7 includes two loading examples first one using unstructured API and the other using local unstructured loader

#### Multidocument 

- [multidocs_extraction.ipynb](multidocs_extraction.ipynb): This notebook provides examples of text extraction from multiple docs using Unstructured.io as file loader. The input format could be a mixed of formats.

### Included files
- [sample_data](sample_data): Contains sample data for running the notebooks.

- [src](src): contains the source code for some functionalities used in the notebooks.
