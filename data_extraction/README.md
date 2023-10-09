## Data Extraction Examples

This kit demonstrates various methods for extracting text from documents in different input formats.

### Getting started

1. Clone repo.
```
git clone https://github.sambanovasystems.com/SambaNova/ai-starter-kit.git
```
2. Install requirements: It is recommended to use virtualenv or conda environment for installation.
```
cd data_extraction
python3 -m venv data_extract
source data_extract/bin/activate
pip install -r requirements.txt
```
### PDF Documents

- [pdf_extraction_non_OCR.ipynb](pdf_extraction_non_ocr.ipynb): This notebook provides examples of text extraction from PDF documents using different packages. Depending on your specific use case, some packages may perform better than others.

- [qa_qc_util.ipynb](qa_qc_util.ipynb): This notebook offers a simple utility for visualizing text boxes extracted using the PyMuPDF or Fitz package. This visualization can be particularly helpful when dealing with complex multi-column PDF documents, aiding in the debugging process.

- [sample_data](sample_data): Contains sample data for running the notebooks.

- [src](src): contains the source code for some functionalities.
