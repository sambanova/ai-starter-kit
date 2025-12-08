import argparse
import json
from typing import Any, Dict, List, Optional

from langchain_classic.docstore.document import Document
from langchain_classic.document_loaders.base import BaseLoader

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.staging.base import elements_to_json
except ImportError as e:
    raise ValueError('unstructured package not found, please install it with `pip install unstructured`') from e


def extract_elements_from_file(input_file: str, output_image_dir: str) -> List[Any]:
    """
    Extracts elements from a PDF file using Unestructured Pytesseract OCR tool.

    Args:
        input_file (str): The path to the input PDF file.
        output_image_dir (str): The directory to save the extracted images.

    Returns:
        list: A list of raw PDF elements.

    """
    raw_pdf_elements = partition_pdf(
        filename=input_file,
        extract_images_in_pdf=False,
        # Specifies whether to extract images from the PDF.
        # If False, unstructured processing first identifies embedded image blocks.
        strategy='hi_res',
        # The processing strategy to use. Valid strategies are "hi_res", "ocr_only", and "fast".
        model_name='yolox',
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        infer_table_structure=True,
        # Additional optional parameters (commented out for brevity):
        # chunking_strategy="by_title",
        # If used, it specifies how to aggregate text blocks based on document titles.
        # Titles are any sub-section of the document
        # max_characters=4000,
        # Chunking params to aggregate text blocks
        # new_after_n_chars=3800,
        # Attempt to create a new chunk 3800 chars
        # combine_text_under_n_chars=2000,
        # Attempt to keep chunks > 2000 chars
        image_output_dir_path=output_image_dir,
    )

    return raw_pdf_elements


def process_json_file(input_json_filename: str, output_file_path: str) -> List[Any] | str:
    """
    Processes a JSON file and write the extracted elements (text, tables) in the specified plain text file.

    Args:
        input_json_filename (str): The path to the input JSON file.
        output_file (str): The path to the output file to write the extracted elements.

    Returns:
        str: A string with all text and tables extarcted.
    """

    # Read the JSON file
    with open(input_json_filename, 'r') as file:
        data = json.load(file)

    # Iterate over the JSON data and extract required table elements
    extracted_elements = []
    for entry in data:
        if entry['type'] == 'Table':
            extracted_elements.append(entry['metadata']['text_as_html'])
        else:
            extracted_elements.append(entry['text'])
        # comment out this if need only table extraction

    # Write the extracted elements to the output file
    if output_file_path:
        with open(output_file_path, 'w') as output_file:
            for element in extracted_elements:
                output_file.write(element + '\n\n')
                # Adding two newlines for separation
    return '\n\n'.join(extracted_elements)


def process_elements(elements: str, output_file_path: str) -> str:
    """
    Processes a JSON file and write the extracted elements (text, tables) in the specified plain text file.

    Args:
        elements (str): json like String of elements.
        output_file (str): The path to the output file to write the extracted elements.

    Returns:
        str: A string with all text and tables extarcted.
    """
    # Read the elements string
    data = json.loads(elements)
    # Iterate over the JSON data and extract required table elements
    extracted_elements = []
    for entry in data:
        if entry['type'] == 'Table':
            extracted_elements.append(entry['metadata']['text_as_html'])
        else:
            extracted_elements.append(entry['text'])
        # comment out this if need only table extraction

    # Write the extracted elements to the output file
    if output_file_path:
        with open(output_file_path, 'w') as output_file:
            for element in extracted_elements:
                output_file.write(element + '\n\n')
                # Adding two newlines for separation
    return '\n\n'.join(extracted_elements)


class UnstructuredPdfPytesseractLoader(BaseLoader):
    """Load pdf files using Unstructured pytesseract module.

    Args:
        file_path: Path to the file to load.
        output_image_dir: dir for storing images extracted form pdf file , default = None.
        output_file: text file for saving the result of the extraction , default = None.
        save_json: flag for storing elements structure along the input file , default = False.
    """

    def __init__(
        self,
        file_path: str,
        output_image_dir: Optional[str] = None,
        output_file: Optional[str] = None,
        save_json: bool = False,
        **unstructured_kwargs: Any,
    ) -> None:
        # Initialize with file path.
        self.file_path = file_path
        self.output_file = output_file
        self.output_image_dir = output_image_dir
        self.save_json = save_json

    def _get_metadata(self) -> Dict[str, Any]:
        return {'source': self.file_path}

    def _get_elements(self) -> List[Any] | str:
        assert self.output_image_dir is not None
        self.pdf_elements = extract_elements_from_file(self.file_path, self.output_image_dir)
        if self.save_json:
            json_output = f'{self.file_path.split(".")[0]}.json'
            elements_to_json(self.pdf_elements, filename=json_output)
            assert self.output_file is not None
            return process_json_file(json_output, self.output_file)
        else:
            elements = elements_to_json(self.pdf_elements)
            assert elements is not None and self.output_file is not None
            return process_elements(elements, self.output_file)

    def load(self) -> List[Document]:
        elements = self._get_elements()
        metadata = self._get_metadata()
        assert isinstance(elements, str)
        return [Document(page_content=elements, metadata=metadata)]


def main(input_file: str, output_file: str, output_image_dir: str) -> None:
    pdf_elements = extract_elements_from_file(input_file, output_image_dir)
    json_output_file = f'{input_file.split(".")[0]}.json'
    elements_to_json(pdf_elements, filename=json_output_file)
    process_json_file(json_output_file, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract text and tables from PDF')
    # Required arguments
    parser.add_argument('input_file', type=str, help='Path to the input pdf file')
    parser.add_argument('output_file', type=str, help='Path to the output text file')
    parser.add_argument(
        'output_image_dir',
        type=str,
        help='Path to the output directory to store images',
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.output_image_dir)
