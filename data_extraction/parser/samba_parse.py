import os
import json
import logging
import requests
import yaml
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd
import matplotlib.pyplot as plt
from unstructured.staging.base import elements_to_json
from unstructured.staging.base import dict_to_elements
from unstructured.chunking.basic import chunk_elements
from langchain_core.documents.base import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnstructuredAPIConfig(BaseModel):
    api_url: str = "https://api.unstructured.io/general/v0/general"
    api_key: str = Field(..., env="UNSTRUCTURED_API_KEY")
    strategy: str = "fast"
    hi_res_model_name: str = "detectron2onnx"
    ocr_languages: List[str] = []
    languages: List[str] = []
    coordinates: bool = False
    skip_infer_table_types: List[str] = []
    encoding: str = "utf-8"
    xml_keep_tags: bool = False
    include_page_breaks: bool = False
    unique_element_ids: bool = False
    chunking_strategy: Optional[str] = None
    chunking_params: Dict = Field(default_factory=dict)
    parallel_mode_enabled: bool = False
    parallel_mode_url: Optional[str] = None
    parallel_mode_threads: int = 3
    parallel_mode_split_size: int = 1
    parallel_retry_attempts: int = 2
    return_langchain_docs: bool = False
    chunk_size: int = 1500

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class UnstructuredAPIClient:
    """
    A Python wrapper class for the Unstructured API.
    """

    def __init__(self, config: UnstructuredAPIConfig):
        self.config = config

    def _make_request(self, file_path: str) -> List[Dict]:
        """
        Makes a request to the Unstructured API with the given file.

        Args:
            file_path (str): The path to the file to process.

        Returns:
            List[Dict]: The raw JSON response from the API.
        """
        try:
            with open(file_path, "rb") as f:
                files = {"files": f}
                payload = {
                    "strategy": self.config.strategy,
                    "hi_res_model_name": self.config.hi_res_model_name,
                    "ocr_languages": self.config.ocr_languages,
                    "languages": self.config.languages,
                    "coordinates": self.config.coordinates,
                    "skip_infer_table_types": self.config.skip_infer_table_types,
                    "encoding": self.config.encoding,
                    "xml_keep_tags": self.config.xml_keep_tags,
                    "include_page_breaks": self.config.include_page_breaks,
                    "unique_element_ids": self.config.unique_element_ids,
                    "chunking_strategy": self.config.chunking_strategy,
                    # **self.config.chunking_params,
                }
                headers = {
                    "accept": "application/json",
                    "unstructured-api-key": self.config.api_key,
                }

                if self.config.parallel_mode_enabled:
                    payload["split_pages"] = True
                    payload["parallel_url"] = self.config.parallel_mode_url
                    payload["max_concurrent_pages"] = self.config.parallel_mode_threads
                    payload["pages_per_batch"] = self.config.parallel_mode_split_size
                    payload["retry_attempts"] = self.config.parallel_retry_attempts

                response = requests.post(
                    self.config.api_url, files=files, data=payload, headers=headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error making request to Unstructured API: {e}")
            raise

    def parse(
        self,
        file_path: str,
        extract_tables: bool = False,
        extract_html_tables: bool = False,
        extract_combined_tables: bool = False,
        replace_table_text: bool = False,
    ) -> Tuple[
        List[Dict],
        Tuple[List[str], List[Dict]],
        Dict[str, Tuple[List[str], Dict[str, Union[str, List[str]]]]],
    ]:
        """
        Parses the given file using the Unstructured API.

        Args:
            file_path (str): The path to the file to process.
            extract_tables (bool): Whether to extract tables from the document. Defaults to False.
            extract_html_tables (bool): Whether to extract tables as HTML. Defaults to False.
            extract_combined_tables (bool): Whether to extract all tables combined into a single element. Defaults to False.
            replace_table_text (bool): Whether to replace the 'text' element of a table with the 'table_as_html' value. Defaults to False.

        Returns:
            Tuple[List[Dict], Tuple[List[str], List[Dict]], Dict[str, Tuple[List[str], Dict[str, Union[str, List[str]]]]]]: A tuple containing:
                - The raw JSON response from the API.
                - A tuple containing:
                    - A list of text elements extracted from the document.
                    - A list of metadata dictionaries for each element.
                - A dictionary mapping table names to tuples of (texts, metadata).
        """
        logger.info(f"Parsing {file_path}...")

        elements = self._make_request(file_path)
        elements_processed = dict_to_elements(elements)
        chunks = chunk_elements(
            elements_processed, max_characters=self.config.chunk_size
        )
        elements = elements_to_json(chunks)
        elements = json.loads(elements)

        if replace_table_text:
            for element in elements:
                print(element)
                if element["type"] == "Table":
                    element["text"] = element["metadata"]["text_as_html"]

        texts = []
        metadata_list = []
        table_data = {}

        try:
            for i, element in enumerate(tqdm(elements, desc="Processing elements")):
                text = element.get("text")
                metadata = element.get("metadata", {}).copy()

                for key in element:
                    if key not in ["text", "metadata"]:
                        metadata[key] = element[key]

                if text:
                    if element["type"] == "Table" and extract_tables:
                        if extract_html_tables:
                            html_table = element["metadata"]["text_as_html"]
                            table_data[f"table_{i}"] = (
                                [html_table],
                                metadata,
                            )
                        else:
                            table_data[f"table_{i}"] = (
                                [text],
                                metadata,
                            )
                    else:
                        texts.append(text)

                metadata_list.append(metadata)

            if extract_combined_tables:
                combined_tables = []
                combined_metadata = {}
                for table_text, table_metadata in table_data.values():
                    combined_tables.extend(table_text)
                    combined_metadata.update(table_metadata)
                table_data["combined_tables"] = (combined_tables, combined_metadata)

        except Exception as e:
            logger.error(f"Error processing elements: {e}")
            raise

        if self.config.return_langchain_docs:
            langchain_docs = self.get_langchain_docs(texts, metadata_list)
            return elements, (texts, metadata_list), table_data, langchain_docs
        else:
            return elements, (texts, metadata_list), table_data

    def get_langchain_docs(
        self,
        texts: List[str],
        metadata_list: List[Dict],
    ) -> List[Document]:
        """
        Creates a list of langchain Document objects from the parsed text and metadata.

        Args:
            texts (List[str]): The list of text elements extracted from the document.
            metadata_list (List[Dict]): The list of metadata dictionaries for each element.

        Returns:
            List[Document]: A list of langchain Document objects.
        """
        return [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(texts, metadata_list)
        ]


def process_file(
    client: UnstructuredAPIClient,
    file_path: str,
    output_dir: str,
    extract_tables: bool,
    extract_html_tables: bool,
    extract_combined_tables: bool,
    replace_table_text: bool,
) -> Union[
    Tuple[
        float,
        List[Dict],
        Tuple[List[str], List[Dict]],
        Dict[str, Tuple[List[str], Dict[str, Union[str, List[str]]]]],
    ],
    Tuple[
        float,
        List[Dict],
        Tuple[List[str], List[Dict]],
        Dict[str, Tuple[List[str], Dict[str, Union[str, List[str]]]]],
        List[Document],
    ],
]:
    """
    Processes a single file and saves the output to the specified directory.

    Args:
        client (UnstructuredAPIClient): The API client instance.
        file_path (str): The path to the file to process.
        output_dir (str): The directory to save the output files.
        extract_tables (bool): Whether to extract tables from the document.
        extract_html_tables (bool): Whether to extract tables as HTML.
        extract_combined_tables (bool): Whether to extract all tables combined into a single element.
        replace_table_text (bool): Whether to replace the 'text' element of a table with the 'table_as_html' value.

    Returns:
        Tuple[float, List[Dict], Tuple[List[str], List[Dict]], Dict[str, Tuple[List[str], Dict[str, Union[str, List[str]]]]]]: A tuple containing:
            - The time taken to process the file.
            - The raw JSON response from the API.
            - A tuple containing:
                - A list of text elements extracted from the document.
                - A list of metadata dictionaries for each element.
            - A dictionary mapping table names to tuples of (texts, metadata).
    """
    start_time = time.time()
    elements, (texts, metadata_list), table_data = client.parse(
        file_path,
        extract_tables,
        extract_html_tables,
        extract_combined_tables,
        replace_table_text,
    )
    end_time = time.time()
    processing_time = end_time - start_time

    file_name = Path(file_path).stem
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{file_name}_elements.json"), "w") as f:
            json.dump(elements, f, indent=2)
        with open(os.path.join(output_dir, f"{file_name}_texts.json"), "w") as f:
            json.dump(texts, f, indent=2)
        with open(os.path.join(output_dir, f"{file_name}_metadata.json"), "w") as f:
            json.dump(metadata_list, f, indent=2)
        with open(os.path.join(output_dir, f"{file_name}_tables.json"), "w") as f:
            json.dump(table_data, f, indent=2)

    return processing_time, elements, (texts, metadata_list), table_data


def process_directory(
    client: UnstructuredAPIClient,
    input_dir: str,
    output_dir: str,
    extract_tables: bool,
    extract_html_tables: bool,
    extract_combined_tables: bool,
    replace_table_text: bool,
) -> pd.DataFrame:
    """
    Processes all files in a directory and saves the output to the specified directory.

    Args:
        client (UnstructuredAPIClient): The API client instance.
        input_dir (str): The directory containing the files to process.
        output_dir (str): The directory to save the output files.
        extract_tables (bool): Whether to extract tables from the document.
        extract_html_tables (bool): Whether to extract tables as HTML.
        extract_combined_tables (bool): Whether to extract all tables combined into a single element.
        replace_table_text (bool): Whether to replace the 'text' element of a table with the 'table_as_html' value.

    Returns:
        pd.DataFrame: A DataFrame containing the processing times for each file.
    """
    file_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]
    results = []
    for file_path in file_paths:
        processing_time, _, _, _ = process_file(
            client,
            file_path,
            output_dir,
            extract_tables,
            extract_html_tables,
            extract_combined_tables,
            replace_table_text,
        )
        results.append({"file_path": file_path, "processing_time": processing_time})
    return pd.DataFrame(results)


def compare_strategies(
    client: UnstructuredAPIClient,
    file_path: str,
    strategies: List[str],
    extract_tables: bool,
    extract_html_tables: bool,
    extract_combined_tables: bool,
    replace_table_text: bool,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compares the processing times for different strategies on a single file.

    Args:
        client (UnstructuredAPIClient): The API client instance.
        file_path (str): The path to the file to process.
        strategies (List[str]): The list of strategies to compare.
        extract_tables (bool): Whether to extract tables from the document.
        extract_html_tables (bool): Whether to extract tables as HTML.
        extract_combined_tables (bool): Whether to extract all tables combined into a single element.
        replace_table_text (bool): Whether to replace the 'text' element of a table with the 'table_as_html' value.
        output_dir (Optional[str]): The directory to save the output files. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the processing times for each strategy.
    """
    results = []
    for strategy in strategies:
        client.config.strategy = strategy
        processing_time, _, _, _ = process_file(
            client,
            file_path,
            output_dir,
            extract_tables,
            extract_html_tables,
            extract_combined_tables,
            replace_table_text,
        )
        results.append({"strategy": strategy, "processing_time": processing_time})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Process files using the Unstructured API"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config YAML file"
    )
    parser.add_argument("--input_dir", type=str, help="Path to the input directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--file_path", type=str, help="Path to a single file to process"
    )
    parser.add_argument(
        "--compare_strategies",
        action="store_true",
        help="Compare processing times for different strategies",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["fast", "hi_res", "ocr_only"],
        help="List of strategies to compare",
    )

    args = parser.parse_args()

    config = UnstructuredAPIConfig.from_yaml(args.config)
    client = UnstructuredAPIClient(config)

    extract_tables = True
    extract_html_tables = True
    extract_combined_tables = True
    replace_table_text = True

    if args.file_path:
        processing_time, elements, (texts, metadata_list), table_data = process_file(
            client,
            args.file_path,
            args.output_dir,
            extract_tables,
            extract_html_tables,
            extract_combined_tables,
            replace_table_text,
        )
        print(f"Processing time for {args.file_path}: {processing_time:.2f} seconds")

    if args.input_dir:
        df_dir = process_directory(
            client,
            args.input_dir,
            args.output_dir,
            extract_tables,
            extract_html_tables,
            extract_combined_tables,
            replace_table_text,
        )
        print("Processing times for directory:")
        print(df_dir)

    if args.compare_strategies:
        if not args.file_path:
            print("Please provide a file path to compare strategies.")
            return

        df_strategies = compare_strategies(
            client,
            args.file_path,
            args.strategies,
            extract_tables,
            extract_html_tables,
            extract_combined_tables,
            replace_table_text,
        )
        print("Processing times for different strategies:")
        print(df_strategies)

        plt.figure(figsize=(8, 6))
        plt.bar(df_strategies["strategy"], df_strategies["processing_time"])
        plt.xlabel("Strategy")
        plt.ylabel
        plt.ylabel("Processing Time (seconds)")
        plt.title("Processing Times for Different Strategies")
        plt.savefig(os.path.join(args.output_dir, "strategy_comparison.png"))


if __name__ == "__main__":
    main()
