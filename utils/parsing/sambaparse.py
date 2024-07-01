import os
import yaml
import subprocess
import json
import logging
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv
from langchain.docstore.document import Document
from typing import List, Dict, Optional, Tuple, Union, Any

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SambaParse:
    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def run_ingest(
        self,
        source_type: str,
        input_path: Optional[str] = None,
        additional_metadata: Optional[Dict] = None,
    ):
        """
        Runs the ingest process for the specified source type and input path.

        Args:
            source_type (str): The type of source to ingest (e.g., 'local', 'confluence', 'github', 'google-drive').
            input_path (Optional[str]): The input path for the source (only required for 'local' source type).
            additional_metadata (Optional[Dict]): Additional metadata to include in the processed documents.

        Returns:
            Tuple[List[str], List[Dict], List[Document]]: A tuple containing the extracted texts, metadata, and LangChain documents.
        """
        output_dir = self.config["processor"]["output_dir"]

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Delete contents of the output directory using shell command
        del_command = f"rm -rf {output_dir}/*"
        logger.info(f"Deleting contents of output directory: {output_dir}")
        subprocess.run(del_command, shell=True, check=True)

        command = [
            "unstructured-ingest",
            source_type,
            "--output-dir",
            output_dir,
            "--num-processes",
            str(self.config["processor"]["num_processes"]),
        ]

        if self.config["processor"]["reprocess"] == True:
            command.extend(["--reprocess"])

        # Add partition arguments
        command.extend(
            [
                "--strategy",
                self.config["partitioning"]["strategy"],
                "--ocr-languages",
                ",".join(self.config["partitioning"]["ocr_languages"]),
                "--encoding",
                self.config["partitioning"]["encoding"],
                "--fields-include",
                ",".join(self.config["partitioning"]["fields_include"]),
                "--metadata-exclude",
                ",".join(self.config["partitioning"]["metadata_exclude"]),
                "--metadata-include",
                ",".join(self.config["partitioning"]["metadata_include"]),
            ]
        )

        if self.config["partitioning"]["pdf_infer_table_structure"]:
            command.append("--pdf-infer-table-structure")

        if self.config["partitioning"]["skip_infer_table_types"]:
            command.extend(
                [
                    "--skip-infer-table-types",
                    ",".join(self.config["partitioning"]["skip_infer_table_types"]),
                ]
            )

        if self.config["partitioning"]["flatten_metadata"]:
            command.append("--flatten-metadata")

        if source_type == "local":
            if input_path is None:
                raise ValueError("Input path is required for local source type.")
            command.extend(["--input-path", f'"{input_path}"'])

            if self.config["sources"]["local"]["recursive"]:
                command.append("--recursive")
        elif source_type == "confluence":
            command.extend(
                [
                    "--url",
                    self.config["sources"]["confluence"]["url"],
                    "--user-email",
                    self.config["sources"]["confluence"]["user_email"],
                    "--api-token",
                    self.config["sources"]["confluence"]["api_token"],
                ]
            )
        elif source_type == "github":
            command.extend(
                [
                    "--url",
                    self.config["sources"]["github"]["url"],
                    "--git-branch",
                    self.config["sources"]["github"]["branch"],
                ]
            )
        elif source_type == "google-drive":
            command.extend(
                [
                    "--drive-id",
                    self.config["sources"]["google_drive"]["drive_id"],
                    "--service-account-key",
                    self.config["sources"]["google_drive"]["service_account_key"],
                ]
            )
            if self.config["sources"]["google_drive"]["recursive"]:
                command.append("--recursive")
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        if self.config["processor"]["verbose"]:
            command.append("--verbose")

        if self.config["partitioning"]["partition_by_api"]:
            api_key = os.getenv("UNSTRUCTURED_API_KEY")
            partition_endpoint_url = f"{self.config['partitioning']['partition_endpoint']}:{self.config['partitioning']['unstructured_port']}"
            if api_key:
                command.extend(["--partition-by-api", "--api-key", api_key])
                command.extend(["--partition-endpoint", partition_endpoint_url])
                command.extend(["--pdf-infer-table-structure"])
            else:
                raise ValueError(
                    "UNSTRUCTURED_API_KEY environment variable is not set."
                )

        if self.config["partitioning"]["strategy"] == "hi_res":
            if (
                "hi_res_model_name" in self.config["partitioning"]
                and self.config["partitioning"]["hi_res_model_name"]
            ):
                command.extend(
                    [
                        "--hi-res-model-name",
                        self.config["partitioning"]["hi_res_model_name"],
                    ]
                )
            logger.warning(
                "You've chosen the high-resolution partitioning strategy. Grab a cup of coffee or tea while you wait, as this may take some time due to OCR and table detection."
            )

        if self.config["chunking"]["enabled"]:
            command.extend(
                [
                    "--chunking-strategy",
                    self.config["chunking"]["strategy"],
                    "--chunk-max-characters",
                    str(self.config["chunking"]["chunk_max_characters"]),
                    "--chunk-overlap",
                    str(self.config["chunking"]["chunk_overlap"]),
                ]
            )

            if self.config["chunking"]["strategy"] == "by_title":
                command.extend(
                    [
                        "--chunk-combine-text-under-n-chars",
                        str(self.config["chunking"]["combine_under_n_chars"]),
                    ]
                )
            if self.config["chunking"]["chunk_elements"] == True:
                command.extend(
                    [
                        "--chunk-elements",
                    ]
                )

        if self.config["embedding"]["enabled"]:
            command.extend(
                [
                    "--embedding-provider",
                    self.config["embedding"]["provider"],
                    "--embedding-model-name",
                    self.config["embedding"]["model_name"],
                ]
            )

        if self.config["destination_connectors"]["enabled"]:
            destination_type = self.config["destination_connectors"]["type"]
            if destination_type == "chroma":
                command.extend(
                    [
                        "chroma",
                        "--host",
                        self.config["destination_connectors"]["chroma"]["host"],
                        "--port",
                        str(self.config["destination_connectors"]["chroma"]["port"]),
                        "--collection-name",
                        self.config["destination_connectors"]["chroma"][
                            "collection_name"
                        ],
                        "--tenant",
                        self.config["destination_connectors"]["chroma"]["tenant"],
                        "--database",
                        self.config["destination_connectors"]["chroma"]["database"],
                        "--batch-size",
                        str(self.config["destination_connectors"]["batch_size"]),
                    ]
                )
            elif destination_type == "qdrant":
                command.extend(
                    [
                        "qdrant",
                        "--location",
                        self.config["destination_connectors"]["qdrant"]["location"],
                        "--collection-name",
                        self.config["destination_connectors"]["qdrant"][
                            "collection_name"
                        ],
                        "--batch-size",
                        str(self.config["destination_connectors"]["batch_size"]),
                    ]
                )
            else:
                raise ValueError(
                    f"Unsupported destination connector type: {destination_type}"
                )

        command_str = " ".join(command)
        logger.info(f"Running command: {command_str}")
        logger.info(
            "This may take some time depending on the size of your data. Please be patient..."
        )

        subprocess.run(command_str, shell=True, check=True)

        logger.info("Ingest process completed successfully!")

        # Call the additional processing function if enabled
        if self.config["additional_processing"]["enabled"]:
            logger.info("Performing additional processing...")
            texts, metadata_list, langchain_docs = additional_processing(
                directory=output_dir,
                extend_metadata=self.config["additional_processing"]["extend_metadata"],
                additional_metadata=additional_metadata,
                replace_table_text=self.config["additional_processing"][
                    "replace_table_text"
                ],
                table_text_key=self.config["additional_processing"]["table_text_key"],
                return_langchain_docs=self.config["additional_processing"][
                    "return_langchain_docs"
                ],
                convert_metadata_keys_to_string=self.config["additional_processing"][
                    "convert_metadata_keys_to_string"
                ],
            )
            logger.info("Additional processing completed.")
            return texts, metadata_list, langchain_docs


def convert_to_string(value: Union[List, Tuple, Dict, Any]) -> str:
    """
    Convert a value to its string representation.

    Args:
        value (Union[List, Tuple, Dict, Any]): The value to be converted to a string.

    Returns:
        str: The string representation of the value.
    """
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value))
    elif isinstance(value, dict):
        return json.dumps(value)
    else:
        return str(value)


def additional_processing(
    directory: str,
    extend_metadata: bool,
    additional_metadata: Optional[Dict],
    replace_table_text: bool,
    table_text_key: str,
    return_langchain_docs: bool,
    convert_metadata_keys_to_string: bool,
):
    """
    Performs additional processing on the extracted documents.

    Args:
        directory (str): The directory containing the extracted JSON files.
        extend_metadata (bool): Whether to extend the metadata with additional metadata.
        additional_metadata (Optional[Dict]): Additional metadata to include in the processed documents.
        replace_table_text (bool): Whether to replace table text with the specified table text key.
        table_text_key (str): The key to use for replacing table text.
        return_langchain_docs (bool): Whether to return LangChain documents.
        convert_metadata_keys_to_string (bool): Whether to convert non-string metadata keys to string.

    Returns:
        Tuple[List[str], List[Dict], List[Document]]: A tuple containing the extracted texts, metadata, and LangChain documents.
    """
    if os.path.isfile(directory):
        file_paths = [directory]
    else:
        file_paths = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".json")
        ]

    texts = []
    metadata_list = []
    langchain_docs = []

    for file_path in file_paths:
        with open(file_path, "r") as file:
            data = json.load(file)

        for element in data:
            if extend_metadata and additional_metadata:
                element["metadata"].update(additional_metadata)

            if replace_table_text and element["type"] == "Table":
                element["text"] = element["metadata"][table_text_key]

            metadata = element["metadata"].copy()
            if convert_metadata_keys_to_string:
                metadata = {
                    str(key): convert_to_string(value)
                    for key, value in metadata.items()
                }
            for key in element:
                if key not in ["text", "metadata", "embeddings"]:
                    metadata[key] = element[key]
            if "page_number" in metadata:
                metadata["page"] = metadata["page_number"]
            else:
                metadata["page"] = 1

            metadata_list.append(metadata)
            texts.append(element["text"])

        if return_langchain_docs:
            langchain_docs.extend(get_langchain_docs(texts, metadata_list))

        with open(file_path, "w") as file:
            json.dump(data, file, indent=2)

    return texts, metadata_list, langchain_docs


def get_langchain_docs(texts: List[str], metadata_list: List[Dict]) -> List[Document]:
    """
    Creates LangChain documents from the extracted texts and metadata.

    Args:
        texts (List[str]): The extracted texts.
        metadata_list (List[Dict]): The metadata associated with each text.

    Returns:
        List[Document]: A list of LangChain documents.
    """
    return [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(texts, metadata_list)
    ]


def parse_doc_universal(
    doc: str, additional_metadata: Optional[Dict] = None, source_type: str = "local"
) -> Tuple[List[str], List[Dict], List[Document]]:
    """
    Extract text, tables, images, and metadata from a document or a folder of documents.

    Args:
        doc (str): Path to the document or folder of documents.
        additional_metadata (Optional[Dict], optional): Additional metadata to include in the processed documents.
            Defaults to an empty dictionary.
        source_type (str, optional): The type of source to ingest. Defaults to 'local'.

    Returns:
        Tuple[List[str], List[Dict], List[Document]]: A tuple containing:
            - A list of extracted text per page.
            - A list of extracted metadata per page.
            - A list of LangChain documents.
    """
    if additional_metadata is None:
        additional_metadata = {}

    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Join the current directory with the relative path of the config file
    config_path = os.path.join(current_dir, "config.yaml")

    wrapper = SambaParse(config_path)
    texts, metadata_list, langchain_docs = wrapper.run_ingest(
        source_type, input_path=doc, additional_metadata=additional_metadata
    )

    return texts, metadata_list, langchain_docs
