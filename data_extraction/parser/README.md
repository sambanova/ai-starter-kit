# SambaParse

SambaParse is a Python package that provides a wrapper for the Unstructured API, allowing you to extract and parse data from various file types. It is built on top of the Unstructured API and adds specific parsing capabilities for common use cases such as tables. The package can be used with any type of file and negates the need for file-specific labelling.

## Setup

### Docker Container

To run the Unstructured API using Docker, follow these steps:

NOTE: If self-hosting. Create your own UNSTRUCTURED_API_KEY .env key and  pass it into the docker when you run it. If it's another hosted instance, talk to the owner to be provisioned a key.

1. Make sure you have Docker installed on your machine. If you don't have Docker installed, please refer to the [official Docker installation guide](https://docs.docker.com/get-docker/).
2. Pull the Docker image from the Unstructured API image repository:
    
    ```
    docker pull downloads.unstructured.io/unstructured-io/unstructured-api:latest
    
    ```
    
3. Create an `env` file containing your `UNSTRUCTURED_API_KEY`:
    
    ```
    UNSTRUCTURED_API_KEY=your_api_key
    
    ```
    
4. Run the Docker container, passing the `env` file:
    
    ```
    docker run -p 8000:8000 -it --rm --name unstructured-api --env-file ./env downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0 /bin/bash
    
    ```
    

You can now access the Unstructured API through the Docker container.

### Local Setup

To set up SambaParse locally, follow these steps:

1. Make sure you have Python 3.10.12 installed on your machine.
2. Clone the AI Starter kit repository and cd into the data extraction folder and parser:
    
    ```
    git clone https://github.com/sambanova/ai-starter-kit
    cd data_extraction/parser
    
    ```
    
3. Install the required dependencies:
    
    ```
    pip install -r requirements.txt
    
    ```
    
4. Set the `UNSTRUCTURED_API_KEY` environment variable to your API key

## Usage

SambaParse provides several methods for parsing and extracting data from files:

- `parse(file_path)`: Parses a single file and returns the raw JSON response, a tuple of text elements and metadata, and table data.
- `process_file(file_path, output_dir)`: Processes a single file and saves the output to the specified directory.
- `process_directory(input_dir, output_dir)`: Processes all files in a directory and saves the output to the specified directory.
- `compare_strategies(file_path, strategies)`: Compares the processing times for different strategies on a single file.

To use SambaParse in your own applications, you can simply call the `parse` method with the path to your file:

```python
from samba_parse import UnstructuredAPIClient, UnstructuredAPIConfig

config = UnstructuredAPIConfig.from_yaml("config.yaml")
client = UnstructuredAPIClient(config)

file_path = "path/to/your/file.pdf"
elements, (texts, metadata_list), table_data = client.parse(file_path)

```

The `parse` method returns the raw JSON response from the API, a tuple containing a list of text elements and a list of metadata dictionaries for each element, and a dictionary mapping table names to tuples of texts and metadata.

If you want to inspect the output more closely or save it to files, you can use the `process_file` or `process_directory` methods:

```python
processing_time, elements, (texts, metadata_list), table_data = process_file(
    client,
    file_path,
    output_dir,
    extract_tables=True,
    extract_html_tables=True,
    extract_combined_tables=True,
    replace_table_text=True,
)

```

The `process_file` method returns the processing time, the raw JSON response, a tuple of text elements and metadata, and table data. It also saves the output to the specified directory.

You can also compare the processing times for different strategies using the `compare_strategies` method:

```python
pdf_strategies = compare_strategies(
    client,
    file_path,
    strategies=["fast", "hi_res", "ocr_only"],
    extract_tables=True,
    extract_html_tables=True,
    extract_combined_tables=True,
    replace_table_text=True,
)

```

This method returns a DataFrame containing the processing times for each strategy.

## Features

SambaParse offers the following features:

- Extraction of tables: By setting `extract_tables=True`, SambaParse extracts tables from the document and saves them as separate elements.
- Extraction of table HTML: By setting `extract_html_tables=True`, SambaParse extracts tables as HTML, providing cleaner table representations.
- Extraction of combined tables: By setting `extract_combined_tables=True`, SambaParse extracts all tables combined into a single element.
- Replacement of table text with HTML: By setting `replace_table_text=True`, SambaParse replaces the 'text' element of a table with the 'table_as_html' value.
- Extraction of text elements and metadata: SambaParse returns a tuple containing a list of text elements and a list of metadata dictionaries for each element. This output can be directly fed into Langchain or Llama Index for vector database loading.
- Chunking strategy: The default chunking strategy is 'title', but it can be changed to 'basic'. No further chunking is required.

By extracting tables into their own list, SambaParse enables better quality Q&A on specific data, such as financials.

## Acknowledgements

SambaParse is built on top of the Unstructured API and leverages its capabilities for file parsing. Additional parsing capabilities have been added for common use cases, and more will be added as needed.

