# SambaParse

SambaParse is a Python library that simplifies the process of extracting and processing unstructured data using the Unstructured.io API. It provides a convenient wrapper around the Unstructured.io CLI tool, allowing you to ingest data from various sources, perform partitioning, chunking, embedding, and load the processed data into a vector database. It's designed to be used within AI Starter kits and SN Apps, unifying our data ingestion and document intelligence platform. This allows us to keep our code base centralized for data ingestion kits.

## Prerequisites

Before using SambaParse, make sure you have the following:

- Docker installed on your machine (or access to another API server)
- An Unstructured.io API key





## Prerequisites

Before using SambaParse, make sure you have the following:

- Create a `.env` file in the root directory of the project and add your Unstructured.io API key:

     ```
     UNSTRUCTURED_API_KEY=your_api_key_here
     ```

## Setup


1. Install the required dependencies:



   ```bash
   pip install git+https://github.com/sambanova/ai-starter-kit
   ```

2. Start the local Unstructured API server:

   

   - Run Docker Compose Up (Ensure you have docker compose installed) (run from the parsing dir):

     ```bash
    docker-compose up
     ```

     This script will start the Unstructured API container using the specified API key and expose it on the custom port defined in the YAML configuration file (default: 8005).

   - Alternatively, if you have another Unstructured API server running on a different instance, make sure to update the `partition_endpoint` and `unstructured_port` values in the YAML configuration file accordingly.

## Usage

1. Import the `SambaParse` class from the `ai-starter-kit` library:

   ```python
   from ai-starter-kit.utils.parsing.sambaparse import SambaParse
   ```

2. Create a YAML configuration file (e.g., `config.yaml`) to specify the desired settings for the ingestion process. Here's an example configuration:

   ```yaml
   processor:
     verbose: True
     output_dir: 'output'
     num_processes: 2

   sources:
     local:
       recursive: True

   partitioning:
     partition_by_api: True

   chunking:
     enabled: True
     strategy: 'basic'
     chunk_max_characters: 1500
     chunk_overlap: 300

   embedding:
     enabled: True
     provider: 'langchain-huggingface'
     model_name: 'your_model_name'

   additional_processing:
     enabled: True
     extend_metadata: True
     replace_table_text: True
     table_text_key: 'text_as_html'
     return_langchain_docs: True

Make sure to place the `config.yaml` file in the desired folder.

3. Create an instance of the `SambaParse` class, passing the path to the YAML configuration file:

```python
sambaparse = SambaParse('path/to/config.yaml')
```

4. Use the `run_ingest` method to process your data:
    - For a single file:
        
       
        
        `source_type = 'local' input_path = 'path/to/your/file.pdf' additional_metadata = {'key': 'value'} texts, metadata_list, langchain_docs = sambaparse.run_ingest(source_type, input_path=input_path, additional_metadata=additional_metadata)`
        
    - For a folder:
        
      
        
        `source_type = 'local' input_path = 'path/to/your/folder' additional_metadata = {'key': 'value'} texts, metadata_list, langchain_docs = sambaparse.run_ingest(source_type, input_path=input_path, additional_metadata=additional_metadata)`
        
    - For a data source like Confluence:
        
      
        
        `source_type = 'confluence' additional_metadata = {'key': 'value'} texts, metadata_list, langchain_docs = sambaparse.run_ingest(source_type, additional_metadata=additional_metadata)`
        
    
    The `run_ingest` method returns a tuple containing the extracted texts, metadata, and LangChain documents (if `return_langchain_docs` is set to `True` in the configuration).
    
5. Process the returned data as needed:
    - `texts`: A list of extracted text elements from the documents.
    - `metadata_list`: A list of metadata dictionaries for each text element.
    - `langchain_docs`: A list of LangChain `Document` objects, which combine the text and metadata.

Configuration Options

The YAML configuration file allows you to customize various aspects of the ingestion process. Here are some of the key options:

- `processor`: Settings related to the processing of documents, such as the output directory and the number of processes to use.
- `sources`: Configuration for different data sources, including local files, Confluence, GitHub, and Google Drive.
- `partitioning`: Options for partitioning the documents, including the strategy, OCR languages, and API settings.
- `chunking`: Settings for chunking the documents, such as enabling chunking, specifying the chunking strategy, and setting the maximum chunk size and overlap.
- `embedding`: Options for embedding the documents, including enabling embedding, specifying the embedding provider, and setting the model name.
- `additional_processing`: Configuration for additional processing steps, such as extending metadata, replacing table text, and returning LangChain documents.

Make sure to review and modify the configuration file according to your specific requirements.

