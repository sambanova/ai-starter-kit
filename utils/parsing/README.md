# SambaParse

SambaParse is a Python library that simplifies the process of extracting and processing unstructured data using the Unstructured.io API. It provides a convenient wrapper around the Unstructured.io CLI tool, allowing you to ingest data from various sources, perform partitioning, chunking, embedding, and load the processed data into a vector database. It's designed to be used within AI Starter kits and SN Apps, unifying our data ingestion and document intelligence platform. This allows us to keep our code base centralized for data ingestion kits.

## Prerequisites

Before using SambaParse, make sure you have the following:

- Docker installed on your machine (or access to another API server)
- An Unstructured.io API key

Before using SambaParse, make sure you have the following:

- Create a `.env` file in the ai-starter-kit root directory (not in the parsing folder root):

     ```bash
     UNSTRUCTURED_API_KEY=your_api_key_here
     ```

## Setup

### Pre Reqs

Using pyenv to manage virtualenv's is recommended
Mac install instructions. See pyenv-virtualenv repo for more detailed instructions.

  ```bash
  brew install pyenv-virtualenv
  ```

- Create a python venv using python version 3.10.12
 
  ```bash
  pyenv install 3.10.12
  pyenv  virtualenv 3.10.12 sambaparse
  pyenv activate sambaparse
  ```

- Clone the ai-starter-kit repo and cd:

  ```bash
  git clone https://github.com/sambanova/ai-starter-kit
  ```

- cd into utils/parsing and pip install the requirements

  ```bash
  pip install -r requirements.txt
  ```

- cd into the unstructured-api foder and Install the unstructured-api make-file:

  ```bash
  cd  unstructured-api
  ```

- Run 
   
  ```bash
  make install
  ```

- Run The Web Server:

  ```bash
  make run-web-app
  ```

  This script will start the Unstructured API server using the specified API key and expose it on port 8005.

  - Alternatively, if you have another Unstructured API server running on a different instance, make sure to update the `partition_endpoint` and `unstructured_port` values in the YAML configuration file accordingly.

## Usage

1. Import the `SambaParse` class from the `ai-starter-kit` library:

    ```python
    from utils.parsing.sambaparse import SambaParse
    ```

2. Create a YAML configuration file (e.g., `config.yaml`) to specify the desired settings for the ingestion process. Here's the configuration for use cases 1 and 2 ie local files and folders:

    ```yaml
    processor:
    verbose: True
    output_dir: './output'
    num_processes: 2

    sources:
      local:
        recursive: True
      confluence:
        api_token: 'your_confluence_api_token'
        user_email: 'your_email@example.com'
        url: 'https://your-confluence-url.atlassian.net'
      github:
        url: 'owner/repo'
        branch: 'main'
      google_drive:
        service_account_key: 'path/to/service_account_key.json'
        recursive: True
        drive_id: 'your_drive_id'

    partitioning:
      pdf_infer_table_structure: True
      skip_infer_table_types: []
      strategy: 'auto'
      hi_res_model_name: 'yolox'
      ocr_languages: ['eng']
      encoding: 'utf-8'
      fields_include: ['element_id', 'text', 'type', 'metadata', 'embeddings']
      flatten_metadata: False
      metadata_exclude: []
      metadata_include: []
      partition_endpoint: 'http://localhost'
      unstructured_port: 8005
      partition_by_api: True

    chunking:
      enabled: True
      strategy: 'basic'
      chunk_max_characters: 1500
      chunk_overlap: 300

    embedding:
      enabled: False
      provider: 'langchain-huggingface'
      model_name: 'intfloat/e5-large-v2'

    destination_connectors:
      enabled: False
      type: 'chroma'
      batch_size: 80
      chroma:
        host: 'localhost'
        port: 8004
        collection_name: 'snconf'
        tenant: 'default_tenant'
        database: 'default_database'
      qdrant:
        location: 'http://localhost:6333'
        collection_name: 'test'

    additional_processing:
      enabled: True
      extend_metadata: True
      replace_table_text: True
      table_text_key: 'text_as_html'
      return_langchain_docs: True
      convert_metadata_keys_to_string: True
    ```

    Make sure to place the `config.yaml` file in the desired folder.

3. Create an instance of the `SambaParse` class, passing the path to the YAML configuration file:

    ```python
    sambaparse = SambaParse('path/to/config.yaml')
    ```

4. Use the `run_ingest` method to process your data:

- For a single file:

    ```python
    source_type = 'local' 
    input_path = 'path/to/your/file.pdf' 
    additional_metadata = {'key': 'value'} 
    texts, metadata_list, langchain_docs = sambaparse.run_ingest(source_type, input_path=input_path, additional_metadata=additional_metadata)
    ```

  - For a folder:

    ```python
    source_type = 'local' 
    input_path = 'path/to/your/file.pdf' 
    additional_metadata = {'key': 'value'} 
    texts, metadata_list, langchain_docs = sambaparse.run_ingest(source_type, input_path=input_path, additional_metadata=additional_metadata)
    ```

  - For Confluence:

    ```python
    source_type = 'confluence' 
    additional_metadata = {'key': 'value'} 
    texts, metadata_list, langchain_docs = sambaparse.run_ingest(source_type, additional_metadata=additional_metadata)
      ```
      
    Note that for conflence you must enable embedding and destinatation connectors automatically ie Chroma and turn off additional processing (ie langchain), an example yaml to do that is below

      ```yaml
        processor:
        verbose: True
        output_dir: './output'
        num_processes: 2

        sources:
          local:
            recursive: True
          confluence:
            api_token: 'your_confluence_api_token'
            user_email: 'your_email@example.com'
            url: 'https://your-confluence-url.atlassian.net'
          github:
            url: 'owner/repo'
            branch: 'main'
          google_drive:
            service_account_key: 'path/to/service_account_key.json'
            recursive: True
            drive_id: 'your_drive_id'

        partitioning:
          pdf_infer_table_structure: True
          skip_infer_table_types: []
          strategy: 'auto'
          hi_res_model_name: 'yolox'
          ocr_languages: ['eng']
          encoding: 'utf-8'
          fields_include: ['element_id', 'text', 'type', 'metadata', 'embeddings']
          flatten_metadata: False
          metadata_exclude: []
          metadata_include: []
          partition_endpoint: 'http://localhost'
          unstructured_port: 8005
          partition_by_api: True

        chunking:
          enabled: True
          strategy: 'basic'
          chunk_max_characters: 1500
          chunk_overlap: 300

        embedding:
          enabled: True
          provider: 'langchain-huggingface'
          model_name: 'intfloat/e5-large-v2'

        destination_connectors:
          enabled: True
          type: 'chroma'
          batch_size: 80
          chroma:
            host: 'localhost'
            port: 8004
            collection_name: 'snconf'
            tenant: 'default_tenant'
            database: 'default_database'
          qdrant:
            location: 'http://localhost:6333'
            collection_name: 'test'

        additional_processing:
          enabled: False
          extend_metadata: True
          replace_table_text: True
          table_text_key: 'text_as_html'
          return_langchain_docs: True
          convert_metadata_keys_to_string: True
      ```

    In addition for confluence you will need to have a Chroma Server running on port 8004, you can do this by running the docker command below

      ```bash
      docker run -d --rm --name chromadb -v ./chroma:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=TRUE -p 8004:8000 chromadb/chroma:latest
      ```
      
  The `run_ingest` method returns a tuple containing the extracted texts, metadata, and LangChain documents (if `return_langchain_docs` is set to `True` in the configuration).
      
5. Process the returned data as needed:
    - `texts`: A list of extracted text elements from the documents.
    - `metadata_list`: A list of metadata dictionaries for each text element.
    - `langchain_docs`: A list of LangChain `Document` objects, which combine the text and metadata.

  #### Configuration Options

  The YAML configuration file allows you to customize various aspects of the ingestion process. Here are some of the key options:

  - `processor`: Settings related to the processing of documents, such as the output directory and the number of processes to use.
  - `sources`: Configuration for different data sources, including local files, Confluence, GitHub, and Google Drive.
  - `partitioning`: Options for partitioning the documents, including the strategy, OCR languages, and API settings.
  - `chunking`: Settings for chunking the documents, such as enabling chunking, specifying the chunking strategy, and setting the maximum chunk size and overlap.
  - `embedding`: Options for embedding the documents, including enabling embedding, specifying the embedding provider, and setting the model name.
  - `additional_processing`: Configuration for additional processing steps, such as extending metadata, replacing table text, and returning LangChain documents.

  Make sure to review and modify the configuration file according to your specific requirements.