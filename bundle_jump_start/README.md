<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>

SambaNova AI Starter Kits
====================
## Using Composition of Experts (bundle) LLM Router Kit

This repository provides a Python script, a Jupyter Notebook, and a Streamlit app that demonstrate how to route user queries to a specific SambaNova bundle model using an LLM as a router. The implementation offers different approaches for doing this, including expert mode, simple mode, end-to-end mode with vector database, and bulk evaluation mode.

<!-- TOC -->
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Key dependencies](#key-dependencies)
- [Installation](#installation)
- [Configuration](#configuration)
- [Starter kit usage](#starter-kit-usage)
  - [Using the Python script](#using-the-python-script)
  - [Using the Jupyter Notebook](#using-the-jupyter-notebook)
  - [Using the Streamlit app](#using-the-streamlit-app)
- [Modes of Operation](#modes-of-operation)
- [Supported models](#supported-models)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

<!-- /TOC -->

## Features

This AI starter kit supports the following features:

- Multiple modes of operation: Expert, Simple, E2E with Vector DB, and Bulk Evaluation
- Customizable expert categories and mappings
- Integration with Langchain for document loading, splitting, and retrieval
- Use of Chroma as the vector database for efficient storage and retrieval
- Streamlit app for easy interaction and visualization
- Supports various configuration options through the `config.yaml` file

## Set up the models, environment variables and config file

### Set up the generative model

The next step is to set up your environment variables to use one of the inference models available from SambaNova. For this kit you can deploy your models using SambaStudio.

- **SambaStudio (Option 1)**: Follow the instructions [here](../README.md#use-sambastudio-option-2) to set up your endpoint and environment variables.
    Then, in the [config file](./config.yaml), set the llm `api` variable to `"sambastudio"`, and set the `bundle` and `select_expert` configs if you are using a bundle endpoint.

## Key dependencies

_(Installed below)_
- Langchain
- Sentence Transformers
- YAML
- Requests
- Streamlit
- Matplotlib
- Seaborn

## Installation
_These steps assume a Mac/Linux/Unix shell environment. If using Windows, you will need to adjust some commands for navigating folders, activating virtual environments, etc._

1. Clone the main repository and cd into bundle starter kit:

  ```bash
  git clone https://github.com/sambanova/ai-starter-kit.git
  cd bundle_jump_start
  ```

2. (Recommended) Create a virtual environment and activate it:
  
  ```bash
    python<version> -m venv <virtual-environment-name>
    source <virtual-environment-name>/bin/activate
  ```

3. Install the required dependencies:

  ```bash
    pip install -r requirements.txt # With pip
  ```


## Configuration
The config.yaml file is crucial for customizing the behavior of the bundle LLM Router. Here's a breakdown of its main sections:

1. API and LLM Settings:

  ```yaml
  api: sambastudio

  llm: 
      "temperature": 0.0
      "do_sample": False
      "max_tokens_to_generate": 1200
      "bundle": True #set as true if using Sambastudio bundle endpoint
      "select_expert": "Meta-Llama-3.3-70B-Instruct" #set if using SambaStudio bundle llm expert
  ```
  These settings define the API to use and the parameters for the language model.

2. Retrieval Settings:

  ```yaml
  retrieval:
    chunk_size: 1000
    chunk_overlap: 200
    db_type: "faiss"
  ```
  These settings are used for document chunking and vector database configuration.

3. Supported Experts Map:

  ```yaml
  supported_experts_map:
    finance: "Finance expert"
    economics: "Finance expert"
    maths: "Math expert"
  ```


4. Expert Prompt:
The expert_prompt section is a crucial part of the configuration. It defines how the router classifies incoming queries. You can customize this prompt to add new categories or modify the classification logic. For example:

  ```yaml
  expert_prompt: |
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  A message can be classified as only one of the following categories: 'finance', 'economics', 'maths', 'code generation', 'legal', 'medical', 'history', 'turkish language', 'japanese language', 'literature', 'physics', 'chemistry', 'biology', 'psychology', 'sociology' or 'None of the above'.

  Examples for these categories are given below:
  - 'finance': What is the current stock price of Apple?
  - 'economics': Explain the concept of supply and demand.
  # ... (other examples)

  Based on the above categories, classify this message:
  {input}

  Always remember the following instructions while classifying the given statement:
  - Think carefully and if you are not highly certain then classify the given statement as 'Generalist'
  - For 'turkish language' and 'japanese language' categories, the input may be in Turkish or Japanese respectively. Classify based on the language used, not just the content.
  - Always begin your response by putting the classified category of the given statement after '<<detected category>>:'
  - Explain your answer
  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  ```

To add new categories, you would add them to the list of categories, provide examples, and update the classification instructions as needed.

5. bundle Name Map:

  ```yaml
  bundle_name_map:
    Finance expert: "finance-chat"
    Math expert: "deepseek-llm-67b-chat"
    Code expert: "deepseek-llm-67b-chat"
  ```

This crucial section maps the expert categories to specific bundle models. When adding new categories or changing the routing logic, make sure to update this section to map your categories to the appropriate bundle models available in your SambaStudio environment.

Remember to update the .env file in the root directory of the ai-starter-kit with your API keys and endpoints:

  ```bash

# NEEDED FOR SAMBASTUDIO BUNDLE MODEL
SAMBASTUDIO_BASE_URL="https://your-sambastudio.url"
SAMBASTUDIO_PROJECT_ID="your-bundle-model-project-id"
SAMBASTUDIO_ENDPOINT_ID="your-bundle-model-endpoint-id"
SAMBASTUDIO_API_KEY="your-bundle-model-api-key"
VECTOR_DB_URL=http://localhost:6333

# NEEDED FOR SAMBASTUDIO EMBEDDINGS MODEL
SAMBASTUDIO_EMBED_BASE_URL="https://your-sambastudio.url"
SAMBASTUDIO_EMBED_PROJECT_ID="your-embedding-model-project-id"
SAMBASTUDIO_EMBED_ENDPOINT_ID="your-embedding-model-endpoint-id"
SAMBASTUDIO_EMBED_API_KEY="your-embedding-model-api-key"
  ```


## Starter kit usage
Using the Python script
Update the config.yaml file with your desired configuration, then run the use_bundle_model.py script with the desired mode:


### Using the Python script
Update the config.yaml file with your desired configuration, then run the use_bundle_model.py script with the desired mode:
  
  ```bash
  python src/use_bundle_model.py <mode> [--query <query>] [--dataset <dataset_path>] [--num_examples <num>]
  ```

  1. Expert Mode: Get only the expert category for a given query

  ```bash
  python src/use_bundle_model.py expert --query "What is the capital of France?"
  ```

  2. Simple Mode: Run a simple LLM invoke with routing

  ```bash
  python src/use_bundle_model.py simple --query "Explain the concept of supply and demand."
  ```

  3. Bulk Mode: Run bulk routing evaluation on a dataset (For an example JSONL File to test this on see the notebook)

  ```bash
  python src/use_bundle_model.py bulk --dataset path/to/your/dataset.jsonl --num_examples 100
  ```

  The dataset file should be in JSONL (JSON Lines) format, where each line is a valid JSON object containing a 'prompt' and a 'router_label'. Here's an example of how your dataset.jsonl file should be formatted:


  ```json
  {"prompt": "What is the current inflation rate in the United States?", "router_label": "economics"}
  {"prompt": "Solve the quadratic equation x^2 + 5x + 6 = 0", "router_label": "maths"}
  {"prompt": "Write a Python function to find the maximum element in a list", "router_label": "code generation"}
  ```
  Ensure that the 'router_label' values match the categories defined in your config.yaml file.

Note: E2E With Vector DB Is not run via script but is run via the streamlit app detailed in the following section.

## Using the Jupyter Notebook

Open the Coe_LLM_Router.ipynb notebook in Jupyter.
Follow the instructions in the notebook to run each example:

- Expert Mode: Get only the expert category for a given query
- Simple Mode: Run a simple LLM invoke with routing
- E2E Mode with Vector Database: Use vector database for complex queries
- Bulk Evaluation Mode: Evaluate the router's performance on a large dataset


Update the config.yaml file as needed for each example.
Run the notebook cells to execute the code and observe the results.


## Using the Streamlit app

1. Run the Streamlit app:

  ```bash
  cd streamlit
  streamlit run app.py
  ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

3. Use the sidebar to select different modes:

- Config: Customize the configuration settings
- Expert: Get the expert category for a query
- Simple: Run a simple query with routing
- E2E With Vector DB: Upload a document and run queries against it
- Bulk Evaluation: Evaluate the router on a large dataset (visualisations displayed and results stored in results folder)


4. Follow the instructions on each page to interact with the bundle LLM Router.

## Modes of Operation

1. Expert Mode: This mode only returns the expert category for a given query without invoking the expert model.
2. Simple Mode: This mode routes the query to the appropriate expert model and returns both the expert category and the model's response.
3. E2E Mode with Vector Database: This mode uses a vector database for more complex queries that may require context from multiple documents. It determines the appropriate expert and provides a response based on the document context. You can provide a path to your file on the streamlit app via upload.
4. Bulk Evaluation Mode: This mode is used for evaluating the router's performance on a large dataset of queries. It provides accuracy metrics and a confusion matrix for analysis.

## Supported Models
The script, notebook, and Streamlit app support various models for bundle as per SambaStudio support, including:

Mistral-7B-Instruct-v0.2
LLama3-70B-Instruct
finance-chat
deepseek-llm-67b-chat
medicine-chat
law-chat

For a full list of supported bundle models, see the Supported Models documentation and your SambaStudio instances ModelHub.

## Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License
This project is licensed under the Apache 2.0 license. See the LICENSE.md file in the parent folder (ai-starter-kit) for more details.

## Acknowledgements

Langchain for the powerful framework for building applications with LLMs.
Sentence Transformers for the state-of-the-art models for generating embeddings.
Chroma for the efficient vector database.
Streamlit for the easy-to-use framework for creating data apps.
Matplotlib and Seaborn for data visualization.

## Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory.