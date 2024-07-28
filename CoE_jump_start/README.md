<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

SambaNova AI Starter Kits
====================
## Using Composition of Experts (CoE) LLM Router Kit

This repository provides a Python script, a Jupyter Notebook, and a Streamlit app that demonstrate how to route user queries to a specific SambaNova CoE model using a LLM as a router. The implementation offers different approaches for using doing this, including expert mode, simple mode, end-to-end mode with vector database, and bulk evaluation mode.

<!-- TOC -->
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Key dependencies](#key-dependencies)
- [Installation](#installation)
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

https://github.com/sambanova/ai-starter-kit/assets/150964187/9841b787-5222-4c41-8ad5-34bd9b4fe2c5

## Features

This AI starter kit supports the following features:

- Multiple modes of operation: Expert, Simple, E2E with Vector DB, and Bulk Evaluation
- Customizable expert categories and mappings
- Integration with Langchain for document loading, splitting, and retrieval
- Use of Chroma as the vector database for efficient storage and retrieval
- Streamlit app for easy interaction and visualization
- Supports various configuration options through the `config.yaml` file

## Prerequisites

- Python (>3.11.3 and <3.12)
- A [Sambaverse](https://sambaverse.sambanova.ai/) account
- A [SambaStudio](https://docs.sambanova.ai/sambastudio/latest/index.html) account with at least two running endpoints:
  - A Composition of Experts (CoE) model
  - A text embedding model with a batch size larger than 1 (set/view this via the model parameters)

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

1. Clone the main repository and cd into CoE starter kit:

  ```bash
  git clone https://github.com/sambanova/ai-starter-kit.git
  cd CoE_jump_start
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

4. Set up the config.yaml file with your routing, retrieval and supported experts and mapping preferences. For example:
  
    ```yaml
    api: sambastudio
    llm:
      temperature: 0.1
      max_tokens_to_generate: 1024
      sambaverse_model_name: "Mistral/Mistral-7B-Instruct-v0.2"
      samabaverse_select_expert: "Mistral-7B-Instruct-v0.2"
      coe_routing: true

    retrieval:
      chunk_size: 1000
      chunk_overlap: 200
      db_type: "faiss"

    supported_experts_map:
      finance: "Finance expert"
      economics: "Finance expert"
      maths: "Math expert"
      mathematics: "Math expert"
      code generation: "Code expert"
      computer science: "Code expert"
      legal: "Legal expert"
      medical: "Medical expert"
      history: "History expert"
      turkish language: "Turkish expert"
      japanese language: "Japanese expert"
      literature: "Literature expert"
      physics: "Science expert"
      chemistry: "Science expert"
      biology: "Science expert"
      psychology: "Social Science expert"
      sociology: "Social Science expert"
      ```




5. In the root directory of the ai-starter-kit, create a .env file and add the necessary API keys:

```bash
 # NEEDED FOR SAMBAVERSE
SAMBAVERSE_API_KEY="133-adb-you-key-here" # Found in your profile (upper right corner of Sambaverse)
SAMBAVERSE_URL="https://sambaverse.sambanova.ai/" # Adjust as needed

# For below, SambaStudio endpoint URLs follow the format:
# <BASE_URL>/api/predict/generic/<PROJECT_ID>/<ENDPOINT_ID>
# Both the endpoint URL and the endpoint API key can be found by clicking into an endpoint's details page

# NEEDED FOR SAMBASTUDIO COE MODEL
SAMBASTUDIO_BASE_URL="https://yoursambstudio.url"
SAMBASTUDIO_PROJECT_ID="your-samba-studio_coe_model-projectid"
SAMBASTUDIO_ENDPOINT_ID="your-samba-studio-coe_model-endpointid"
SAMBASTUDIO_API_KEY="your-samba-studio-coe_model-apikey"
VECTOR_DB_URL=http://localhost:6333

# NEEDED FOR SAMBASTUDIO EMBEDDINGS MODEL
SAMBASTUDIO_EMBED_BASE_URL="https://yoursambstudio.url"
SAMBASTUDIO_EMBED_PROJECT_ID="your-samba-studio_embedding_model-projectid"
SAMBASTUDIO_EMBED_ENDPOINT_ID="your-samba-studio-embedding_model-endpointid"
SAMBASTUDIO_EMBED_API_KEY="your-samba-studio-embedding_model-apikey"
```


## Installation


Starter kit usage
Using the Python script

Update the config.yaml file with your desired configuration.
Run the use_coe_model.py script with the desired mode:


```bash
python use_coe_model.py <mode> [--query <query>] [--dataset <dataset_path>] [--num_examples <num>]
```

Available modes:

- expert: Get only the expert category for a given query
- simple: Run a simple LLM invoke with routing
- e2e: Run the end-to-end vector database example
- bulk: Run bulk routing evaluation on a dataset

For example:

```bash
python use_coe_model.py simple --query "What is the current inflation rate?"
```

Using the Jupyter Notebook

Open the Coe_LLM_Router.ipynb notebook in Jupyter.
Follow the instructions in the notebook to run each example:


- Expert Mode: Get only the expert category for a given query
- Simple Mode: Run a simple LLM invoke with routing
- E2E Mode with Vector Database: Use vector database for complex queries
- Bulk Evaluation Mode: Evaluate the router's performance on a large dataset


Update the config.yaml file as needed for each example.
Run the notebook cells to execute the code and observe the results.

Using the Streamlit app

Run the Streamlit app:

```bash
cd streamlit
streamlit run app.py
```

Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).
Use the sidebar to select different modes:

Config: Customize the configuration settings
Expert: Get the expert category for a query
Simple: Run a simple query with routing
E2E With Vector DB: Upload a document and run queries against it
Bulk Evaluation: Evaluate the router on a large dataset


Follow the instructions on each page to interact with the CoE LLM Router.

Modes of Operation

- Expert Mode: This mode only returns the expert category for a given query without invoking the expert model.
- Simple Mode: This mode routes the query to the appropriate expert model and returns both the expert category and the model's response.
- E2E Mode with Vector Database: This mode uses a vector database for more complex queries that may require context from multiple documents. It determines the appropriate expert and provides a response based on the document context.
- Bulk Evaluation Mode: This mode is used for evaluating the router's performance on a large dataset of queries. It provides accuracy metrics and a confusion matrix for analysis.

Supported Models
The script, notebook, and Streamlit app support various models for CoE as per SambaStudio support, including:

Mistral-7B-Instruct-v0.2
finance-chat
deepseek-llm-67b-chat
medicine-chat
law-chat

For a full list of supported CoE models, see the Supported Models documentation.
Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.
License
This project is licensed under the Apache 2.0 license. See the LICENSE.md file in the parent folder (AI-STARTER-KIT) for more details.


Acknowledgements

Langchain for the powerful framework for building applications with LLMs.
Sentence Transformers for the state-of-the-art models for generating embeddings.
Chroma for the efficient vector database.
Streamlit for the easy-to-use framework for creating data apps.
Matplotlib and Seaborn for data visualization.