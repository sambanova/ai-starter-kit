<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

SambaNova AI Starter Kits
====================
## Using Composition of Experts (CoE) Models with Langchain

This repository provides a Python script and a Jupyter Notebook that demonstrate how to call SambaNova CoE models using the Langchain framework. The script offers different approaches for calling CoE models, including using Sambaverse, using SambaStudio with a named expert, and using SambaStudio with routing.

<!-- TOC -->
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Key dependencies](#key-dependencies)
- [Installation](#installation)
- [Starter kit usage](#starter-kit-usage)
  - [Calling CoE models with the Python script](#calling-coe-models-with-the-python-script)
  - [Using the Jupyter Notebook](#using-the-jupyter-notebook)
- [Supported models](#supported-models)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

<!-- /TOC -->

https://github.com/sambanova/ai-starter-kit/assets/150964187/9841b787-5222-4c41-8ad5-34bd9b4fe2c5

## Features

This AI starter kit supports the following features:

- Call CoE models using Sambaverse, providing the expert name and API key.
- Call CoE models using SambaStudio with a named expert.
- Call CoE models using SambaStudio with routing, automatically determining the appropriate expert based on the user query.
- Integrate with Langchain for document loading, splitting, and retrieval.
- Use Chroma as the vector database for efficient storage and retrieval.
- Supports various configuration options through the `config.yaml` file.

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

## Installation
_These steps assume a Mac/Linux/Unix shell environment. If using Windows, you will need to adjust some commands for navigating folders, activating virtual environments, etc._

1. Clone the main repository and cd into CoE starter kit:

  ```bash
  git clone https://github.com/sambanova/ai-starter-kit.git
  cd CoE_jump_start
  ```

2a. Create a venv:  

  ```bash
  python<version> -m venv <virtual-environment-name>
  ```

2b. - Install the requirements via pip: 

  ```bash
  - With pip: `pip install -r requirements.txt` 
  - With poetry: `poetry install --no-root`
  ```





3. Set up the config.yaml file with your API credentials and preferences. For example:
=======
2. (Recommended) Create a virtual environment and activate it: 
  ```bash
  python<version> -m venv <virtual-environment-name>
  source <virtual-environment-name>/bin/activate
  ```

3. Install the required dependencies:
  ```bash
  pip install -r requirements.txt # With pip
  poetry install --no-root # With poetry
  ```

4. Set up the `config.yaml` file with your API credentials and preferences. For example:

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
  ```

4. In the CoE starter kit root directory (that is, in the parent folder to where this `README` is saved), create a `.env` file and add the necessary API keys based on your chosen entry point:

```env
# NEEDED FOR SAMBAVERSE
SAMBAVERSE_API_KEY="133-adb-you-key-here" # Found in your profile (upper right corner of Sambaverse)
SAMBAVERSE_URL="https://sambaverse.sambanova.ai/" # Adjust as needed

# For below, SambaStudio endpoint URLs follow the format:
# <BASE_URL>/api/predict/generic/<PROJECT_ID>/<ENDPOINT_ID>
# Both the endpoint URL and the endpoint API key can be found by clicking into an endpoint's details page

# NEEDED FOR SAMBASTUDIO COE MODEL
BASE_URL="https://yoursambstudio.url"
PROJECT_ID="your-samba-studio_coe_model-projectid"
ENDPOINT_ID="your-samba-studio-coe_model-endpointid"
API_KEY="your-samba-studio-coe_model-apikey"
VECTOR_DB_URL=http://localhost:6333

# NEEDED FOR SAMBASTUDIO EMBEDDINGS MODEL
EMBED_BASE_URL="https://yoursambstudio.url"
EMBED_PROJECT_ID="your-samba-studio_embedding_model-projectid"
EMBED_ENDPOINT_ID="your-samba-studio-embedding_model-endpointid"
EMBED_API_KEY="your-samba-studio-embedding_model-apikey"
```

The script supports both Sambaverse and SambaStudio APIs. Depending on which API you want to use, provide the corresponding API keys and URLs in the `.env` file. If you want to use the `SNSDK` instead of `requests`, make sure you have it installed and configured with your credentials.

## Starter kit usage

### Calling CoE models with the python script

1. Update the `config.yaml` file with your desired configuration.

2. Run the `use_coe_model.py` script:

  ```bash
  python use_coe_model.py
  ``` 

  The script will load documents, create a vector database, set up the language model based on the configuration, and invoke the retrieval chain with a user query.

### Using the Jupyter Notebook

1. Open the `calling_coe_models.ipynb` notebook in Jupyter.

2. Follow the instructions in the notebook to run each example:

- Example 1: Using Sambaverse to call CoE Model
- Example 2: Using SambaStudio to call CoE with Named Expert
- Example 3: Using SambaStudio to call CoE with Routing

3. Update the `config.yaml` file as needed for each example.

4. Run the notebook cells to execute the code and observe the results.

## Supported Models

The script and notebook support various models for calling CoE, including:

* Mistral-7B-Instruct-v0.2
* finance-chat
* deepseek-llm-67b-chat
* medicine-chat
* law-chat

For a full list of supported CoE models, see the [Supported Models](https://docs.sambanova.ai/sambastudio/latest/samba-1.html#_samba_1_expert_models) documentation.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the Apache 2.0 license. See the `LICENSE.md` file in the parent folder (`AI-STARTER-KIT`) for more details.

## Acknowledgements

- [Langchain](https://github.com/hwchase17/langchain) for the powerful framework for building applications with LLMs.
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for the state-of-the-art models for generating embeddings.
- [Chroma](https://github.com/chroma-core/chroma) for the efficient vector database.
