<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../../../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Basic Examples
====================

# Overview

In this quickstart, we will show you how to load environment variables, initialize the LLM, and do a simple LLM call via SambaNova with LangChain in Python. We provide the following Jupyter notebooks:
- `example_with_sambastudio.ipynb` (**recommended for the workshop**): using SambaStudio API key.
- `example_with_sambaverse.ipynb`: using Sambaverse API key.
  
# Setup and run instructions

1. Clone this repo:
```
git clone https://github.com/sambanova/ai-starter-kit.git 
```

2. Set your environment variables
   - In the repo root directory, find or create the `.env` config file in `ai-starter-kit/.env` and specify the SambaStudio API key and endpoint info (to be provided during the workshop). For example:
    ``` bash
    SAMBASTUDIO_BASE_URL="https://sjc3-e2.sambanova.net/"
    SAMBASTUDIO_PROJECT_ID="348281f6-4c62-4c39-b15a-4a9e3a9bbfef"
    SAMBASTUDIO_ENDPOINT_ID="cca1567d-0426-4967-9037-8255dee33f4d"
    SAMBASTUDIO_API_KEY="62096281-a7a3-48cd-8af0-54a6fd82158b"
    ```
    - Alternatively, you can create a Sambaverse account at [Sambaverse](https://sambaverse.sambanova.ai/) and get your [Sambaverse API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) (from the user button). Then, add this key to the same `.env` config file in `ai-starter-kit/.env`. For example: 
    ``` bash
    SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
    ``` 

2. Update `pip` and install dependencies. It is recommended to use a virtual env or conda environment for installation, and Python version 3.10.11 or higher. For example: 
```
cd ai-starter-kit/workshops/ai_engineer_2024/basic_examples
conda create -n basic_ex python=3.10.11
conda activate basic_ex
pip  install  -r  requirements.txt
```

4. Open the Jupyter notebook `example_with_sambastudio.ipynb` if you're using a SambaStudio API key and start executing the cells. Alternatively, open `example_with_sambaverse.ipynb` if you're using a Sambaverse API key.
