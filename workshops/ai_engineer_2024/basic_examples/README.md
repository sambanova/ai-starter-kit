<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../../../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Basic Examples
====================

# Overview

These are basic examples to get you started

# Setup instructions

1. Clone this repo:
```
git clone https://github.com/sambanova/ai-starter-kit.git 
```

2. Set your SambaStudio API key
   - In the repo root directory, find or create the `.env` config file in `ai-starter-kit/.env` and specify the SambaStudio API key and endpoint info (with no spaces). For example:
    ``` bash
    SAMBASTUDIO_BASE_URL="https://sjc3-e2.sambanova.net/"
    SAMBASTUDIO_PROJECT_ID="348281f6-4c62-4c39-b15a-4a9e3a9bbfef"
    SAMBASTUDIO_ENDPOINT_ID="cca1567d-0426-4967-9037-8255dee33f4d"
    SAMBASTUDIO_API_KEY="62096281-a7a3-48cd-8af0-54a6fd82158b"
    ```

2. Update `pip` and install dependencies. It is recommended to use a virtual env or conda environment for installation. For example: 
```
cd ai-starter-kit/workshops/ai_engineer_2024/basic_examples
conda create -n basic_ex python=3.10
conda activate basic_ex
pip  install  -r  requirements.txt
```

4. Open the Jupyter Notebook `test_sambastudio.ipynb` and start executing the cells
