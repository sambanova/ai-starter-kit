
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

SambaNova Financial Assistant
======================

Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

Welcome to the Sambanova Financial Insights application.

Table of Contents:

- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the inference endpoint, configs and environment variables](#set-up-the-inference-endpoint-configs-and-environment-variables)
    - [Update the Embeddings API information](#update-the-embeddings-api-information)
- [Deploy the starter kit GUI](#deploy-the-starter-kit-gui)
    - [Deployment: Use a virtual environment 3.11 preferred](#workshop-deployment-use-a-virtual-environment-311-preferred)
- [Environment variables](#environment-variables)

# Overview

This app demonstrates the capabilities of large language models (LLMs)
in extracting and analyzing financial data using function calling, web scraping,
and retrieval-augmented generation (RAG).

Use the navigation menu to explore various features including:

- **Stock Data Analysis**: Query and analyze stocks based on
    <a href="https://pypi.org/project/yfinance/" target="_blank">Yahoo Finance</a> data.
- **Stock Database**: Create and query an SQL database based on
    <a href="https://pypi.org/project/yfinance/" target="_blank">Yahoo Finance</a> data.
- **Financial News Scraping**: Scrape financial news articles from 
    <a href="https://uk.finance.yahoo.com/" target="_blank">Yahoo Finance</a> News.
- **Financial Filings Analysis**: Query and analyze financial filings based on 
     <a href="https://www.sec.gov/edgar/search/" target="_blank">SEC EDGAR </a> data.
- **Generate PDF Report**: Generate a PDF report based on the saved answered queries
    or on the whole chat history.
- **Print Chat History**: Print the whole chat history.

# Before you begin

You have to set up your environment before you can run or customize the starter kit.

## Clone this repository

Clone the starter kit repo.

```bash
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the models, environment variables and config file

### Set up the generative model

The next step is to set up your environment variables to use one of the inference models available from SambaNova. You can obtain a free API key through SambaNova Cloud. Alternatively, if you are a current SambaNova customer, you can deploy your models using SambaStudio.

- **SambaNova Cloud (Option 1)**: Follow the instructions [here](../README.md#use-sambanova-cloud-option-1) to set up your environment variables.
    Then, in the [config file](./config.yaml), set the llm `api` variable to `"sncloud"` and set the `select_expert` config depending on the model you want to use.

- **SambaStudio (Option 2)**: Follow the instructions [here](../README.md#use-sambastudio-option-2) to set up your endpoint and environment variables.
    Then, in the [config file](./config.yaml), set the llm `api` variable to `"sambastudio"`, and set the `CoE` and `select_expert` configs if you are using a CoE endpoint.

### Set up the embedding model

You have the following options to set up your embedding model:

* **CPU embedding model (Option 1)**: In the [config file](./config.yaml), set the variable `type` in `embedding_model` to `"cpu"`.
This will make use of `langchain_community.embeddings.sentence_transformer.SentenceTransformerEmbeddings`.

* **SambaStudio embedding model (Option 2)**: To increase inference speed, you can use a SambaStudio embedding model endpoint instead of using the default (CPU) Hugging Face embedding. Follow the instructions [here](../README.md#use-sambastudio-embedding-option-2) to set up your endpoint and environment variables. Then, in the [config file](./config.yaml), set the variable `type` in `embedding_model` to `"sambastudio"`, and set the configs `batch_size`, `coe` and `select_expert` according to your SambaStudio endpoint.

## Windows requirements

- If you are using Windows, make sure your system has Microsoft Visual C++ Redistributable installed. You can install it from [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and make sure to check all boxes regarding C++ section. (Compatible versions: 2015, 2017, 2019 or 2022)

# Deploy the starter kit GUI

We recommend that you run the starter kit in a virtual environment or use a container. We also recommend using Python >= 3.10 and < 3.12.

## Use a virtual environment

If you want to use virtualenv or conda environment:

1. Install and update pip.

    ```bash
    cd ai_starter_kit/financial_assistant
    python3 -m venv financial_assistant
    source financial_assistant/bin/activate
    pip  install  -r  requirements.txt
    ```

2. Run the following command:

    ```bash
        streamlit run streamlit/app.py --browser.gatherUsageStats false 
    ```

## Further settings

For the `SEC-EDGAR` functionalities, company name and email are used to form a user-agent of the form:
USER_AGENT: ```<Company Name> <Email Address>```.

```
# Your organization
SEC_API_ORGANIZATION="<your organization>"

# Your email address
SEC_API_EMAIL="<user@email_provider.com>"
```

For `Weave` users:
```
WANDB_API_KEY = "<your-wandb-api-key>"
```

## Exit the app
Once you have finished using the app, you can exit the app by clicking on `Exit` at the top of the sidebar.
This will clear the cache.
Otherwise, the cache will be automatically cleared after a predefined time period.
