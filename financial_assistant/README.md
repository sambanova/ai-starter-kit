
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="./../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

SambaNova Financial Assistant
======================

Welcome to the Sambanova Financial Insights application.

- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the inference endpoint, configs and environment variables](#set-up-the-inference-endpoint-configs-and-environment-variables)
    - [Update the Embeddings API information](#update-the-embeddings-api-information)
- [Deploy the starter kit GUI](#deploy-the-starter-kit-gui)
    - [Deployment: Use a virtual environment 3.11 preferred](#workshop-deployment-use-a-virtual-environment-311-preferred)
- [Environment variables](#environment-variables)

<!-- /TOC -->

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
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the inference endpoint, configs and environment variables

The next step is to set up your environment variables to use one of the models available from SambaNova. If you're a current SambaNova customer, you can deploy your models with SambaStudio. If you are not a SambaNova customer, you can self-service provision API endpoints using SambaNova Fast API or Sambaverse. Note that Sambaverse, although freely available to the public, is rate limited and will not have fast RDU optimized inference speeds.

- If using **SambaNova Fast-API** Please follow the instructions [here](../README.md#use-sambanova-fast-api-option-1) for setting up your environment variables.
    Then in the [config file](./config.yaml) set the llm `api` variable to `"fastapi"` and set the `select_expert` config depending on the model you want to use.

- If using **SambaStudio** Please follow the instructions [here](../README.md#use-sambastudio-option-3) for setting up endpoint and your environment variables.
    Then in the [config file](./config.yaml) set the llm `api` variable to `"sambastudio"`, set the `CoE` and `select_expert` configs if using a CoE endpoint.

- If using **Sambaverse** Please follow the instructions [here](../README.md#use-sambaverse-option-2) for getting your api key and setting up your environment variables.
    Then in the [config file](./config.yaml) set the llm `api` variable to `"sambaverse"` and set the `sambaverse_model_name`, and `select_expert` config depending on the model you want to use.

# Deploy the starter kit GUI

We recommend that you run the starter kit in a virtual environment or use a container. 

## Deployment: Use a virtual environment (3.11 preferred)

If you want to use virtualenv or conda environment:

1. Install and update pip.

```
cd ai_starter_kit/
python3 -m venv financial_assistant_env
source financial_assistant_env/bin/activate
pip install --upgrade pip
pip  install  -r  financial_assistant/requirements.txt
```

2. Run the following command:
```
cd financial_assistant/streamlit/
streamlit run app.py --browser.gatherUsageStats false 
```

## Environment variables
Please set up the following environment variables in your virtual environment or container before launching the app.
These can be included in the project `.env` file.

For `FAST-API`:
```
FASTAPI_URL = "https://fast-api.snova.ai/v1/chat/completions"
FASTAPI_API_KEY = "<your-fastapi-api-key>"
```

For the `SEC-EDGAR` functionalities, company name and email are used to form a user-agent of the form:
USER_AGENT: ```<Company Name> <Email Address>```
```
# Your organization
SEC_API_ORGANIZATION="<your organization>"

# Your email address
SEC_API_EMAIL="<name.surname@email_provider.com>"
```

For `Weave` users:
```
WANDB_API_KEY = "<your-wandb-api-key>"
```

## Exit the app
Once you have finished using the app, you can exit the app by clicking on `Exit` at the top of the sidebar.
This will clear the cache.
Otherwise, the cache will be automatically cleared after a predefined time period.
