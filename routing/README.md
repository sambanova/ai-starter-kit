<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Routing
======================

Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

Table of Contents:
<!-- TOC -->
- [Routing](#Routing)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the models, environment variables and config file](#set-up-the-models-environment-variables-and-config-file)
        - [Set up the generative model](#set-up-the-generative-model)
        - [Set up the embedding model](#set-up-the-embedding-model)
    - [Windows requirements](#use-the-starter-kit)
    - [Install dependencies](#install-dependencies)
- [Use the starter kit](#use-the-routing)
- [Customizing the starter kit](#customizing-the-starter-kit)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview
This AI Starter Kit is an example of routing a user query to different RAG pipeline or LLM based on keywords from the datasource.

The Kit includes:
- An implementation of a keyword extractor to extract keywords from documents
- An implementation of a workflow to route user query to different pipeline 

# Before you begin

You have to set up your environment before you can run or customize the starter kit. 
<!-- To use this in your application you need an instruction model, we recommend to use the Mistral 7B Instruct or Meta Llama3 8B, either from Sambaverse or from SambaStudio CoE. For embedding model, we recommend to use intfloat/e5-large-v2. -->

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

* **SambaStudio embedding model (Option 2)**: To increase inference speed, you can use a SambaStudio embedding model endpoint instead of using the default (CPU) Hugging Face embedding. Follow the instructions [here](../README.md#use-sambastudio-embedding-option-2) to set up your endpoint and environment variables. Then, in the [config file](./config.yaml), set the variable `type` in `embedding_model` to `"sambastudio"`, and set the configs `batch_size`, `coe` and `select_expert` according to your SambaStudio endpoint.

## Windows requirements

- If you are using Windows, make sure your system has Microsoft Visual C++ Redistributable installed. You can install it from [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and make sure to check all boxes regarding C++ section. (Compatible versions: 2015, 2017, 2019 or 2022)

## Install dependencies

We recommend that you run the starter kit in a virtual environment.

NOTE: python 3.10 or higher is required to use this kit.

Install the python dependencies in your project environment.

    ```bash
    cd ai_starter_kit/routing
    python3 -m venv routing_env
    source routing_env/bin/activate
    pip  install  -r  requirements.txt
    ```

# Use the starter kit 

After you've set up the environment, you can use the starter kit. Follow these steps:

1. Extract keywords from datasource:

    You should create keywords from your datasource and save them in the local.

    The class and main functions are in [keyword_extractior.py](keyword_extractior.py).

2. Load keywords and pass it to the prompt. An example is in [routing.py](routing.py).

# Customizing the starter kit


# Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory.