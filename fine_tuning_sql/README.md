<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>

Fine-tuning SQL
======================

Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

<!-- TOC -->

- [Fine-tuning SQL](#fine-tuning-sql)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Install dependencies](#install-dependencies)
- [Use the starter kit](#use-the-starter-kit)
    - [Data download](#data-download)
    - [Data preparation](#data-preparation)
    - [Basic QA-QC](#basic-qa-qc)
    - [Dataset Uploading and Training](#dataset-uploading-and-training)
    - [Inference](#inference)
        - [Hosting](#hosting)
        - [Inference Pipeline](#inference-pipeline)
    - [Benchmarking](#benchmarking)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

This Starter Kit is an example of LLM fine-tuning process leveraging SambaStudio platform, this workflow shows how to fine-tune an SQL model for Question-Answering purpose, enhancing SQL generation tasks performance. The Kit includes:

- A Jupyter Notebook for downloading pre-training and fine-tuning SQL datasets
- A detailed in Notebook guide for generating the training files
- A Notebook for quality control and evaluation of the generated training files
- A guide on uploading datasets and fine-tuning a model of choice using the SambaStudio graphical user interface
- A Notebook for performing inference with the trained model

# Before you begin

You have to set up your environment before you can run the starter kit.

## Clone this repository

Clone the starter kit repo.

```bash
git clone --recurse-submodules  https://github.com/sambanova/ai-starter-kit.git
```

## Install dependencies

We recommend that you run the starter kit in a virtual environment

```bash
cd ai-starter-kit/
git submodule update --init.  
cd fine_tuning_sql
python3 -m venv fine_tuning_sql_env
source fine_tuning_sql_env/enterprise_knowledge_env/bin/activate
pip  install  -r  requirements.txt
```

Then login with your hugging face account in your terminal through [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

# Use the starter kit 

This starter kit is covered on the jupyter notebooks in the [notebooks folder](notebooks/), you can sequentially follow them to do the complete fine-tuning and pretraining process, from downloading the datasets to hosting and using your trained model.

## Data download
Follow the notebook [1_download_data.ipynb](notebooks/1_download_data.ipynb) to download and store pre-training and fine-tuning datasets.

> You will need to request access to each of the example datasets in the notebook in their HuggingFace datasets page.

## Data preparation

Follow the Notebook [2_data_preparation.ipynb](notebooks/2_data_preparation.ipynb) to do the data preparation step in which the downloaded data of the previous steps will be converted to .hdf5 files, which will be used as dataset for SambaStudio training jobs

## Basic QA-QC

One can do basic QA-QC by loading the HDF5 and jsonl files as shown in the notebook [3_qa_data.ipynb](notebooks/3_qa_data.ipynb).

## Dataset Uploading and Training

You will find comprehensive guide of how to upload an train your models in the notebook [4_upload_and_train.ipynb](notebooks/4_upload_and_train.ipynb)

## Inference

### Hosting

The final fine-tuned model can then be hosted on SambaStudio. Once hosted, the API information, including environmental variables such as BASE_URL, Base URI, PROJECT_ID, ENDPOINT_ID, and API_KEY, can be utilized to execute inference, se more details on how to host your model [here](../README.md#getting-a-sambanova-api-key-and-setting-your-models).

### Inference Pipeline

The notebook [5_inference__model.ipynb](notebooks/5_inference__model.ipynb) uses the fine-tuned model in langchain to generate a SQL query from user input, execute the query against the database, and finally generate a final answer.

## Benchmarking

The [Archerfix repository](https://github.com/archerfish-bench/benchmark) can be used to benchmark your fine-tuned SQL model

# Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory.
