
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Fine-tuning SQL
======================

Questions? Just <a href="https://discord.gg/XF5Sf2sa" target="_blank">message us</a> on Discord <a href="https://discord.gg/XF5Sf2sa" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

<!-- TOC -->

- [Fine-tuning SQL](#fine-tuning-sql)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Install dependencies](#install-dependencies)
- [Use the starter kit](#use-the-starter-kit)
    - [Data download](#data-download)
    - [Data preparation](#data-preparation)
        - [Pretraining data prep](#pretraining-data-prep)
        - [Fine-tuning data prep](#fine-tuning-data-prep)
    - [Basic QA-QC](#basic-qa-qc)
    - [Load the dataset on Sambastudio](#load-the-dataset-on-sambastudio)
    - [Training](#training)
        - [Pre-training](#pre-training)
        - [Fine-tuning](#fine-tuning)
    - [Hosting](#hosting)
    - [Inference](#inference)
    - [Benchmarking](#benchmarking)

<!-- /TOC -->

# Overview

This Starter Kit is an example of LLM fine-tuning process leveraging SambaStudio platform, this workflow shows how to fine-tune an SQL model for Question-Answering purpose, enhancing SQL generation tasks performance. The Kit includes:

- A Jupyter Notebook for downloading pre-training and fine-tuning SQL datasets
- A detailed guide for generating the training files
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
cd ai_starter_kit/
git submodule update --init.  
cd fine_tuning_sql
python3 -m venv fine_tuning_sql_env
source fine_tuning_sql_env/enterprise_knowledge_env/bin/activate
pip  install  -r  requirements.txt
```

# Use the starter kit 

## Data download
Follow the notebook [1_download_data.ipynb](notebooks/1_download_data.ipynb) to download and store pre-training and fine-tuning datasets.

> You will need to request access to each of the example datasets in the notebook in their HuggingFace datasets page, and install and login with the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) in your terminal

## Data preparation

Clone the [generative_data_prep](https://github.com/sambanova/generative_data_prep) github repo and follow the instructions. The repo required jsonl files produced in [Data download](#data-download) section above.

Below are the example commands used to prepare the data

### Pretraining data prep

Sample command:
```python
python -m generative_data_prep pipeline \
--input_file_path={input_path}/pretrain-squad-smol-sql.jsonl \
--output_path={output_path} \
--pretrained_tokenizer=meta-llama/Llama-2-7b-hf \
--max_seq_length=4096 \
--shuffle=on_RAM \
--keep_split_jsonls
```

### Fine-tuning data prep

Sample command:
```python
python -m generative_data_prep pipeline \
--input_file_path={input_path}/fine-tune-nstext2_test.jsonl \
--output_path={output_path} \
--pretrained_tokenizer=meta-llama/Llama-2-7b-hf \
--max_seq_length=4096 \
--shuffle=on_RAM \
--input_packing_config=greedy::drop \
--keep_split_jsonls
```

## Basic QA-QC

One can do basic QA-QC by loading the HDF5 and jsonl files as shown in the notebook [2_qa_data.ipynb](notebooks/2_qa_data.ipynb).

## Load the dataset on Sambastudio

Once the data preparation is done the datasets can be uploaded directly to SambaStudio.
Please refer to the [SambaStudio documentation for uploading datasets](https://docs.sambanova.ai/sambastudio/latest/add-datasets.html) 

## Training

### Pre-training
Once the datasets are uploaded, one can pre-train and fine-tune base-models on these datasets.
We use **Llama7B Base** as the starting model for further training. Below is a snapshot showing the hyperparameters
for the training job.

![](images/Pretraining_SN.png)

### Fine-tuning
We start with a continuously pretrained model to do further fine-tuning. Below is a snapshot showing the hyperparameters
for the fine-tuning job.

![](images/Fine_tuning_SN.png)

The training loss curve for the fine-tuning job is shown below.

![](images/Fine_tuning_loss_SN.png)

## Hosting

The final fine-tuned model can then be hosted on SambaStudio. Once hosted, the API information, including environmental variables such as BASE_URL, PROJECT_ID, ENDPOINT_ID, and API_KEY, can be utilized to execute inference.

## Inference

The notebook [3_inference__model.ipynb](notebooks/3_inference__model.ipynb) uses the fine-tuned model in langchain to generate a SQL query from user input and then executes the query against the database.

## Benchmarking

The [Archerfix repository](https://github.com/archerfish-bench/benchmark) can be used to benchmark your fine-tuned SQL model

# Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

langchain (version 0.2.6)
langchain-community (version 0.2.6)
transformers (version 4.41.2)
datasets (version 2.20.0)
jupyter_client (version 8.6.0)
jupyter_core (version 5.7.1)
jupyterlab-widgets (version 3.0.9)
SQLAlchemy (version 2.0.30)