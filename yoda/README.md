<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>


YoDa
======================

<!-- TOC -->

- [Overview](#overview)
- [Workflow overview](#workflow-overview)
- [Getting Started](#getting-started)
    - [Deploy a SambaStudio inference endpoint](#deploy-a-sambastudio-inference-endpoint)
    - [Get your SambaStudio API key](#get-your-sambastudio-api-key)
    - [Set the starter kit environment](#set-the-starter-kit-environment)
- [Starterkit: Usage](#starterkit-usage)
    - [Data preparation](#data-preparation)
        - [Generate pretraining data](#generate-pretraining-data)
        - [Generate finetuning data](#generate-finetuning-data)
        - [Generate both pretraining and fine-tuning data](#generate-both-pretraining-and-finetuning-data)
    - [Preprocess the data](#preprocess-the-data)
    - [Perform pretraining/finetuning and host endpoints on SambaStudio](#perform-pretrainingfinetuning-and-host-endpoints-on-sambastudio)
    - [Evaluation](#evaluation)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

YoDa is an acronym for **Your Data, Your Model**. This starter kit aims to train a Language Model (LLM) using private data. The goal is to compete with general solutions on tasks that are related to the private data.

# Workflow overview

When you work with YoDa, you'll go through several phases until you arrive at a trained and tested model. 

1. **Data generation**. Generation of synthetic data relevant to your domain. Two main data generation methods, which vary depending on the task requirements, can be used:
   * Pretraining Generation: Generate a JSONL file containing sections of the provided data. Enables the model to do completion over queries.
   * Finetuning Generation: Process each document to create a series of synthetic questions and answers based on the content. This method uses a powerful LLM (Llama 2 70B) and a pipeline composed of prompting and postprocessing techniques. The generated data is stored in JSONL files. This method teaches the model to follow instructions and answer questions.
2. **Data preparation**. Preprocessing and formatting the generated data to make it suitable for training. This step transforms the data into the required format and structure necessary for training the large language model.
3. **Training / Finetuning**. In this stage, you fine tune the model in SambaStudio using your data. Finetuning includes updating the model's parameters to adapt it to the specific characteristics and patterns present in the prepared dataset.
4. **Evaluation**. The evaluation phase creates a set of responses to assess the performance of the finetuned language model. It involves using the set of evaluation queries for:
   * Obtaining responses from a baseline model.
   * Obtaining responses from your custom model.
   * Obtaining responses from your custom model giving them in the exact context used in question generation of the evaluation queries.
   * Obtaining responses from your custom model employing a simple RAG pipeline for response generation.
Evaluation facilitates further analysis of your model's effectiveness in solving the domain specific tasks.

# Getting Started

These instructions will guide you on how to generate training data, preprocess it, train the model, launch the online inference service, and evaluate it.

## Deploy a SambaStudio inference endpoint

SambaStudio includes a rich set of open source models that have been customized to run efficiently on RDU. Deploy the LLM of choice (e.g. Llama 2 13B chat, etc) to an endpoint for inference in SambaStudio either through the GUI or CLI. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html). 

## Get your SambaStudio API key
>Optional 
In this Starter kit you can use the SambaNova SDK `SKSDK` to run training inference jobs in SambaStudio, you will only need to set your environment API Authorization Key (The Authorization Key will be used to access to the API Resources on SambaStudio), the steps for getting this key is described [here](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html#_acquire_the_api_key)

## Set the starter kit environment

1. Clone the repo.
    ```bash
    git clone https://github.com/sambanova/ai-starter-kit.git
    ```
2. Update the LLM API information for SambaStudio. 
     (Step 1) Update the environment variables file in the root repo directory `sn-ai-starter-kit/.env` to point to the SambaStudio endpoint. For example, for an endpoint with the URL "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef012 update the env file (with no spaces) as:
   ```
   BASE_URL="https://api-stage.sambanova.net"
   PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
   ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
   API_KEY="89abcdef-0123-4567-89ab-cdef01234567"

   YODA_BASE_URL="https://api-stage.sambanova.net"
   YODA_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
   BASELINE_ENDPOINT_ID="987654ef-fedc-9876-1234-01fedbac9876"
   BASELINE_API_KEY="12fedcba-9876-1234-abcd76543"

   SAMBASTUDIO_KEY="1234567890abcdef987654321fedcba0123456789abcdef"
   ```
(Step 2) In the [config file](./config.yaml) file, set the variable `api` to `"sambastudio"`    
3. (Optional) Set up a virtual environment. 
    We recommend that you use virtualenv or a conda environment for installation and run `pip update`. 
    ```bash
    cd ai-starter-kit/yoda
    python3 -m venv yoda_env
    source/yoda_env/bin/activate
    pip install -r requirements.txt
    ```
4. Download your dataset and update
the path to the data source folder in `src_folder` and the list of subfolders in the `src_subfolders` variable in your [sn expert config file](./sn_expert_conf.yaml) . The dataset structure consists of the `src_folder` (str) which contains one or more subfolders that represent a different file. Each subfolder should contain at least one
txt file containing the content of that file. The txt files will be used as context retrievals for RAG. We have added an illustration of the data structure in the `data` folder which acts as our `src_folder` and `['sambanova_resources_blogs','sambastudio']` which are our `src_subfolders`.

5. (Optional) Download and install SambaNova SNSDK.
    Follow the instructions in this [guide](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html) for installing Sambanova SNSDK and SNAPI, (you can skip the *Create a virtual environment* step since you are using the ```yoda_env``` environment you just created).

6. Clone the [SambaNova data preparation repository](https://github.com/sambanova/generative_data_prep)
   ```bash
    deactivate
    cd ../..
    git clone https://github.com/sambanova/generative_data_prep
    cd generative_data_prep
    python3 -m venv generative_data_prep_env
    source/generative_data_prep_env/bin/activate
   ```
7. Install the data prep tools following the [installation instructions](https://github.com/sambanova/generative_data_prep?tab=readme-ov-file#installation).

# Starterkit Usage



## Data preparation

Prerequisites for data generation: 
1. Follow the steps above to set up a SambaStudio endpoint to the LLAMA 70B Chat model and add to update the env file.  
2. Request access to the [Meta Llama2 tokenizer](https://huggingface.co/meta-llama/Llama-2-70b) or [download a copy](https://llama.meta.com/llama-downloads/), then put the path of the tokenizer or name of the Hugging Face model in the config file.
3. Replace the value of `--config param` with your actual config file path. An example config is shown in `./sn_expert_conf.yaml`
and this is set as the default parameter for the data generation scripts below.
4. In your config file, set the `dest_folder`, `tokenizer` and `n_eval_samples` parameters.
5. Activate your YoDa starter kit  environment

```bash
deactivate
cd ../..
cd ai-starter-kit/yoda
source/yoda_env/bin/activate
```

### Generate pretraining data

To generate pretraining data, run this script: 

```bash
python src/gen_data.py  --config ./sn_expert_conf.yaml --purpose pretrain 
```

### Generate finetuning data

To generate finetuning data, run this script:
```bash
python src/gen_data.py --config ./sn_expert_conf.yaml --purpose finetune 
```

### Generate both pretraining and finetuning data
Run this script: 
```bash
python src.gen_data --config ./sn_expert_conf.yaml --purpose both 
```

## Preprocess the data

To pretrain and finetune on SambaStudio, the data must be hdf5 files that you can upload to SambaStudio as dataset.

To preprocess the data:
1. open `scripts/preprocess.sh`
2. Replace the variables `ROOT_GEN_DATA_PREP_DIR` with the path to your [generative data preparation](https://github.com/sambanova/generative_data_prep)
directory. Also note that `PATH_TO_TOKENIZER` is the path to either a downloaded tokenizer or the huggingface name of
the model. For example, `meta-llama/Llama-2-7b-chat-hf`. 
> Note: if you want only to pre-train the JSON to use as input is article_data.jsonl,
> if you used finetune as --purpose ,the JSON to use as input is synthetic_qa_train.jsonl 
> if you want to do both in the same training job ,the JSON to use as input is qa_article_mix.jsonl

3. In `scripts/preprocess.sh`, set the `INPUT_FILE` parameter to the absolute path of the output JSONL from [pretraining/finetuning](#data-generation-1) and 
set `OUTPUT_DIR` to the location where you want your hdf5 files to be dumped before you upload them to 
SambaStudio Datasets.
4. Activate `generative_data_prep_env`: 

```bash
deactivate
source ../../generative_data_prep_env/bin/activate
```
5. Then run the script to preprocess the data. 
```bash
sh scripts/preprocess.sh
```

## Perform pretraining/finetuning and host endpoints on SambaStudio

In SambaStudio, you need to create and host your model checkpoints. Connect to the [**SambaStudio GUI**](https://docs.sambanova.ai/sambastudio/latest/dashboard.html) and follow these steps:

1. Upload your generated dataset from [gen_data_prep](#preprocess-the-data) step.

2. Create a [project](https://docs.sambanova.ai/sambastudio/latest/projects.html).

3. Run a [training job](https://docs.sambanova.ai/sambastudio/latest/training.html) .
4. [Create an endpoint](https://docs.sambanova.ai/sambastudio/latest/endpoints.html) for your trained model.

5 Add the endpoint details to the ```.env``` file. Now your .env file should look like this:
    ```yaml

    BASE_URL="https://api-stage.sambanova.net"
    PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
    API_KEY="89abcdef-0123-4567-89ab-cdef01234567"

    YODA_BASE_URL="https://api-stage.sambanova.net"
    YODA_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    BASELINE_ENDPOINT_ID="987654ef-fedc-9876-1234-01fedbac9876"
    BASELINE_API_KEY="12fedcba-9876-1234-abcd76543"

    #finetuned model endpoint details
    FINETUNED_ENDPOINT_ID="your endpoint ID"
    FINETUNED_API_KEY="your endpoint API key"

    SAMBASTUDIO_KEY="1234567890abcdef987654321fedcba0123456789abcdef"
    ```

## Evaluation

For evaluation, you can ask the finetuned model questions from the synthetic question-answer pairs that you procured
when you were generating the finetuning data. You benchmark the approach against responses we get from also using RAG as well as from 
a golden context.

Reactivate the YoDa environment:

```bash
deactivate 
source yoda_env/bin/activate
```

To assess the trained model, run the following script, passing in your config file:

```bash
python src/evaluate.py 
    --config <sn_expert_conf.yaml>
```

# Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory.
