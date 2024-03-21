<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>


YoDa
======================
<p align="left">
  <img src="./docs/YoDa.png" alt="YoDa" width="200">
</p> 

<!-- TOC -->

- [Overview](#overview)
    - [About this kit](#about-this-kit)
- [Workflow](#workflow)
    - [Data generation](#data-generation)
    - [Data preparation](#data-preparation)
    - [Training / Finetuning](#trainign--finetuning)
    - [Evaluation](#evaluation)
- [Getting Started](#getting-started)
    - [Deploy your models in SambaStudio](#deploy-your-models-in-sambastudio)
    - [Get your SambaStudio API key](#get-your-sambastudio-api-key)
    - [Set the starter kit environment](#set-the-starter-kit-environment)
- [Starterkit: Usage](#starterkit-usage)
    - [Data Generation](#data-generation)
        - [To Generate pretraining data](#to-generate-pretraining-data)
        - [To generate finetuning data](#to-generate-finetuning-data)
        - [Both pretraining and fine-tuning data generation](#both-pretraining-and-fine-tuning-data-generation)
    - [Data Preprocessing](#data-preprocessing)
    - [Launching pretraining/finetuning and hosting endpoints on SambaStudio](#launching-pretrainingfinetuning-and-hosting-endpoints-on-sambastudio)
    - [Evaluation](#evaluation)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview
## About this kit
YoDa is an acronym for **Your Data, Your Model**. This project aims to train a Language Model (LLM) using customer's private data. The goal is to compete with general solutions on tasks that are related to the customer's data.

# Workflow 

## Data generation

This phase involves the generation of synthetic data relevant to the customer's domain. Two main data generation methods are employed, which may vary depending on the task requirements:

Pretraining Generation: This method generates a JSONL file containing sections of the provided data. It will enable the model to do completion over queries.

Finetuning Generation: Utilizing a powerful LLM `Llama 2 70B` and a pipeline composed of prompting and postprocessing techniques, this step processes each document to create a series of synthetic questions and answers based on the content. The generated data is stored in JSONL files, this will teach the model to follow instructions and solve questions beyond mere completion.

## Data preparation

Data preparation involves preprocessing and formatting the generated data to make it suitable for training. This step transforms the data into the required format and structure necessary for training the large language model.

## Training / Finetuning

In this stage, the large language model is finetuned in SambaStudio using your data. Finetuning includes updating the model's parameters to adapt it to the specific characteristics and patterns present in the prepared dataset.

## Evaluation

The evaluation phase create a set of responses to assesses the performance of the finetuned language model on relevant queries. 

It involves using the set of evaluation queries for:

- Obtaining responses from a baseline model.
- Obtaining responses from your custom model.
- Obtaining responses from your custom model giving them in the exact context used in question generation of the evaluation queries.
- Obtaining responses from your custom model employing a simple RAG pipeline for response generation.

This will facilitate further analysis of your model's effectiveness in solving the domain specific tasks.

# Getting Started

These instructions will guide you on how to generate training data, preprocess it, train the model, launch the online inference service, and evaluate it.

## Deploy your models in SambaStudio

Begin by deploying a powerful LLM (e.g. Llama 2 70B chat) to an endpoint for inference in SambaStudio either through the GUI or CLI, as described in the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).

Then deploy your baseline model (e.g. Llama 2 7B) to an endpoint for inference in SambaStudio either through the GUI or CLI

## Get your SambaStudio API key
>Optional 
In this Starter kit you can use the Sambanova SDK `SKSDK` to run training inference jobs in SambaStudio, you will only need to set your environment API Authorization Key (The Authorization Key will be used to access to the API Resources on SambaStudio), the steps for getting this key is described [here](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html#_acquire_the_api_key)

## Set the starter kit environment

1. Clone repo.
    ```bash
    git clone https://github.com/sambanova/ai-starter-kit.git
    ```


2. Update API information for the SambaNova LLM and your environment [sambastudio key](#get-your-sambastudio-api-key). 
    
    These are represented as configurable variables in the environment variables file in the root repo directory **```sn-ai-starter-kit/.env```**. For example, a Llama70B chat endpoint with the URL

    "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"

    a Lama7B baseline model with the URL 
    
    "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/987654ef-fedc-9876-1234-01fedbac9876"

    and a samba studio key ```"1234567890abcdef987654321fedcba0123456789abcdef"```
    would be entered in the environment file (with no spaces) as:
    ```yaml
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

3. Install requirements.
    It is recommended to use virtualenv or conda environment for installation, and to update pip.
    ```bash
    cd ai-starter-kit/yoda
    python3 -m venv yoda_env
    source/yoda_env/bin/activate
    pip install -r requirements.txt
    ```

4. Download your dataset and update
the `src_folder` variable in your [sn expert config file](./sn_expert_conf.yaml) with the path of the folder and sub folders in `src_subfolders`, for including your own data follow the same step.

5. Optionally Download and install Sambanova SNSDK.
    Follow the instructions in this [guide](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html) for installing Sambanova SNSDK and SNAPI, (you can omit the *Create a virtual environment* step since you are using the just created ```yoda_env``` environment)

6. Download the [Samabnova data preparation repository](https://github.com/sambanova/generative_data_prep)
   ```bash
    deactivate
    cd ../..
    git clone https://github.com/sambanova/generative_data_prep
    cd generative_data_prep
    python3 -m venv generative_data_prep_env
    source/generative_data_prep_env/bin/activate
   ```

    Then follow the [installation guide](https://github.com/sambanova/generative_data_prep?tab=readme-ov-file#installation)

# Starterkit: Usage

## Data Generation 

For Domain adaptive pre-training and Instruction Finetune data generation run une of the following scripts

> Note: You will need a SambaStudio endpoint to the LLAMA 70B Chat model and add the configurations to your env file, which is used for synthetic data generation.

> you should have requested access to the meta Llama2 tokenizer and have a [local copy](https://llama.meta.com/llama-downloads/) or [Hugging Face model granted access](https://huggingface.co/meta-llama/Llama-2-70b), then put the path of the tokenizer or name of the HF model in the config file

Please replace the value of --config param with your actual config file path. An example config is shown in `./sn_expert_conf.yaml`
and this is set as the default parameter for the data generation scripts below.

> set in your config file the `dest_folder`, `tokenizer` and `n_eval_samples` parameters

Activate your YoDa starter kit  environment

```bash
deactivate
cd ../..
cd ai-starter-kit/yoda
source/yoda_env/bin/activate
```

### To Generate pretraining data
```bash
python -m src/gen_data.py
    --config ./sn_expert_conf.yaml
    --purpose pretrain 
```

### To generate finetuning data

```bash
python src/gen_data.py
    --config ./sn_expert_conf.yaml
    --purpose finetune 
```

### Both pretraining and fine tuning data generation
```bash
python -m src.gen_data
    --config ./sn_expert_conf.yaml
    --purpose both 
```

## Data Preprocessing
In order to pretrain and finetune on SambaStudio,
we fist need the data to be in the format of hdf5 files that we can upload as dataset in SambaStudio
To preprocess the data, open `scripts/preprocess.sh` and replace
the variables `ROOT_GEN_DATA_PREP_DIR` with the path to your [generative data preparation](https://github.com/sambanova/generative_data_prep)
directory, set absolute path of the output JSONL from [pretraining/finetuning](#data-generation-1) In the `INPUT_FILE` parameter of the `scripts/preprocess.sh; and 
an `OUTPUT_DIR` where you want your hdf5 files to be dumped before you upload them to 
SambaStudio Datasets.

Activate the generative_data_prep_env 

```bash
deactivate
source ../../generative_data_prep_env/bin/activate
```

Then run the script

```bash
sh scripts/preprocess.sh
```

## Launching pretraining/finetuning and hosting endpoints on SambaStudio

Then is needed to create and host your model checkpoints which needs to be done on SambaStudio. 
This can be done on the [**SambaStudio GUI**](https://docs.sambanova.ai/sambastudio/latest/dashboard.html) following the next steps

1. First upload your generated Dataset from [gen_data_prep](#data-preparation) step

2. Create a [project](https://docs.sambanova.ai/sambastudio/latest/projects.html)

3. Run a [training job](https://docs.sambanova.ai/sambastudio/latest/training.html) 

4. [Create an endpoint](https://docs.sambanova.ai/sambastudio/latest/endpoints.html) for your trained model

5. Add the endpoint details to the ```.env``` file, now your .env file should look like this:
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

This training process can also be done as well as with **snapapi** and **snapsdk**. If you are 
interested in how this done via **SNSDK**, please have a look at the WIP [notebook](./notebooks/SambaStudio_job_spinup.ipynb) using the yoda env

## Evaluation

For our evaluation, we pose the finetuned model questions from the held-out synthetic question-answer pairs we procured
when we were generating the finetuning data. We benchmark the approach against responses we get from also using RAG as well as from 
a golden context.

Reactivate Activate the YoDa env

```bash
deactivate 
source yoda_env/bin/activate
```

To assess the trained model, execute the following script:

```bash
python src/evaluate.py 
    --config sn_expert_conf.yaml
```

Please replace  `--config` parameter with your actual config file path.

# Third-party tools and data sources
All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- scikit-learn  (version 1.4.1.post1)
- jsonlines  (version 4.0.0)
- transformers (version4.33)
- wordcloud  (version 1.9.3)
- sacrebleu  (version 2.4.0)
- datasets  (version 2.18.0)
- sqlitedict  (version 2.1.0)
- accelerate  (version 0.27.2)
- omegaconf  (version 2.3.0)
- evaluate  (version 0.4.1)
- pycountry  (version 23.12.11)
- rouge_score  (version 0.1.2)
- parallelformers  (version 1.2.7)
- peft  (version 0.9.0)
- plotly (version 5.18.0)
- langchain (version 0.1.2)
- pydantic (version1.10.13)
- python-dotenv (version 1.0.0)
- sseclient (version 0.0.27)
