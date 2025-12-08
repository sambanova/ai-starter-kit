<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../images/light-logo.png" height="100">
  <img alt="SambaNova logo" src="../images/dark-logo.png" height="100">
</picture>
</a>

BYOC (Bring Your Own Checkpoint)
======================

<!-- TOC -->

- [BYOC Bring Your Own Checkpoint](#byoc-bring-your-own-checkpoint)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Install dependencies](#install-dependencies)
        - [Install an set up Snapi and Snsdk](#install-an-set-up-snapi-and-snsdk)
        - [Install requirements](#install-requirements)
- [Using the BYOC utility](#using-the-byoc-utility)
    - [Get your checkpoint](#get-your-checkpoint)
    - [Set up the config file](#set-up-the-config-file)
    - [Update tokenizer_config chat templates](#update-tokenizer_config-chat-templates)
    - [Initializing BYOC](#initializing-byoc)
    - [Check your chat template](#check-your-chat-template)
    - [Finding config parameters and suitable app IDs](#finding-config-parameters-and-suitable-app-ids)
    - [Upload your checkpoint](#upload-your-checkpoint)
    - [Create a new composite model Bundle](#create-a-new-composite-model-bundle)
    - [Deploy your model](#deploy-your-model)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview
The BYOC (Bring Your Own Checkpoint) utility allows users to upload their local or downloaded checkpoints to SambaStudio platform. This enables users to train and do inferences over these models in SambaStudio platform, as well as include them in bundle models as experts.

# Before you begin

## Clone this repository

Clone the starter kit repo.

``` bash
  git clone https://github.com/sambanova/ai-starter-kit.git
```

## Install dependencies

### Install an set up Snapi and Snsdk
Follow the instructions in the [Snapi and Snsdk installation guide](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html) to install and set up Snapi and Snsdk on your virtual environment.

### Set your SambaStudio variables
- Create the `.env` file at `ai-starter-kit/.env` if the file does not exist.
- Set your SambaStudio variables:

``` bash
SAMBASTUDIO_HOST_NAME="https://www.environment.com" # set with your environment URL (without `/` at the end)
SAMBASTUDIO_ACCESS_KEY="**************" # set with your generated access key
SAMBASTUDIO_TENANT_NAME="default" # Set with your tenant name
```

### Install requirements
Install the python dependencies in your previously created environment.

    ```bash
      cd ai_starter_kit/utils/byoc
      pip install uv
      uv pip install -r requirements.txt
    ``` 

# Using the BYOC utility

The following steps are shown in the [usage Notebook](./usage.ipynb), which can be used as a guide for downloading, uploading, and deploying your own checkpoints

## Get your checkpoint

You can use your own fine-tuned models or alternatively  you can download and use [Huggingface model checkpoints](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) for this you should download the checkpoints to an specific location :

```python
import os
from huggingface_hub import hf_hub_download, HfApi

model_name = "meta/Hermes-2-Theta-Llama-3-8B"  # Replace with your desired model

target_dir = "./models"  
os.makedirs(target_dir, exist_ok=True)

repo_files = HfApi().list_repo_files(model_name)

for file_name in repo_files:
    hf_hub_download(repo_id=model_name, filename=file_name, cache_dir=target_dir)
```

## Set up the config file

Create a config.yaml file with the required parameters. See the example [config.yaml](./config.yaml) file.

> Note: If you have downloaded checkpoints from HuggingFace, inside [./models](./models/) directory you will see a directory with the model name, inside you will see a `./snapshots` folder with a subfolder for the snapshot, set the path of these subfolder as `checkpoint_path` in the [config.yaml](./config.yaml) file.

## Update tokenizer_config chat templates

inside your checkpoint folder you will find a `tokenizer_config.json` file, there you will update the chat template to use jinja2 format, this is an example of how the `chat_template`key should look for a model with this chat format

><|begin_of_text|><|im_start|>system
This is a system prompt.<|im_end|>
<|im_start|>user
This is a user prompt.<|im_end|>
<|im_start|>assistant
This is a response from the assistant.<|im_end|>
<|im_start|>user
This is an user follow up<|im_end|>
<|im_start|>assistant

```json
  "chat_template": "{% for message in messages %}{% set content = '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>'+'\n' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}<|im_start|>assistant\n"
```

> see more about jinja2 templates [here](https://realpython.com/primer-on-jinja-templating/#get-started-with-jinja)

## Initializing BYOC

```python 
from utils.byoc.src.snsdk_byoc_wrapper import BYOC
byoc = BYOC("./config.yaml")
```

## Check your chat template

You can write an example chat template and look how it will be parsed using the previously set jinja templates for each checkpoint in the [config.yaml](./config.yaml)  

```python
  test_messages = [
      {"role": "system", "content": "This is a system prompt."},
      {"role": "user", "content": "This is a user prompt."},
      {"role": "assistant", "content": "This is a response from the assistant."},
      {"role": "user", "content": "This is an user follow up"}
      ]
  byoc.check_chat_templates(test_messages)
```

Output
```
  <|begin_of_text|><|im_start|>system
  This is a system prompt.<|im_end|>
  <|im_start|>user
  This is a user prompt.<|im_end|>
  <|im_start|>assistant
  This is a response from the assistant.<|im_end|>
  <|im_start|>user
  This is an user follow up<|im_end|>
  <|im_start|>assistant
```

## Finding config parameters and suitable app IDs

In checkpoint configs you will need to provide the `model_arch`, `seq_length`,  `vocab_size` and `app_id` if you are not sure of those values you can run the following commands

```python
  byoc.find_config_params()
```

this will return the `model_arch`, `seq_length`,  `vocab_size` in the checkpoint config. then update those values in your [config.yaml](./config.yaml)  

To get the suitable Sambastudio app IDs run the following command

```python
  byoc.get_suitable_apps()
```
Select the app from the returned list ad update their id in `chekpoints` config

## Upload your checkpoint

When checkpoints config is set you can upload the checkpoints with the following command

```python 
  byoc.upload_checkpoints()
```

> Note: this step can take a while depending on the size of your checkpoints and your network speed, you can set the max amount of retries if upload fails with the retries parameter

you can check the status of the model running

```python
  byoc.get_checkpoints_status()
```

## Create a new composite model (Bundle)

After uploading your checkpoints you can create a bundle model with you uploaded and existing checkpoints in SambaStudio updating the `composite_model` section in your [config.yaml](./config.yaml) file with the composite `model_name` and `model_list` to include then run:

```python
  byoc.create_composite_model()
```

## Deploy your model

You cna deploy your bundle model in SambaStudio updating the details in `project`, and `endpoint` sections in your [config.yaml](./config.yaml) an the executing the following command

```python
  byoc.create_project()
  byoc.create_endpoint()
  byoc.get_endpoint_details()
```

# Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- huggingface-hub                (version 0.25.2)
- Jinja2                         (version 3.1.4)
- python-dotenv                  (version 1.0.1)
- langchain                      (version 0.3.7)
- langchain-community            (version 0.3.7)
- langchain-core                 (version 0.3.15)