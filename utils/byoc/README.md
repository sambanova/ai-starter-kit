<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

BYOC (Bring Your Own Checkpoint)
======================

# Overview
The BYOC (Bring Your Own Checkpoint) utility allows users to upload their local or downloaded checkpoints to SambaStudio platform. This enables users to train and do inferences over these models in SambaStudio platform, as well as include them in bundle models as experts.

# Before you begin

## Clone this repository

Clone the starter kit repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Install dependencies

### Install an set up Snapi and Snsdk
Follow the instructions in the [Snapi and Snsdk installation guide](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html) to install and set up Snapi and Snsdk on your virtual environment.

### Install requirements
Install the python dependencies in your previously created environment.

    ```bash
      cd ai_starter_kit/utils/byoc
      pip install -r requirements.txt
    ``` 

# Using the BYOC utility

## Get your checkpoint

Yo can use your own fine-tuned models or alternatively  you can download and use [Huggingface model checkpoints](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) for this you should download the checkpoints to an specific location :

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

## Initializing BYOC

```python 
from src.snsdk_byoc_wrapper import BYOC
byoc = BYOC("./config.yaml")
```

