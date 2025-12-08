<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>

End to End BYOC and Fine-Tuning SambaStudio Kit
======================

<!-- TOC -->

- [Overview](#overview)
- [End to End BYOC and Fine-Tuning SambaStudio Kit](#end-to-end-byoc-and-fine-tuning-sambastudio-kit)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Install dependencies](#install-dependencies)
        - [Install an set up Snapi and Snsdk](#install-an-set-up-snapi-and-snsdk)
        - [Install requirements](#install-requirements)
- [Bring your own checkpoint](#bring-your-own-checkpoint)
- [Upload a dataset](#upload-a-dataset)
- [Fine tune your model](#fine-tune-your-model)
- [Deploy and do inference over your models](#deploy-and-do-inference-over-your-models)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

This End to End BYOC and Fine-Tuning SambaStudio Kit provides a comprehensive step by step guide for users to bring their own model checkpoints and fine-tune them on their own datasets within the SambaStudio platform. The kit streamlines in a series of [notebooks](./notebooks) the entire workflow into four seamless steps

- Model Upload: Easily upload your own model checkpoints, and update configurations for compatibility with SambaStudio.

- Dataset Preparation: Prepare and upload your datasets to SambaStudio, setting the stage for fine-tuning.

- Training: Execute training jobs using your uploaded model and dataset, and promote the best-generated model for further use.

- Deployment and Inference: Add the fine-tuned checkpoint and base checkpoint to a bundle, deploy it, and perform inference on the model within the SambaStudio platform.

> All these steps will be done programmatically using in behind the sambastudio API trough SNSDK and Snapi packages

# Before you begin

## Clone this repository

Clone the starter kit repo.

``` bash
  git clone https://github.com/sambanova/ai-starter-kit.git
```

## Install dependencies

### Install an set up Snapi and SNSDK
Follow the instructions in the [Snapi and SNSDK installation guide](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html) to install and set up Snapi and SNSDK on your virtual environment.

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

  ``` bash
    cd ai_starter_kit/e2e_fine_tuning
    pip install uv
    uv pip install -r requirements.txt
  ```

# Bring your own checkpoint

This step guides you through the process of uploading a model checkpoint to SambaStudio. For a detailed implementation, please refer to the [1_checkpoints.ipynb](./notebooks/1_checkpoints.ipynb) notebook, which provides a step-by-step guide.

In this step, you will instantiate the [SambaStudio client for BYOC](../utils/byoc/README.md) and configure your model checkpoint, including setting the model name, publisher, description, and parameter count. You can also download a base checkpoint from Hugging Face, if desired.

Additionally, you will need to set a padding token, which is required for training, and optionally define a chat template.

The notebook will walk you through the process of getting the necessary model parameters and identifying suitable SambaStudio apps for your checkpoint.

Finally, upload the checkpoint to SambaStudio, either individually or in a streamlined way using a config file (e.g., [checkpoints_config.yaml](./checkpoints_config.yaml)). See the [notebook](./notebooks/1_checkpoints.ipynb) for a detailed understanding of the implementation.

# Upload a dataset

This step guides you through the process of uploading a dataset to SambaStudio. For a detailed implementation, please refer to the [2_datasets.ipynb](./notebooks/2_datasets.ipynb) notebook, which provides a step-by-step guide.

In this step, you will prepare your dataset for fine-tuning by converting it to a suitable format (hdf5) using the [generative data prep utility](https://github.com/sambanova/generative_data_prep). You can use your own dataset or download an existing one from Hugging Face.

The notebook will walk you through the process of setting dataset configs, including dataset name, description, job types, and apps availability.

Finally, you will upload the dataset to SambaStudio. The notebook also demonstrates how to upload a dataset in a streamlined way using a config file (e.g., [dataset_config.yaml](./dataset_config.yaml)). See the [notebook](./notebooks/2_datasets.ipynb) for a detailed understanding of the implementation.

# Fine tune your model

This step guides you through the process of fine-tuning your model using a dataset in SambaStudio. For a detailed implementation, please refer to the [3_fine_tuning.ipynb](./notebooks/3_fine_tuning.ipynb) notebook, which provides a step-by-step guide.

In this step, you will create a project, set up a training job, and execute the job in sambastudio.
You will also promote the best-performing checkpoint to a new SambaStudio model. The notebook will walk you through the process of setting up the project and job configs, including model and dataset selection, hyperparameter setting, and checkpoint promotion.

The notebook also demonstrates how to fine-tune a model in a streamlined way using a config file (e.g., [finetune_config.yaml](./finetune_config.yaml)). See the [notebook](./notebooks/3_fine_tuning.ipynb) for a detailed understanding of the implementation.

# Deploy and do inference over your models

This final step guides you through the process of including your fine-tuned model and base model in a bundle model in SambaStudio and creating an endpoint for inference. For a detailed implementation, please refer to the [4_deploy.ipynb](./notebooks/4_deploy.ipynb) notebook, which provides a step-by-step guide.

In this step, you will create a project, a bundle model, and an endpoint in SambaStudio. After endpoint deployment you will also retrieve the endpoint details, including endpoint url and the API key. Finally, you can test the inference capabilities of your deployed bundle model. 

The notebook also demonstrates how to deploy a model in a streamlined way using a config file (e.g., [deploy_config.yaml](./deploy_config.yaml)). See the [notebook](./notebooks/4_deploy.ipynb) for a detailed understanding of the implementation.

# Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- huggingface-hub                (version 0.25.2)
- Jinja2                         (version 3.1.4)
- python-dotenv                  (version 1.0.1)
- langchain                      (version 0.3.8)
- langchain-community            (version 0.3.8)
- langchain-core                 (version 0.3.21)
