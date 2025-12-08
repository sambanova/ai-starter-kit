<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>

SambaStudio FineTuning Util
======================

<!-- TOC -->

- [SambaStudio FineTuning Util](#sambastudio-finetuning-util)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Install dependencies](#install-dependencies)
        - [Install an set up Snapi and Snsdk](#install-an-set-up-snapi-and-snsdk)
        - [Install requirements](#install-requirements)
- [Using the FineTuning utility](#using-the-finetuning-utility)
    - [Prepare your dataset](#prepare-your-dataset)
    - [Set up the config file](#set-up-the-config-file)
        - [SambaStudio config](#sambastudio-config)
        - [Dataset config](#dataset-config)
        - [Project config](#project-config)
        - [Job config and Model checkpoint config](#job-config-and-model-checkpoint-config)
        - [Endpoint config](#endpoint-config)
    - [Initializing Fine-tuning wrapper](#initializing-fine-tuning-wrapper)
    - [Dataset upload](#dataset-upload)
    - [Project creation](#project-creation)
    - [Training Job creation and checkpoint promotion](#training-job-creation-and-checkpoint-promotion)
    - [Create a new Composite / Bundle model Optional](#create-a-new-composite--bundle-model-optional)
    - [Deploy your model](#deploy-your-model)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview
This fine-tuning utility for SambaStudio allows users to upload their own datasets, and start training jobs in SambaStudio platform. after this users can do inference over these models in SambaStudio platform.

# Before you begin

## Clone this repository

Clone the starter kit repo.

```bash
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

# Using the FineTuning utility

## Prepare your dataset 

You can use your own dataset (see [synthetic data generation util](../synthetic_data_gen/notebooks/quickstart_synthetic_data_gen.ipynb)) or alternatively you can download and use an existing dataset like the ones in[Huggingface datasets](https://huggingface.co/datasets?modality=modality:text&sort=trending), then execute the gen_data_prep_pipeline which will convert your dataset in a suitable format (`hdf5` files) for SambaStudio.

```python
from src import sambastudio_utils
sambastudio_utils.gen_data_prep_pipeline(
    input_files = "<your input dataset path>", # set with your jsonl path or list of jsonl file paths
    output_path = os.path.join(current_dir,"sambastudio_fine_tuning_dataset"), # hdf5 dataset output path
    tokenizer = "meta-llama/Llama-2-7b-hf", # set with the model tokenizer 
    max_seq_length = 4096, # set with the model sequence length
    prompt_keyword = 'prompt', # set with the prompt keyword in your jsonl dataset
    completion_keyword = 'completion', # set with the completion keyword in your jsonl dataset
    )
```

## Set up the config file

Create a config.yaml file with the required parameters. See the example [config.yaml](./config.yaml) and [config_embeddings.yaml](./config_embeddings.yaml) files.

### SambaStudio config
You should set the following parameters

``` yaml
sambastudio:
    snapi_path: "" # leave it blank if you have done the default snapi and snsdk installation
    rdu_arch: "sn40" # RDU architecture
```

### Dataset config
You should set the following parameters

``` yaml
    dataset: 
        dataset_name: "dataset name"
        dataset_description: "dataset description"
        dataset_path: "/Users/my-user/Documents/ai-starter-kit/utils/fine_tuning/sambastudio_fine_tuning_dataset" #absolute pat of your output dataset (dataset preparation output path)
        dataset_apps_availability: 
            - 'Mistral'
            - 'Llama 3'
            - 'Llama 2 with dynamic batching'
            - 'Llama 2 7B'
            - 'Llama 2 70B with dynamic batching'
            - 'Llama 2 70B'
            - 'Llama 2 13B' 
        dataset_job_types:
            - "evaluation"
            - "train"
        dataset_source_type: "localMachine" 
        dataset_language: "english"
        dataset_filetype: "hdf5"
        dataset_url: ""
        dataset_metadata:
            labels_file: ""
            train_filepath: ""
            validation_filepath: ""
            test_filepath: ""
```

> You can check all apps available running the following command:

```python
    sambastudio_wrapper.list_apps()
```

### Project config

Your training jobs and deployed endpoints are organized inside projects, to set/create you will need to set these two parameters in the config file

```yaml
    project: 
        project_name: "example project"
        project_description: "this project will be used to test e2e fine-tuning pipeline implementation"
```

### Job config and Model checkpoint config

For the training job you should set the following parameters:

``` yaml
    job:
        job_name: "snsdk_test_job"
        job_description: "snsdk test training project"
        job_type: "train"
        model: "Llama-2-7b-chat-hf" # base checkpoint to train with
        model_version: "1"
        parallel_instances: 1
        load_state: false
        sub_path: ""
        hyperparams:
            batch_size: 256
            do_eval: False
            eval_steps: 50
            evaluation_strategy: "no"
            learning_rate: 0.00001
            logging_steps: 1
            lr_schedule: "fixed_lr"
            max_sequence_length: 4096
            num_iterations: 100
            prompt_loss_weight: 0.0
            save_optimizer_state: True
            save_steps: 50
            skip_checkpoint: False
            subsample_eval: 0.01
            subsample_eval_seed: 123
            use_token_type_ids: True
            vocab_size: 32000
            warmup_steps: 0
            weight_decay: 0.1
```

And the trained checkpoint to promote params

``` yaml
    model_checkpoint:
    checkpoint_name: "" #set the name of checkpoint to to promote as model in modelhub (after training)
    model_name: "llama2_7b_fine_tuned" # set the name of your trained checkpoint
    model_version: "1"
    model_description: "finetuned llama2_7b model"
    model_type: "finetuned"
```

### Endpoint config

Finally you can set the params required to deploy your fine-tuned model

```yaml
    endpoint:
        endpoint_name: "test-endpoint-sql"
        endpoint_description: "endpoint of finetuned sql model llama2 7b"
        endpoint_instances: 1
        hyperparams: null
```

## Initializing Fine-tuning wrapper

```python
from src.snsdk_wrapper import SnsdkWrapper
sambastudio_wrapper = SnsdkWrapper(config_path=os.path.join(current_dir,"config.yaml"))

```

## Dataset upload

To upload your dataset using the configs in your config file run:

```python
    sambastudio_wrapper.create_dataset()
```

And check their availability running:

```python
    sambastudio_wrapper.list_datasets()
```

## Project creation

To create your  training and deploying project using the configs in your config file run:

```python
    sambastudio_wrapper.create_project()
```

And check their availability running:

```python
    sambastudio_wrapper.create_project()
```

## Training Job creation and checkpoint promotion

To start the training job in sambastudio using the configs in your config file run:

```python
    sambastudio_wrapper.run_training_job()
```

And check the training progress

```python
    sambastudio_wrapper.check_job_progress()
```

When the train job is finished you can list the checkpoints saved during the training job process

```python
    sambastudio_wrapper.list_checkpoints()
```

Select the one with best metrics for your application and update the config file with the checkpoint ID, the run:

```python
    sambastudio_wrapper.promote_checkpoint()
```

Then you can check your model is now available to be used running:

```python
   sambastudio_wrapper.list_models(filter_job_types=["deploy"])
```

## Create a new Composite / Bundle model (Optional)

After promoting your checkpoint you can create a bundle model with you uploaded and existing checkpoints in SambaStudio creating a `composite_model` section in your [config.yaml](./config.yaml) file with the composite `model_name` and `model_list` to include 

``` yaml 
    composite_model:
    model_name:  "MyCompositeModel"
    description: "CoE including fine tuned model"
    rdu_required: 8
    model_list:
        - "llama2_7b"
        - "llama2_7b_fine_tuned"
```

Then run:

```python
  byoc.create_composite_model()
```

## Deploy your model

You can deploy your fine tuned or bundled model in SambaStudio executing the following command

```python
  byoc.create_endpoint()
  byoc.get_endpoint_details()
```

The previous steps are shown in the [usage Notebook](./usage.ipynb), which can be used as a guide for uploading datasets, executing training jobs, and deploying your fine-tuned models.

# Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- huggingface-hub                (version 0.25.2)
- Jinja2                         (version 3.1.4)
- python-dotenv                  (version 1.0.1)
- langchain                      (version 0.3.7)
- langchain-community            (version 0.3.7)
- langchain-core                 (version 0.3.15)