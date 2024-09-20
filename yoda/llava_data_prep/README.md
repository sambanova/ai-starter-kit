
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

LlaVa Synthetic Data Preparation (In Development)
======================

Table of Contents:

<!-- TOC -->
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Install LaTex](#install-latex)
    - [Clone this repository](#clone-this-repository)
    - [Set up the models, environment variables and config file](#set-up-the-models-environment-variables-and-config-file)
        - [Setup the generative model](#set-up-the-generative-model)
- [Windows requirements](#windows-requirements)
- [Workflow](#workflow)
- [Scripts](#scripts)
    - [Iteration for table formatting/templating](#iteration-for-table-formattingtemplating)
    - [Generate synthetic tables](#generate-synthetic-tables)
- [Future work](#future-work)
- [Current testing](#current-testing)

# Overview

This is a starterkit to demonstrate and provide assistance with the creation of synthetic tables that can be stylized as needed to match a target domain.  The tables will be created with slightly modified tsv format inputs ("*" is used as a placeholder for tabbed row text).  Agents with preset system prompts with instructions for modifying the contents of the tables, creation of pseudo user prompts for table OCR and pseudo user prompts for table QA question pairs are also implemented for a variety of tasks.  It is highly recommended to use Llama 3.1 70b and above for these agents and sufficient results are not gauranteed otherwise.  Table augmentations for table formatting is implemented and more control over the augmentations will be coming in future releases.  Image augmentations using the [albumentations](https://albumentations.ai/) library is used to provide further augmentations, post table-image generations.  

# Before you begin

You have to set up your environment before you can run or customize the starter kit.

## Install LaTex

FOr mac users, you will need a LaTex system dependency:

```bash
brew install --cask mactex
```

## Clone this repository

Clone the starter kit repo.

```bash
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the models, environment variables and config file

### Set up the generative model

**We should not use cloud for this workflow, but until 3.1 is on Studio with matching accuracy, then we have to for now**
- **SambaNova Cloud (Option 1)**: Follow the instructions [here](../README.md#use-sambanova-cloud-option-1) to set up your environment variables.
    Then, in the [config file](./config.yaml), set the llm `api` variable to `"sncloud"` and set the `select_expert` config depending on the model you want to use.

The next step is to set up your environment variables to use one of the inference models available from SambaNova. If you are a current SambaNova customer, you can deploy your models using SambaStudio.

- **SambaStudio**: In the [config file](./config.yaml), set the llm `api` variable to `"sambastudio"`, and set the `CoE` and `select_expert` configs if you are using a CoE endpoint.

## Windows requirements

- If you are using Windows, make sure your system has Microsoft Visual C++ Redistributable installed. You can install it from [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and make sure to check all boxes regarding C++ section. (Compatible versions: 2015, 2017, 2019 or 2022)

# Workflow

The challenge in fine tuning models of any modality is obtaining high quality, relevant data for the target domain of interest.  In recent years, large language models (LLMs) have been released that have used massive training datasets with prompt completions that cover number of capabilities, including natural, human aligned completions, sturctured outputs, etc.  

Now that were are working with multimodal, in this case image-text, models, we also need to curate instruction tuning data consisting of images and text.  To begin, we recommend iterating on the [generate tables](./scripts/generate_tables.py) script to get formatting correct for each table of interest.  Outputs are placed directly in: [synthetic_data/tmp/images](./../synthetic_data/tmp/images/) under the yoda folder by default.  You may use the [table_templates.json](./table_templates/table_templates.json) file directly and iterate there, or perhaps, more simply, uncomment the placeholder column and table template in the script directly and iterate in that file as needed.  All final templates should be added to [table_templates.json](#./../table_templates/table_templates.json) after iteration.  

Once building the table templates is complete, proceed to running [create_synthetic_tables](./scripts/create_synthetic_tables.py) with flags listed in the [Scripts](#scripts) section.

# Scripts

## Iteration for table formatting/templating

[generate_tables.py](./scripts/generate_tables.py):  This is a utility script that be used to iterate on properly formatting tables as needed.  In particular, it allows users to test if their columns definitions and modified tsv formatted table provide the expected output correctly.  This script is only intended for testing and each table template should be eventually placed in [table_templates.json](./table_templates/table_templates.json).

## Generate synthetic tables

[create_synthetic_tables.py](./scripts/create_synthetic_tables.py): This script is used to generate synthetic tables, along with logging information for QC.  

Flags:
 - --name: The name of the run.  A folder will be created as "logs" and the name specified with the .log extension.  The logging is set to DEBUG level.
 - --num-its: The number of iterations/synthetic tables to generate.  At present, only online inference is used with no concurrent requests used for batch generation.  
 - --split: Accepts choices of "train" or "val".  At present, this does not control much, other than the naming of the json file created with synthetic prompts data.  There is likely to be contamination until better splitting logic is implemented.



# Future work

Future work will potentially include:
- increased control for table augmentations.
- increased controls of [albumentations](https://albumentations.ai/) image augmentations.
- increased control over the output dpi (a random choice of threee resolutions is currently hard coded).
- batch queueing for improved performance, depending on the model batch size and handling.
- improved train/val split logic.
- more templates as examples to start.

# Current testing

- Create virtual environment.
- pip install yoda/llava_data_prep/requirements.txt
- Note any system dependancies or issues installing the requirements.
- Run yoda/llava_data_prep/notebooks/edgar_ingestion.ipynb and ensure it completes.  Check some of the outputs.
- Run yoda/llava_data_prep/scripts/generate_tables.py and ensure yoda/synthetic_data/tmp/images/test_table.jpg is created.
- Run:
```bash
python yoda/llava_data_prep/scripts/create_synthetic_tables.py --name test --num-its 4
```
and ensure yoda/llava_data_prep/test is created with a subfolder of synthetic images and yoda/llava_data_prep/test/annotations_train.json exists.