<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Router
======================

Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

Table of Contents:
<!-- TOC -->
- [Router](#Router)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the models, environment variables and config file](#set-up-the-models-environment-variables-and-config-file)
        - [Set up the generative model](#set-up-the-generative-model)
        - [Set up the embedding model](#set-up-the-embedding-model)
    - [Install dependencies](#install-dependencies)
    - [Windows requirements](#use-the-starter-kit)
- [Use the starter kit](#use-the-starter-kit)
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

## Install dependencies

We recommend that you run the starter kit in a virtual environment.

NOTE: python 3.9 or higher is required to use this kit.

Install the python dependencies in your project environment.

```bash
cd ai_starter_kit/router
python3 -m venv router_env
source router_env/bin/activate
pip  install  -r  requirements.txt
```

## Windows requirements

- If you are using Windows, make sure your system has Microsoft Visual C++ Redistributable installed. You can install it from [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and make sure to check all boxes regarding C++ section. (Compatible versions: 2015, 2017, 2019 or 2022)


# Use the starter kit 

After you've set up the environment, you can use the starter kit. Follow these steps:

1. Put your documents under the [data](./data/) folder.

2. Update the `keyword_path` under `router` in the [config file](./config.yaml).
    
2. We provide an example to call the router and connect it with a RAG pipeline in [notebook/RAG_with_router.ipynb](./notebook/RAG_with_router.ipynb).

# Customizing the starter kit
You can further customize the starter kit based on the use case.

## Customize the keyword extractor method

The [keyword extractor](./src/keyword_extractor.py) provides two methods to extract keywords:

* Use the [KeyBert](https://github.com/MaartenGr/KeyBERT) library. It uses BERT-embeddings and cosine similarity to find the sub-phrases in a document that are the most similar to the document itself.

* Use a generative language model. It uses prompt engineering to guide the LLM model to find keywords from documents.

*  Keywords can be extracted more efficiently by finding similarities between documents. We assume that highly similar documents will have the same keywords, so we extract keywords from only one document in each cluster and assign the keywords to all documents in the same cluster. To enble this feature, please set `use_clusters=True` under `router` in the [config file](./config.yaml).

## Customize the embedding model

By default, the keywords are exrtacted using a BERT-based embedding model. To change the embedding model, do the following:

* If using CPU embedding (i.e., `type` in `embedding_model` is set to `"cpu"` in the [config file](./config.yaml)), [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) from HuggingFaceInstruct is used by default. If you want to use another model, you will need to manually modify the `EMBEDDING_MODEL` variable and the `load_embedding_model()` function in the [api_gateway.py](../utils/model_wrappers/api_gateway.py). 
* If using SambaStudio embedding (i.e., `type` in `embedding_model` is set to `"sambastudio"` in the [config file](./config.yaml)), you will need to change the SambaStudio endpoint and/or the configs `batch_size`, `coe` and `select_expert` in the config file. 

## Customize the LLM model and/or use it to extract keywords

To change the LLM model or modify the parameters for calling the model, make changes to the `router` in [config file](./config.yaml).

The prompt for the model can be customized in [prompts/rag_routing_prompt.yaml](./prompts/rag_routing_prompt.yaml).

You can also use your own yaml file by placing the file under [prompts](./prompts) folder and changing the path of `router_prompt` in [config file](./config.yaml).

This LLM model can be applied to extract keywords by setting `use_llm=True` and `use_bert=False` in [config file](./config.yaml)

The prompt for the model can be customized in [prompts/keyword_extractor_prompt.yaml](./prompts/keyword_extractor_prompt.yaml)

## Customize the keyphrase extraction

The keyword extractor uses [KeyphraseVectorizers](https://github.com/TimSchopf/KeyphraseVectorizers) to extract keyphrase from documents. You can choose other keyphrase extration methods by changing the `vectorizer` in `extract_keywords` function in [keyword_extractor.py](./src/keyword_extractor.py).

```bash
if use_vectorizer:
    vectorizer = KeyphraseTfidfVectorizer()
    keyphrase_ngram_range = None
```

## Customize the RAG pipeline

The RAG pipeline uses functions in [document_retrieval.py](../enterprise_knowledge_retriever/src/document_retrieval.py). Please refer to [enterprise_knowledge_retriever](../enterprise_knowledge_retriever/README.md) for how to customize the RAG.

# Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory.