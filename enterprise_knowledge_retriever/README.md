
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Enterprise Knowledge Retrieval
======================

Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

Table of Contents:

<!-- TOC -->

- [Enterprise Knowledge Retrieval](#enterprise-knowledge-retrieval)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the models and config file](#set-up-the-models-and-config-file)
        - [Set up the inference endpoint, configs and environment variables](#set-up-the-inference-endpoint-configs-and-environment-variables)
        - [Update the Embeddings API information](#update-the-embeddings-api-information)
    - [Install system dependencies](#install-system-dependencies)
    - [Deploy the AI starter kit Parser util](#deploy-the-ai-starter-kit-parser-util)
- [Deploy the starter kit GUI](#deploy-the-starter-kit-gui)
    - [Option 1: Use a virtual environment](#option-1-use-a-virtual-environment)
    - [Option 2: Deploy the starter kit in a Docker container](#option-2-deploy-the-starter-kit-in-a-docker-container)
- [Use the starter kit](#use-the-starter-kit)
    - [Ingestion workflow](#ingestion-workflow)
    - [Retrieval workflow](#retrieval-workflow)
    - [Q&A workflow](#qa-workflow)
- [Customizing the starter kit](#customizing-the-starter-kit)
    - [Import Data](#import-data)
    - [Split Data](#split-data)
    - [Embed data](#embed-data)
    - [Store embeddings](#store-embeddings)
    - [Retrieval and Reranking](#retrieval-and-reranking)
    - [Customize the LLM](#customize-the-llm)
        - [Sambaverse endpoint](#sambaverse-endpoint)
        - [SambaStudio endpoint](#sambastudio-endpoint)
    - [Experiment with prompt engineering](#experiment-with-prompt-engineering)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

This AI Starter Kit is an example of a semantic search workflow. You send your PDF or TXT file to the SambaNova platform, and get answers to questions about the documents content. The Kit includes:

- A configurable SambaStudio connector. The connector generates answers from a deployed model.
- A configurable integration with a third-party vector database.
- An implementation of a semantic search workflow using [Langchain LCEL](https://python.langchain.com/v0.1/docs/expression_language/)
- An implementation of a semantic search workflow using [LangGraph](https://langchain-ai.github.io/langgraph/#why-langgraph)
- Prompt construction strategies.

This sample is ready-to-use. We provide:

- Instructions for setup with SambaStudio or Sambaverse.
- Instructions for running the model as is.
- Instructions for customizing the model.
- A video overview of the setup process:

https://github.com/sambanova/ai-starter-kit/assets/150964187/958528ae-00c8-4934-b64f-80a03dd746f0

# Before you begin

You have to set up your environment before you can run or customize the starter kit.

## Clone this repository

Clone the starter kit repo.

```bash
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the models and config file

### Set up the inference endpoint, configs and environment variables

The next step is to set up your environment to use one of the models available from SambaNova. If you're a current SambaNova customer, you can deploy your models with SambaStudio. If you are not a SambaNova customer, you can self-service provision API endpoints using SambaNova Fast API or Sambaverse. Note that Sambaverse, although freely available to the public, is rate limited and will not have fast RDU optimized inference speeds.

- If using **SambaStudio** Please follow the instructions [here](../README.md#use-sambastudio-option-3) for setting up endpoint and your environment variables.
    Then in the [config file](./config.yaml) set the llm `api` variable to `"sambastudio"`, set the `CoE` and `select_expert` configs if using a CoE endpoint.

- If using **SambaNova Fast-API** Please follow the instructions [here](../README.md#use-sambanova-fast-api-option-1) for setting up your environment variables.
    Then in the [config file](./config.yaml) set the llm `api` variable to `"fastapi"` and set the `select_expert` config depending on the model you want to use.

- If using **Sambaverse** Please follow the instructions [here](../README.md#use-sambaverse-option-2) for getting your api key and setting up your environment variables.
    Then in the [config file](./config.yaml) set the llm `api` variable to `"sambaverse"` and set the `sambaverse_model_name`, and `select_expert` config depending on the model you want to use.

### Update the Embeddings API information

You have these options to specify the embedding API info: 

* **Option 1: Use a CPU embedding model**

    In the [config file](./config.yaml), set the variable `type` in `embedding_model` to `"cpu"`

* **Option 2: Set a SambaStudio embedding model**

To increase inference speed, you can use SambaStudio E5 embedding model endpoint instead of using the default (CPU) Hugging Face embeddings.

1. Follow the instructions [here](../README.md#update-api-information-for-sambanova-embeddings-model-optional) for setting up your environment variables.

2. In the [config file](./config.yaml), set the variable `type` `embedding_model` to `"sambastudio"` and set the configs `batch_size`, `coe` and `select_expert` according your sambastudio endpoint

    > NOTE: Using different embedding models (cpu or sambastudio) may change the results, and change How the embedding model is set and what the parameters are.

## Install system dependencies

- Ubuntu installation:

    ```bash
    sudo apt install tesseract-ocr
    ```

- Mac Homebrew installation:

    ```bash
    brew install tesseract
    ```

- Windows installation:
    > [Windows tessearct installation](https://github.com/UB-Mannheim/tesseract/wiki)

- For other linux distributions, follow the [**Tesseract-OCR installation guide**](https://tesseract-ocr.github.io/tessdoc/Installation.html)

## Deploy the AI starter kit Parser util 

This Starter kit uses a custom implementation of the Unstructured module so you will need to deploy the Unstructured API

Go to  the [parser util readme](../utils/parsing/README.md), and follow the instructions to deploy the parser util

> Alternatively you can omit this step setting the parameter 'partition_by_api' in `partition` section in the parser util [config file](../utils/parsing/config.yaml) as false, but then you will be able to parse only PDF documents

# Deploy the starter kit GUI

We recommend that you run the starter kit in a virtual environment or use a container. 

## Option 1: Use a virtual environment

If you want to use virtualenv or conda environment:

1. Install and update pip.

    ```bash
    cd ai_starter_kit/enterprise_knowledge_retriever
    python3 -m venv enterprise_knowledge_env
    source enterprise_knowledge_env/bin/activate
    pip  install  -r  requirements.txt
    ```

2. Run the following command:

    > This kit can be deployed using either a simple [LCEL]() implementation or a [LangGraph]() implementation, both of which produce identical results. This demonstrates the flexibility of the kit and showcases how to leverage these two approaches in Rag applications.

    - LCEL version

        ```bash
            streamlit run streamlit/app.py --browser.gatherUsageStats false 
        ```

    - LangGraph version

        ```bash
            streamlit run streamlit/langgraph_app.py --browser.gatherUsageStats false 
        ```

After deploying the starter kit you see the following user interface:

![capture of enterprise_knowledge_retriever_demo](./docs/enterprise_knowledge_app.png)

## Option 2: Deploy the starter kit in a Docker container 

NOTE: If you are deploying the docker container in Windows be sure to open the docker desktop application. 

To run the starter kit  with docker, run the following command:

    docker-compose up --build

You will be prompted to go to the link (http://localhost:8501/) in your browser where you will be greeted with the streamlit page as above.

Here's a short video demonstrating docker deployment:

https://github.com/sambanova/ai-starter-kit/assets/150964187/4f82e4aa-c9a9-45b4-961d-a4b369be5ec4


# Use the starter kit 

After you've deployed the GUI, you can use the start kit. Follow these steps:

1. In the **Pick a data source** pane, drag and drop or browse for files. The data source can be a [Chroma](https://docs.trychroma.com/getting-started) vectorstore or a series of PDF files.

2. Click **Process** to process all loaded PDFs. A vectorstore is created in memory. You can store on disk if you want.

3. In the main panel, you can ask questions about the PDF data. 

This workflow uses the AI starter kit as is with an ingestion, retrieval, response workflow. 

## Ingestion workflow

This workflow, included with this starter kit, is an example of parsing and indexing data for subsequent Q&A. The steps are:

1. **Document parsing:** Python packages like [unstructured](https://github.com/Unstructured-IO/unstructured-inference) are used to extract text from file documents. On the LangChain website, multiple [integrations](https://python.langchain.com/v0.2/docs/how_to/#document-loaders) for text extraction from multiple file types are available. Depending on the quality and the format of the files, this step might require customization for different use cases. this kit uses in behind for the document parsing step the [parser util](../utils/parser/), which leverages the unstructured module to parse the documents.

2. **Split data:** After the data has been parsed and its content extracted, is needed to split the data into chunks of text to be embedded and stored in a vector database. The size of the chunks of text depends on the context (sequence) length offered by the model. Generally, larger context lengths result in better performance. The method used to split text has an impact on performance (for instance, making sure there are no word breaks, sentence breaks, etc.). The downloaded data is split using the [parser util](../utils/parser/) that leverages unstructured module to split the parsed documents into chunks

3. **Embed data:** For each chunk of text from the previous step, we use an embeddings model to create a vector representation of the text. These embeddings are used in the storage and retrieval of the most relevant content given a user's query. The split text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html).

   NOTE: For more information about what an embeddings is click [here](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526)*

4. **Store embeddings:** Embeddings for each chunk, along with content and relevant metadata (such as source documents) are stored in a vector database. The embedding acts as the index in the database. In this template, we store information with each entry, which can be modified to suit your needs. There are several vector database options available, each with their own pros and cons. This starter kit is set up to use [Chroma](https://github.com/chroma-core/chroma) as the vector database because it is a free, open-source option with straightforward setup, but can easily be updated to use another if desired. In terms of metadata, `filename` and `page` are also attached to the embeddings which are extracted during parsing of the PDF documents.

## Retrieval workflow

This workflow is an example of leveraging data stored in a vector database along with a large language model to enable retrieval-based Q&A off your data. The steps are:

1. **Embed query:** The first step is to convert a user-submitted query to a common representation (an embedding) for subsequent use in identifying the most relevant stored content. Use the same embedding mode for query parsing and to generate embeddings. In this start kit, the query text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html), which is the same embedding model in the ingestion workflow.
 
2. **Retrieve relevant content:** Next, we use the embeddings representation of the query to make a retrieval request from the vector database, which in turn returns *relevant* entries (content) in it. The vector database therefore also acts as a retriever for fetching relevant information from the database.

    *More information about embeddings and their retrieval [here](https://pub.aimind.so/llm-embeddings-explained-simply-f7536d3d0e4b)*

3. **Rerank retrieved content** After retrieving a specified number of relevant chunks of information, a reranker model can be set to rerank the retrieved passages in order of relevance to the user query. Then are selected the top N documents with higher relevance scores and passes those chunks to the QA chain as context.

*Find more information about Retrieval augmented generation with LangChain [here](https://python.langchain.com/docs/modules/data_connection/)*

## Q&A workflow

After the relevant information is retrieved, the content is sent to a SambaNova LLM to generate a final response to the user query.

Before being sent to the LLM, the user's query is combined with the retrieved content along with instructions to form the prompt. This process involves prompt engineering, and is an important part of ensuring quality output. In this AI starter kit, customized prompts are provided to the LLM to improve the quality of response for this use case.

*Learn more about [Prompt engineering](https://www.promptingguide.ai/)*

# Customizing the starter kit

You can further customize the starter kit based on the use case.

## Import Data

Different packages are available to extract text from different file documents. They can be broadly categorized as
- OCR-based: [pytesseract](https://pypi.org/project/pytesseract/), [paddleOCR](https://pypi.org/project/paddleocr/), [unstructured](https://unstructured.io/)
- Non-OCR based: [pymupdf](https://pypi.org/project/PyMuPDF/), [pypdf](https://pypi.org/project/pypdf/), [unstructured](https://unstructured.io/)
Most of these packages have easy [integrations](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) with the Langchain library.

You can find examples of the usage of these loaders in the [Data extraction starter kit](../data_extraction/README.md).

This enterprise knowledge retriever kit includes a custom implementation of unstructured loader that isd able to load files from the following extensions: `[".eml", ".html", ".json", ".md", ".msg", ".rst", ".rtf", ".txt", ".xml", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic", ".csv", ".doc", ".docx", ".epub", ".odt", ".pdf", ".ppt", ".pptx", ".tsv", ".xlsx"]`

* You can modify the loading method in the following location:
```
   file: src/document_retrieval
   function: parse_doc
```

* You can also modify several parameters in the loading strategies changing the config in the [../utils/parsing/config.yaml](../utils/parsing/config.yaml) se more [here](../utils/parsing/README.md)

## Split Data

You can experiment with different ways of splitting the data, such as splitting by tokens or using context-aware splitting for code or markdown files. LangChain provides several examples of different kinds of splitting [here](https://python.langchain.com/docs/modules/data_connection/document_transformers/).

The `chunking` inside the parser utils config, which is used in this starter kit, can be further customized using the `chunk_max_characters` and `chunk_overlap` parameters. For LLMs with a long sequence length, use a larger value of `chunk_max_characters` to provide the LLM with broader context and improve performance. The `chunk_overlap` parameter is used to maintain continuity between different chunks.

* You can modify this and other parameters in the `chunking` config in the [../utils/parsing/config.yaml](../utils/parsing/config.yaml) se more [here](../utils/parsing/README.md)

## Embed data

Several open-source embedding models are available on Hugging Face. [This leaderboard](https://huggingface.co/spaces/mteb/leaderboard) ranks these models based on the Massive Text Embedding Benchmark (MTEB). A number of these models are available on SambaStudio and can be further fine-tuned on specific datasets to improve performance.

You can do this modification in the following location:

```
file: ai-starter-kit/vectordb/vector_db.py
function: load_embedding_model
```

## Store embeddings

The template can be customized to use different vector databases to store the embeddings generated by the embedding model. The [LangChain vector stores documentation](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/) provides a broad collection of vector stores that can be easily integrated.

You can do this modification in the following location:

```
file: app.py
function: create_vector_store
```

For details about the SambaStudio hosted embedding models see the section *Use Sambanova's LLMs and Embeddings Langchain wrappers* [here](../README.md)

## Retrieval and Reranking

A wide collection of retriever options is available. In this starter kit, the vector store was used as a retriever, but it can be enhanced and customized, as shown in some of the examples [here](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/).

You can do this modification in the following location:
file: [config.yaml](config.yaml)

```yaml
    "k_retrieved_documents": 15
    "score_threshold": 0.2
    "rerank": True
    "reranker": 'BAAI/bge-reranker-large'
    "final_k_retrieved_documents": 3
```

There you will be able to select the final number of retrieved documents, and if you want to use or not the reranker.

and the implementation can be customized in file: [document_retrieval.py](src/document_retrieval.py) for LCEL implementation.

``` python
function: get_qa_retrieval_chain
```

or in [utils/rag/rag_components.py](../utils/rag/rag_components.py) for LangGraph implementation

``` python
function: retrieve
```

## Customize the LLM

Certain customizations to the LLM itself can affect the starter kit performance.

### Sambaverse endpoint

With Sambaverse, you can compare the performance of different models:

1. Search the available models in playground and select the three dots. 
2. Click **Show code**, and search for the values of the `modelName` and `select_expert` tags.
3. To modify the parameters for calling the model, change the values of `sambaverse_model_name` and `sambaverse_expert` in *llm* in the `config.yaml` file. 
4. You can also set the values of temperature and maximum generation token in that file. 

### SambaStudio endpoint

The starter kit uses a LLM model in SambaStudio. You can fine tune the SambaStudio model to improve response quality. 

* To train a model in SambaStudio, [prepare your training data](https://docs.sambanova.ai/sambastudio/latest/generative-data-prep.html), [import your dataset into SambaStudio](https://docs.sambanova.ai/sambastudio/latest/add-datasets.html) and [run a training job](https://docs.sambanova.ai/sambastudio/latest/training.html)
* To modify the parameters for calling the model, make changes to the `config.yaml` file. You can also set the values of temperature and maximum generation token in that file. 

## Experiment with prompt engineering

Prompting has a significant effect on the quality of LLM responses. Prompts can be further customized to improve the overall quality of the responses from the LLMs. For example, in this starter kit, the following prompt was used to generate a response from the LLM, where `question` is the user query and `context` is the documents retrieved by the retriever.

```python
custom_prompt_template = """[INST]<<SYS>> You are a helpful assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the answer is not in the context, say that you don't know. Cross check if the answer is contained in provided context. If not than say \"I do not have information regarding this\". Do not use images or emojis in your answer. Keep the answer conversational and professional.<</SYS>>

{context}    

Question: {question}

Helpful answer: [/INST]"""

CUSTOMPROMPT = PromptTemplate(
template=custom_prompt_template, input_variables=["context", "question"]
)
```

You can make modifications in the following location:

```
file: prompts/llama7b-knowledge_retriever-custom_qa_prompt.yaml
```

# Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- streamlit (version 1.25.0)
- pydantic (version 2.7.0)
- pydantic_core (version 2.18.1)
- langchain-community (version 0.2.10)
- langchain-core (version 0.2.25)
- langchain (version 0.2.11)
- sentence_transformers (version 2.2.2)
- instructorembedding (version 1.0.1)
- faiss-cpu (version 1.7.4)
- PyPDF2 (version 3.0.1)
- python-dotenv (version 1.0.0)
- streamlit-extras
- pillow (version 9.1.0)
- sseclient-py (version 1.8.0)
- unstructured[pdf] (version 0.13.3)
- unstructured_inference (version 0.7.27)
- PyMuPDF (version 1.23.4)
- chromadb (version 0.4.24)
- langgraph (version 0.0.55)
