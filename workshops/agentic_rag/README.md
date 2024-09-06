
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="./../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Complex Retrieval Augmented Generation Use Case
======================

<!-- TOC -->

- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the account and config file](#set-up-the-account-and-config-file)
        - [Setup for Sambaverse users](#setup-for-sambaverse-users)
        - [Setup for SambaStudio users](#setup-for-sambastudio-users)
        - [Update the Embedding API Information](#update-the-embedding-api-information)

- [Deploy the starter kit GUI](#deploy-the-starter-kit-gui)
- [Use the starter kit](#use-the-starter-kit)
    - [Ingestion workflow](#ingestion-workflow)
    - [Retrieval workflow](#retrieval-workflow)
    - [Q&A workflow](#qa-workflow)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)
- [Acknowledgements and reference](#acknowledgements-and-references)

<!-- /TOC -->

# Overview

This AI Starter Kit is primarily intended to show off the speed of Llama 3 8B in Samba-1 Turbo for low latency agentic workflows. The Kit includes:
 -   A configurable SambaStudio connector. The connector generates answers from a deployed model.
 -   A configurable integration with a third-party vector database.
 -   An implementation of a semantic search workflow using numerous chains via LangGraph.

This sample is ready-to-use. We provide: 

* Instructions for setup with SambaStudio or Sambaverse. 
* Instructions for running the model as is. 

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



# Deploy the starter kit GUI

We recommend that you run the starter kit in a virtual environment or use a container. 

## Workshop deployment: Use a virtual environment (3.11 preferred)

If you want to use virtualenv or conda environment:

1. Install and update pip.

* Mac
```
cd ai_starter_kit/
python3 -m venv complex_rag_env
source complex_rag_env/bin/activate
pip install --upgrade pip
pip  install  -r  /workshops/agentic_rag/complex_rag/requirements.txt
```
* Windows
```
cd ai_starter_kit/
python3 -m venv complex_rag_env
complex_rag>complex_rag_env\Scripts\activate
pip install --upgrade pip
pip  install  -r  /workshops/agentic_rag/complex_rag/requirements.txt
```

2. Run the following command:
```
streamlit run workshops/genai_summit/complex_rag/streamlit/app.py --browser.gatherUsageStats false 
```

# Use the starter kit 

After you've deployed the GUI, you can use the start kit. Follow these steps:

1. In the **Pick a data source** pane, drag and drop or browse for files. The data source can be a [Chroma](https://docs.trychroma.com/getting-started) vectorstore or a series of PDF files.

2. Click **Process** to process all loaded PDFs. A vectorstore is created in memory. You can store on disk if you want.

3. In the main panel, you can ask questions about the PDF data. 

This workflow uses the AI starter kit as is with an ingestion, retrieval, response workflow. 

## Ingestion workflow

This workflow, included with this starter kit, is an example of parsing and indexing data for subsequent Q&A. The steps are:

1. **Document parsing:** Python packages [pypdf2](https://pypi.org/project/PyPDF2/), [fitz](https://pymupdf.readthedocs.io/en/latest/) and [unstructured](https://github.com/Unstructured-IO/unstructured-inference) are used to extract text from PDF documents. On the LangChain website, multiple [integrations](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) for text extraction from PDF are available. Depending on the quality and the format of the PDF files, this step might require customization for different use cases. For TXT file loading, the default [txt loading](https://python.langchain.com/docs/modules/data_connection/document_loaders/) implementation of langchain is used. 

2. **Split data:** After the data has been parsed and its content extracted, we need to split the data into chunks of text to be embedded and stored in a vector database. The size of the chunks of text depends on the context (sequence) length offered by the model. Generally, larger context lengths result in better performance. The method used to split text has an impact on performance (for instance, making sure there are no word breaks, sentence breaks, etc.). The downloaded data is split using [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter).

3. **Embed data:** For each chunk of text from the previous step, we use an embeddings model to create a vector representation of the text. These embeddings are used in the storage and retrieval of the most relevant content given a user's query. The split text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html).

   NOTE: For more information about what an embeddings is click [here](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526)*

4. **Store embeddings:** Embeddings for each chunk, along with content and relevant metadata (such as source documents) are stored in a vector database. The embedding acts as the index in the database. In this template, we store information with each entry, which can be modified to suit your needs. There are several vector database options available, each with their own pros and cons. This starter kit is set up to use [Chroma](https://github.com/chroma-core/chroma) as the vector database because it is a free, open-source option with straightforward setup, but can easily be updated to use another if desired. In terms of metadata, `filename` and `page` are also attached to the embeddings which are extracted during parsing of the PDF documents.

## Retrieval workflow

This workflow is an example of leveraging data stored in a vector database along with a large language model to enable retrieval-based Q&A off your data. The steps are:

 1. **Embed query:** The first step is to convert a user-submitted query to a common representation (an embedding) for subsequent use in identifying the most relevant stored content. Use the same embedding mode for query parsing and to generate embeddings. In this start kit, the query text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html), which is the same embeddng model in the ingestion workflow.
 
 2. **Retrieve relevant content:** Next, we use the embeddings representation of the query to make a retrieval request from the vector database, which in turn returns *relevant* entries (content) in it. The vector database therefore also acts as a retriever for fetching relevant information from the database.

*More information about embeddings and their retrieval [here](https://pub.aimind.so/llm-embeddings-explained-simply-f7536d3d0e4b)*
 
*Find more information about Retrieval augmented generation with LangChain [here](https://python.langchain.com/docs/modules/data_connection/)*

## Q&A workflow

After the relevant information is retrieved, the content is sent to the LangGraph app that includes numerous Llama 3 8B calls.   Calls at conditional nodes reliably output JSON formatted strings and are parsed by Langchain's JSON output parser.  The value obtained decides the branch to follow in the graph.

# Third-party tools and data sources


- streamlit (version 1.25.0)
- langchain (version 0.2.1)
- langchain-community (version 0.2.1)
- langgraph (version 0.5.5)
- pyppeteer (version 2.0.0)
- datasets (version 2.19.1)
- sentence_transformers (version 2.2.2)
- instructorembedding (version 1.0.1)
- chromadb (version 0.4.24)
- PyPDF2 (version 3.0.1)
- unstructured_inference (version 0.7.27)
- unstructured[pdf] (version 0.13.3)
- PyMuPDF (version 1.23.4)
- python-dotenv (version 1.0.0)

# Acknowledgements and References

The following work aims to show the power of SambaNova Systems RDU acceleration, using Samba-1 Turbo.  The work herein has been leveraged and adapted from the great folks at LangGraph.  Some of the adaptations of the original works also demonstrate how to modularize different components of the LangGraph setup and implement in Streamlit for rapid, early development.  The original tutorial can be found here:

https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_rag_agent_llama3_local.ipynb
