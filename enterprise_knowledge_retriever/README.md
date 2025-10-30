
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>

Enterprise Knowledge Retrieval
======================

Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

Table of Contents:

<!-- TOC -->

- [Enterprise Knowledge Retrieval](#enterprise-knowledge-retrieval)
- [1. Overview](#1-overview)
- [2. Before you begin](#2-before-you-begin)
  - [2.1. Clone this repository](#21-clone-this-repository)
  - [2.2. Set up the models, environment variables and config file](#22-set-up-the-models-environment-variables-and-config-file)
    - [2.2.1. Set up the generative model](#221-set-up-the-generative-model)
    - [2.2.2. Set up the vector database](#222-set-up-the-vector-database)
  - [2.2. Windows requirements](#22-windows-requirements)
- [2. Deploy the starter kit GUI](#2-deploy-the-starter-kit-gui)
  - [2.1. Option 1: Use a virtual environment](#21-option-1-use-a-virtual-environment)
  - [2.2. Option 2: Deploy the starter kit in a Docker container](#22-option-2-deploy-the-starter-kit-in-a-docker-container)
- [2. Use the starter kit](#2-use-the-starter-kit)
- [3. Customizing the starter kit](#3-customizing-the-starter-kit)
  - [3.1. Import Data](#31-import-data)
  - [3.2. Split Data](#32-split-data)
  - [3.3. Store embeddings](#33-store-embeddings)
  - [3.4. Retrieval and Reranking](#34-retrieval-and-reranking)
  - [3.5. Customize the LLM](#35-customize-the-llm)
  - [3.6. Experiment with prompt engineering](#36-experiment-with-prompt-engineering)
- [2. Third-party tools and data sources](#2-third-party-tools-and-data-sources)

<!-- /TOC -->

# 1. Overview

This AI Starter Kit is an example of a semantic search workflow. You send your PDF or TXT file to the SambaNova platform, and get answers to questions about the documents content. The Kit includes:

- A configurable SambaNova connector. The connector generates answers from a deployed model.
- A configurable integration with a third-party vector database.
- An implementation of a semantic search workflow using [Langchain LCEL](https://python.langchain.com/v0.1/docs/expression_language/).
- Prompt construction strategies.

This sample is ready-to-use. We provide:

- Instructions for running the model as is.
- Instructions for customizing the model.

# 2. Before you begin

You have to set up your environment before you can run or customize the starter kit.

## 2.1. Clone this repository

Clone the starter kit repo.

```bash
git clone https://github.com/sambanova/ai-starter-kit.git
```

## 2.2. Set up the models, environment variables and config file

### 2.2.1. Set up the generative model

The next step is to set up your environment variables to use one of the inference models available from SambaNova. You can obtain a free API key through SambaCloud.

Follow the instructions [here](../README.md#getting-a-sambanova-api-key-and-setting-your-generative-models) to set up your environment variables.

Then, in the [config file](./config.yaml), set the `model` config depending on the model you want to use.

### 2.2.2. Set up the vector database

Choose your vector database from the accessible integrations to power your RAG performance. Simply access the [config file](./config.yaml), and under the `retrieval` section, set the value of the variable `db_type` with your choice. You have the following supported open-source options:

* **Chroma (default)**: Specifiy this option by setting it to `"db_type": "chroma"`.

* **Milvus by Zilliz**: Specifiy this option by setting it to `"db_type": "milvus"`.

## 2.2. Windows requirements

- If you are using Windows, make sure your system has Microsoft Visual C++ Redistributable installed. You can install it from [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and make sure to check all boxes regarding C++ section. (Compatible versions: 2015, 2017, 2019 or 2022)

# 2. Deploy the starter kit GUI

We recommend that you run the starter kit in a virtual environment or use a container. We also recommend using Python >= 3.10 and < 3.12.

## 2.1. Option 1: Use a virtual environment

If you want to use virtualenv or conda environment:

1. Install and update pip.

    ```bash
    cd ai-starter-kit/enterprise_knowledge_retriever
    python3 -m venv enterprise_knowledge_env
    source enterprise_knowledge_env/bin/activate
    pip  install  -r  requirements.txt
    ```

2. Run the following command:


   ```bash
   streamlit run streamlit/app.py --browser.gatherUsageStats false 
   ```


After deploying the starter kit you see the following user interface:

![capture of enterprise_knowledge_retriever_demo](./docs/enterprise_knowledge_app.png)

## 2.2. Option 2: Deploy the starter kit in a Docker container 

NOTE: If you are deploying the docker container in Windows be sure to open the docker desktop application. 

To run the starter kit  with docker, run the following command:

    docker-compose up --build

You will be prompted to go to the link (http://localhost:8501/) in your browser where you will be greeted with the streamlit page as above.


# 2. Use the starter kit 

After you've deployed the GUI, you can use the starter kit. Follow these steps:

1. In the **Pick a datasource** pane, either drag and drop files or browse to select them. The data source can be a series of PDF files or a
 [Chroma](https://docs.trychroma.com/getting-started) vectorstore.

2. Click **Process** to process all loaded PDFs. This will create a vectorstore in memory, which you can optionally save to disk. **Note**: This step may take some time, particularly if you are processing large documents or using CPU-based embeddings. 

3. In the main panel, you can ask questions about the PDF data. 

This pipeline uses the AI starter kit as is with an ingestion, retrieval, and Q&A workflows. More details about each workflow are provided below:

<details>
<summary> Ingestion workflow </summary>

This workflow, included with this starter kit, is an example of parsing and indexing data for subsequent Q&A. The steps are:

1. **Document parsing:** Python packages like [PyMuPDF](https://pypi.org/project/PyMuPDF/) or [unstructured](https://github.com/Unstructured-IO/unstructured-inference) are used to extract text from file documents. On the LangChain website, multiple [integrations](https://python.langchain.com/v0.2/docs/how_to/#document-loaders) for text extraction from multiple file types are available. Depending on the quality and the format of the files, this step might require customization for different use cases. This kit uses the [parser util](../utils/parsing/) in the background for the document parsing step, which leverages either PyMuPDF or the unstructured module to parse the documents.

2. **Split data:** After the data has been parsed and its content extracted, it is necessary to split the data into chunks of text to be embedded and stored in a vector database. The size of the text chunks depends on the context (sequence) length offered by the model. Generally, larger context lengths result in better performance. The method used to split text also impacts performance; for instance, ensuring there are no word or sentence breaks is crucial. The downloaded data is split using the [parser util](../utils/parsing/), which leverages either PyMuPDF or the unstructured module to split the parsed documents into chunks.

3. **Embed data:** For each chunk of text from the previous step, we use an embeddings model to create a vector representation of the text. These embeddings are then used for storing and retrieving the most relevant content given a user's query. The split text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceInstructEmbeddings.html).

   *For more information about what embeddings are, click [here](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526).*

4. **Store embeddings:** Embeddings for each chunk, along with content and relevant metadata (such as source documents) are stored in a vector database where the embedding acts as the index. In this template, we store information with each entry, which can be modified to suit your needs. Several vector database options are available, each with their own pros and cons. This starter kit is set up to use [Chroma](https://github.com/chroma-core/chroma) as the vector database because it is a free, open-source option with straightforward setup, but it can easily be updated to use another if desired. In terms of metadata, `filename` and `page` are also attached to the embeddings, which are extracted during parsing of the PDF documents.

</details>

<details>
<summary> Retrieval workflow </summary>

This workflow is an example of leveraging data stored in a vector database along with a large language model to enable retrieval-based Q&A from your data. The steps are:

1. **Embed query:** The first step is to convert a user-submitted query to a common representation (an embedding) for subsequent use in identifying the most relevant stored content. Use the same embedding mode for query parsing and to generate embeddings. In this start kit, the query text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceInstructEmbeddings.html), which is the same embedding model in the ingestion workflow.
 
2. **Retrieve relevant content:** Next, we use the embeddings representation of the query to make a retrieval request from the vector database, which in turn returns *relevant* entries (content) in it. Thus, the vector database also acts as a retriever for fetching relevant information.

   *For more information about embeddings and their retrieval, click [here](https://pub.aimind.so/llm-embeddings-explained-simply-f7536d3d0e4b).*

3. **Rerank retrieved content** After retrieving a specified number of relevant chunks of information, a reranker model can be set to rerank the retrieved passages in order of relevance to the user query. Then, the top N documents with the highest relevance scores are selected and passed to the QA chain as context.

   *For more information about retrieval augmented generation with LangChain, click [here](https://python.langchain.com/docs/modules/data_connection/).*

</details>

<details>
<summary> Q&A workflow </summary>

After the relevant information is retrieved, the content is sent to a SambaNova LLM to generate a final response to the user query.

Before being sent to the LLM, the user's query is combined with the retrieved content along with instructions to form the prompt. This process involves prompt engineering, and is an important part of ensuring quality output. In this AI starter kit, customized prompts are provided to the LLM to improve the quality of response for this use case.

*To learn more about prompt engineering, click [here](https://www.promptingguide.ai/).*

</details>

# 3. Customizing the starter kit

You can further customize the starter kit based on the use case.

## 3.1. Import Data

Different packages are available to extract text from different file documents. They can be broadly categorized as:
- OCR-based: [pytesseract](https://pypi.org/project/pytesseract/), [paddleOCR](https://pypi.org/project/paddleocr/), [unstructured](https://unstructured.io/)
- Non-OCR based: [pymupdf](https://pypi.org/project/PyMuPDF/), [pypdf](https://pypi.org/project/pypdf/)

Most of these packages have easy [integrations](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) with the Langchain library. You can find examples of the usage of these loaders in the [Data extraction starter kit](../data_extraction/README.md).

This enterprise knowledge retriever kit uses either PyMuPDF or a custom implementation of the unstructured loader. This can be configured in the [config.yaml](./config.yaml) file:

* If `pdf_only_mode` is set to True, then PyMuPDF is used as the data loader. Please note that in this case, only PDF documents are supported.

* If `pdf_only_mode` is set to False, then the unstructured loader is used, which works well with all file types. Please note that in this case, you need to install the following system dependencies if they are not already available on your system, for example, using `brew install` for Mac. Depending on what document types you're parsing, you may not need all of these:

    * `libmagic-dev` (filetype detection)
    * `poppler` (images and PDFs)
    * `tesseract-ocr` (images and PDFs)
    * `qpdf` (PDFs)
    * `libreoffice` (MS Office docs)
    * `pandoc` (EPUBs)

*You can also modify several parameters in the loading strategies by changing the [../utils/parsing/config.yaml](../utils/parsing/config.yaml) file, see more [here](../utils/parsing/README.md)*.

## 3.2. Split Data

You can experiment with different ways of splitting the data, such as splitting by tokens or using context-aware splitting for code or markdown files. LangChain provides several examples of different kinds of splitting; see more [here](https://python.langchain.com/docs/modules/data_connection/document_transformers/).

The `chunking` inside the parser utils config, which is used in this starter kit, can be further customized using the `chunk_max_characters` and `chunk_overlap` parameters. For LLMs with a long sequence length, use a larger value of `chunk_max_characters` to provide the LLM with broader context and improve performance. The `chunk_overlap` parameter is used to maintain continuity between different chunks.

You can modify this and other parameters in the `chunking` config in the [../utils/parsing/config.yaml](../utils/parsing/config.yaml); see more [here](../utils/parsing/README.md).

## 3.3. Store embeddings

The template can be customized to use different vector databases to store the embeddings generated by the embedding model. The [LangChain vector stores documentation](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/) provides a broad collection of vector stores that can be easily integrated.

By default, we use Chroma. You can change the vector store by setting `db_type` in the `create_vector_store()` function in [document_retrieval.py](./src/document_retrieval.py). 

## 3.4. Retrieval and Reranking

A wide collection of retriever options is available. In this starter kit, the vector store is used as a retriever, but it can be enhanced and customized, as shown in some of the examples [here](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/).

You can do this modification in the [config.yaml](config.yaml) file:

```yaml
    "k_retrieved_documents": 15
    "score_threshold": 0.2
    "rerank": False
    "reranker": 'BAAI/bge-reranker-large'
    "final_k_retrieved_documents": 5
```

There, you will be able to select the final number of retrieved documents and decide whether to use the reranker: 
* If `rerank` is set to `False`, then no reranker is used, and `final_k_retrieved_documents` represents the number of retrieved documents by the retriever. 
* If `rerank` is set to `True`, `k_retrieved_documents` first represent the number of documents retrieved by the retriever, and `final_k_retrieved_documents` represents the final number of documents after reranking. 

The implementation can be customized by modifying the `get_qa_retrieval_chain()` function in the [document_retrieval.py](src/document_retrieval.py) file.

## 3.5. Customize the LLM

Certain customizations to the LLM itself can affect the starter kit performance. To modify the parameters for calling the model, make changes to the [config file](./config.yaml). You can also set the values of `temperature` and `max_tokens_to_generate` in that file. 

## 3.6. Experiment with prompt engineering

Prompting has a significant effect on the quality of LLM responses. Prompts can be further customized to improve the overall quality of the responses from the LLMs. For example, in this starter kit, the following prompt template was used to generate a response from the LLM, where `question` is the user query and `context` is the documents retrieved by the retriever.

```yaml
template: |
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a knowledge base assistant chatbot powered by Sambanova's AI chip accelerator, designed to answer questions based on user-uploaded documents. 
    Use the following pieces of retrieved context to answer the question. Each piece of context includes the Source for reference. If the question references a specific source, then filter out that source and give a response based on that source. 
    If the answer is not in the context, say: "This information isn't in my current knowledge base." Then, suggest a related topic you can discuss based on the available context.
    Maintain a professional yet conversational tone. Do not use images or emojis in your answer.
    Prioritize accuracy and only provide information directly supported by the context. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    \n ------- \n
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

You can make modifications to the prompt template in the following file: 

```
file: prompts/qa_prompt.yaml
```

# 2. Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory.
