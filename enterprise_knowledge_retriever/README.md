
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Enterprise Knowledge Retrieval
======================

<!-- TOC -->

- [Enterprise Knowledge Retrieval](#enterprise-knowledge-retrieval)
- [Overview](#overview)
- [Getting started](#getting-started)
    - [Deploy your model](#get-access-to-your-model)
    - [Integrate your model](#integrate-your-model)
    - [Deploy the starter kit](#deploy-the-starter-kit)
- [Workflow: Ingestion, retrieval, response](#workflow-ingestion-retrieval-response)
    - [Ingestion](#ingestion)
    - [Retrieval](#retrieval)
- [Workflow: Customizing the template](#workflow-customizing-the-template)
    - [Import Data](#import-data)
    - [Split Data](#split-data)
    - [Embed data](#embed-data)
    - [Store embeddings](#store-embeddings)
    - [Retrieval](#retrieval)
    - [Large language model LLM](#large-language-model-llm)
        - [Prompt engineering](#prompt-engineering)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC --> 

# Overview

This AI Starter Kit is an example of a semantic search workflow. You send your PDF to the SambaNova platform, and get answers to questions about the PDF content. The Kit includes:
 -   A configurable SambaStudio connector. The connector generates answers from a deployed model.
 -   A configurable integration with a third-party vector database.
 -   An implementation of a semantic search workflow 
 -   Prompt construction strategies.

This sample is ready-to-use. We provide two options:
* [Getting Started](#getting-started) help you run a demo by following a few simple steps.
* [Customizing the Template](#customizing-the-template) serves as a starting point for customizing the demo to your organization's needs.
   
# Getting started

## Get access to your model

First, you need access to a model. You have these choices: 

* **Use Sambaverse**. Create an account and [get your API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key). You can now use any of the models included in [Sambaverse](sambaverse.sambanova.net).
* **Use SambaStudio**. Deploy the LLM of choice (e.g. Llama 2 13B chat, etc) to an endpoint for inference in SambaStudio, either through the GUI or CLI. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).

## Integrate your model

To integrate your LLM with this AI starter kit, follow these steps:
1. Clone the ai-starter-kit repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

2. Update the LLM API information in your target SambaNova application. 
- Option 1 **Sambaverse Endpoint:**
    (Step 1): Update the `sn-ai-starter-kit/.env` file in the root repo directory to include your API key. For example, enter an API key "456789ab-cdef-0123-4567-89abcdef0123" in the env file (with no spaces) as:

   ```
      SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
   ```

    (Step 2) In the [config file](./config.yaml) file, set the variable `api` to `"sambaverse"`
    
- Option 2 **SambaStudio Endpoint:**
     (Step 1) Update the environment variables file in the root repo directory `sn-ai-starter-kit/.env` to point to the SambaStudio endpoint. For example, for an endpoint with the URL "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef012 update the env file (with no spaces) as:
   ```
   BASE_URL="https://api-stage.sambanova.net"
   PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
   ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
   API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
   ```

(Step 2) In the [config file](./config.yaml) file, set the variable `api` to `"sambastudio"`

3. Update the Embedding API information in your target SambaNova application.

- Option 1 **In CPU embedding model**

    In the [config file](./config.yaml), set the variable `embedding_model:` to `"cpu"` 

- Option 2 **Set a sambastudio embedding model**

    You can use SambaStudio E5 embedding model endpoint instead of using default in cpu HugginFace embeddings to increase inference speed, follow [this guide](https://docs.sambanova.ai/sambastudio/latest/e5-large.html#_deploy_an_e5_large_v2_endpoint) to deploy your SambaStudio embedding model 
    > *Be sure to set batch size model parameter to 32*

    (Step 1) Update API information for the SambaNova embedding endpoint.  These are represented as configurable variables in the environment variables file in the root repo directory **```sn-ai-starter-kit/.env```**. For example, an endpoint with the URL
    "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
    would be entered in the env file (with no spaces) as:
    ```
    EMBED_BASE_URL="https://api-stage.sambanova.net"
    EMBED_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    EMBED_ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
    EMBED_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
    ```
    (Step 1) In the [config file](./config.yaml), set the variable `embedding_model` to `"sambastudio"`

  > Note that using different embedding models (cpu or sambastudio) may change the results, and change the way they are set and their parameters
  > 
  > You can see the difference in how they are set in the [vectordb.py file](../vectordb/vector_db.py)  *(load_embedding_model method)*

3.  Install system dependencies

   - Ubuntu instalation:
      ```
      sudo apt install tesseract-ocr
      ```
   - Mac Homebrew instalation:
      ```
      brew install tesseract
      ```
  - Windows instalation:
      > [Windows tessearc instalation](https://github.com/UB-Mannheim/tesseract/wiki)

   - For other linux distributions, follow the [**Tesseract-OCR installation guide**](https://tesseract-ocr.github.io/tessdoc/Installation.html) 

4. (Recommended) Use a `venv` or `conda` environment for installation, and do a `pip install`. 
```
cd ai_starter_kit/enterprise_knowledge_retriever
python3 -m venv enterprise_knowledge_env
source enterprise_knowledge_env/bin/activate
pip  install  -r  requirements.txt
```
## Deploy the starter kit
To run the demo, run the following commands:
```
streamlit run streamlit/app.py --browser.gatherUsageStats false 
```

After deploying the starter kit you see the following user interface:

![capture of enterprise_knowledge_retriever_demo](./docs/enterprise_knowledge_app.png)

## Use the starter kit 

1. In the **Pick a data source** pane, drag and drop or browse for files. The data source can be a [Chroma](https://docs.trychroma.com/getting-started) vectorstore or a series of PDF files.

2. Click **Process** to process all loaded PDFs. A vectorstore is created in memory that you can store on disk if you want.

3. Ask questions about the PDF data in the main panel. 

# Workflow: Ingestion, retrieval, response

This workflow uses the AI starter kit as is. 

## Ingestion

This workflow is an example of parsing and indexing data for subsequent Q&A. The steps are:

1. **Document parsing:** Python packages [pypdf2](https://pypi.org/project/PyPDF2/), [fitz](https://pymupdf.readthedocs.io/en/latest/) and [unstructured](https://github.com/Unstructured-IO/unstructured-inference) are used to extract text from PDF documents. There are multiple [integrations](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) available for text extraction from PDF on LangChain website. Depending on the quality and the format of the PDF files, this step might require customization for different use cases.

2.  **Split data:** After the data has been parsed and its content extracted, we need to split the data into chunks of text to be embedded and stored in a vector database. The size of the chunks of text depends on the context (sequence) length offered by the model. Generally, larger context lengths result in better performance. The method used to split text has an impact on performance (for instance, making sure there are no word breaks, sentence breaks, etc.). The downloaded data is split using [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter).

3. **Embed data:** 
For each chunk of text from the previous step, we use an embeddings model to create a vector representation of it. These embeddings are used in the storage and retrieval of the most relevant content given a user's query. The split text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html).

*For more information about what an embeddings is click [here](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526)*

4. **Store embeddings:** Embeddings for each chunk, along with content and relevant metadata (such as source documents) are stored in a vector database. The embedding acts as the index in the database. In this template, we store information with each entry, which can be modified to suit your needs. There are several vector database options available, each with their own pros and cons. This AI template is setup to use [Chroma](https://github.com/chroma-core/chroma) as the vector database because it is a free, open-source option with straightforward setup, but can easily be updated to use another if desired. In terms of metadata, ```filename``` and ```page``` are also attached to the embeddings which are extracted during document parsing of the PDF documents.

## Retrieval workflow
This workflow is an example of leveraging data stored in a vector database along with a large language model to enable retrieval-based Q&A off your data. The steps are:

 1.  **Embed query:** Given a user submitted query, the first step is to convert it into a common representation (an embedding) for subsequent use in identifying the most relevant stored content. Because of this, it is recommended to use the *same* embedding model to generate embeddings. In this sample, the query text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html), which is the same model in the ingestion workflow.
 
 2.  **Retrieve relevant content:** Next, we use the embeddings representation of the query to make a retrieval request from the vector database, which in turn returns *relevant* entries (content) in it. The vector database therefore also acts as a retriever for fetching relevant information from the database.

*More information about embeddings and their retrieval [here](https://pub.aimind.so/llm-embeddings-explained-simply-f7536d3d0e4b)*
 
*Find more information about Retrieval augmented generation with LangChain [here](https://python.langchain.com/docs/modules/data_connection/)*

## Q&A

After the relevant information is retrieved, the content is sent to a SambaNova LLM to generate a final response to the user query. 

The user's query is combined with the retrieved content along with instructions to form the prompt before being sent to the LLM. This process involves prompt engineering, and is an important part of ensuring quality output. In this AI starter kit, customized prompts are provided to the LLM to improve the quality of response for this use case.

*Learn more about [Prompt engineering](https://www.promptingguide.ai/)*

# Workflow: Customizing the template

The example template can be further customized based on the use case.

## Import Data

**PDF Format:** Different packages are available to extract text from PDF files. They can be broadly categorized as
- OCR-based: [pytesseract](https://pypi.org/project/pytesseract/), [paddleOCR](https://pypi.org/project/paddleocr/), [unstructured](https://unstructured.io/)
- Non-OCR based: [pymupdf](https://pypi.org/project/PyMuPDF/), [pypdf](https://pypi.org/project/pypdf/), [unstructured](https://unstructured.io/)
Most of these packages have easy [integrations](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) with the Langchain library.

You can find examples of the usage of these loaders in the [Data extraction starter kit](../data_extraction/README.md)

This starter kit includes three PDF loaders, unstructured, pypdf2, and fitz. You can change the default in the [config.yaml](./config.yaml) file in the parameter `loader`

You can include a new loader in the following location:
```
file: streamlit/app.py
function: get_data_for_splitting
```

## Split Data

You can experiment with different ways of splitting the data, such as splitting by tokens or using context-aware splitting for code or markdown files. LangChain provides several examples of different kinds of splitting [here](https://python.langchain.com/docs/modules/data_connection/document_transformers/).

The **RecursiveCharacterTextSplitter** inside the `vectordb` class, which is used in this starter kit, can be further customized using the `chunk_size` and `chunk_overlap` parameters. For LLMs with a long sequence length, use a larger value of `chunk_size` to provide the LLM with broader context and improve performance. The `chunk_overlap` parameter is used to maintain continuity between different chunks.


You can do this modification in the following location:
file: [config.yaml](config.yaml)
```yaml
retrieval:
    "chunk_size": 1200
    "chunk_overlap": 240
    ...
```

## Embed data

Several open-source embedding models are available on Hugging Face. [This leaderboard](https://huggingface.co/spaces/mteb/leaderboard) ranks these models based on the Massive Text Embedding Benchmark (MTEB). A number of these models are available on SambaStudio and can be further fine-tuned on specific datasets to improve performance.

You can do this modification in the following location:
```
file: ai-starter-kit/vectordb/vector_db.py
function: load_embedding_model
```

## Store embeddings

The template can be customized to use different vector databases to store the embeddings generated by the embedding model. The [LangChain vector stores documentation](https://js.langchain.com/docs/modules/data_connection/vectorstores/integrations/) provides a broad collection of vector stores that can be easily integrated.

You can do this modification in the following location:
```
file: app.py
function: create_vector_store
```

For details about the SambaStudio hosted embedding models see the section *Use Sambanova's LLMs and Embeddings Langchain wrappers* [here](../README.md)

## Retrieval

A wide collection of retriever options is available. In this starter kit, the vector store was used as a retriever, but it can be enhanced and customized, as shown in some of the examples [here](https://js.langchain.com/docs/modules/data_connection/retrievers/).

You can do this modification in the following location:
file: [config.yaml](config.yaml)
```yaml
    "db_type": "chroma"
    "k_retrieved_documents": 3
    "score_treshold": 0.6
```
and
file: [app.py](strematil/app.py)
```
function: get_qa_retrieval_chain 
```

## Large language model (LLM)

**If using Sambaverse endpoint**

You can test the performance of multiple models avalable in Sambaverse by the model:

- Search the available models in playground and select the three dots. Click **Show code**, and search for the values of the `modelName` and `select_expert` tags.
- To modify the parameters for calling the model, change the values of `sambaverse_model_name` and `sambaverse_expert` in *llm* in the `config.yaml` file. You can also set the values of temperature and maximum generation token in that file. 

**If using SambaStudio:**

- The starter kit uses the SN LLM model, which can be further fine-tuned to improve response quality. To train a model in SambaStudio, [prepare your training data](https://docs.sambanova.ai/sambastudio/latest/generative-data-prep.html), [import your dataset into SambaStudio](https://docs.sambanova.ai/sambastudio/latest/add-datasets.html) and [run a training job](https://docs.sambanova.ai/sambastudio/latest/training.html)
- To modify the parameters for calling the model, make changes to the `config.yaml` file. You can also set the values of temperature and maximum generation token in that file. 

### Prompt engineering

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
- langchain (version 0.0.252)
- sentence_transformers (version 2.2.2)
- instructorembedding (version 1.0.1)
- chromadb (version 0.4.8)
- PyPDF2 (version 3.0.1)
- unstructured_inference (version 0.7.23)
- PyMuPDF (version 1.23.4)
- python-dotenv (version 1.0.0)
