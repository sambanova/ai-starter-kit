
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
    - [About this template](#about-this-template)
- [Getting started](#getting-started)
    - [Deploy your model](#deploy-your-model)
    - [Integrate your model](#integrate-your-model)
    - [Deploy the starter kit](#deploy-the-starter-kit)
- [Workflow](#workflow)
    - [Ingestion](#ingestion)
    - [Retrieval](#retrieval)
    - [Response](#response)
- [Customizing the template](#customizing-the-template)
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
## About this template
This AI Starter Kit is an example of a semantic search workflow that can be built using the SambaNova platform to get answers to your questions using your PDFs as the source. It includes:
 -   A configurable SambaStudio connector to run inference off a deployed model.
 -   A configurable integration with a third-party vector database.
 -   An implementation of the semantic search workflow and prompt construction strategies.

This sample is ready-to-use. We provide two options to help you run this demo by following a few simple steps described in the [Getting Started](#getting-started) section. It also serves as a starting point for customization to your organization's needs, which you can learn more about in the [Customizing the Template](#customizing-the-template) section.

# Getting started

## Deploy your model

Begin creating an account and using the available models included in [Sambaverse](sambaverse.sambanova.net), and [get your API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) from the user button

Alternatively begin by deploying your LLM of choice (e.g. Llama 2 13B chat, etc) to an endpoint for inference in SambaStudio either through the GUI or CLI, as described in the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).

## Integrate your model
Integrate your LLM deployed on SambaStudio with this AI starter kit in two simple steps:
1. Clone repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

2. Set the LLM to use

    - Option 1 **Sambaverse Endpoint:**

        Update API information for your Sambaverse account.  These are represented as configurable variables in the environment variables file in the root repo directory **```sn-ai-starter-kit/.env```**. For example, an api key
        "456789ab-cdef-0123-4567-89abcdef0123"
        would be entered in the env file (with no spaces) as:
        ```
        SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
        ```

        Set in the [config.yaml file](./config.yaml), the variable *api* as: "sambaverse"

    - Option 2 **SambaStudio Endpoint:**: 

        Update API information for the SambaNova LLM.  These are represented as configurable variables in the environment variables file in the root repo directory **```sn-ai-starter-kit/.env```**. For example, an endpoint with the URL
        "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
        would be entered in the env file (with no spaces) as:
        ```
        BASE_URL="https://api-stage.sambanova.net"
        PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
        ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
        API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
        ```

        Set in the [config.yaml file](./config.yaml), the variable *api* as: "sambastudio"

3. Set the embedding model to use:

    - Option 1 **Set a local embedding model**:

        Set in the [config.yaml file](./config.yaml), the variable *embedding_model:* as: "cpu" 

    - Option 2 **Set a sambastudio embedding model**

        Update API information for the SambaNova embedding endpoint.  These are represented as configurable variables in the environment variables file in the root repo directory **```sn-ai-starter-kit/.env```**. For example, an endpoint with the URL
        "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
        would be entered in the env file (with no spaces) as:
        ```
        EMBED_BASE_URL="https://api-stage.sambanova.net"
        EMBED_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
        EMBED_ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
        EMBED_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
        ```
        Set in the [config.yaml file](./config.yaml), the variable *embedding_model:* as: "sambastudio"

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

   - For other linux dstributions follow the [**Tesseract-OCR installation guide**](https://tesseract-ocr.github.io/tessdoc/Installation.html) 

4. Install requirements: It is recommended to use virtualenv or conda environment for installation, and to update pip.
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

After deploying the starter kit you should see the following application user interface

![capture of enterprise_knowledge_retriever_demo](./docs/enterprise_knowledge_app.png)

## Starterkit usage 

1- Pick the data source, that could be previous stored [Chroma](https://docs.trychroma.com/getting-started) vectorstore or a series of PDF files

2- Icude PDF, put each of the docuemnts you want to load in the input area

3- Process loaded PDFs, there will be created a vectorstore in memory that you can also store on disk if you want.

5- Ask questions about website data!

# Workflow
This AI Starter Kit implements two distinct workflows that pipelines a series of operations.

## Ingestion
This workflow is an example of parsing and indexing data for subsequent Q&A. The steps are:

1. **Document parsing:** Python packages [pypdf2](https://pypi.org/project/PyPDF2/), [fitz](https://pymupdf.readthedocs.io/en/latest/) and [unstructured](https://github.com/Unstructured-IO/unstructured-inference) are  used to extract text from the PDF documents. There are multiple [integrations](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) available for text extraction from PDF on LangChain website. Depending on the quality and the format of the PDF files, this step might require customization for different use cases.

2.  **Split data:** Once the data has been parsed and its content extracted, we need to split the data into chunks of text to be embedded and stored in a vector database. This size of the chunk of text depends on the context (sequence) length offered by the model, and generally, larger context lengths result in better performance. The method used to split text also has an impact on performance (for instance, making sure there are no word breaks, sentence breaks, etc.). The downloaded data is split using [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter).

3. **Embed data:** 
For each chunk of text from the previous step, we use an embeddings model to create a vector representation of it. These embeddings are used in the storage and retrieval of the most relevant content given a user's query. The split text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html).

*For more information about what an embeddings is click [here](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526)*

4. **Store embeddings:** Embeddings for each chunk, along with content and relevant metadata (such as source documents) are stored in a vector database. The embedding acts as the index in the database. In this template, we store information with each entry, which can be modified to suit your needs. There are several vector database options available, each with their own pros and cons. This AI template is setup to use [Chroma](https://github.com/chroma-core/chroma) as the vector database because it is a free, open-source option with straightforward setup, but can easily be updated to use another if desired. In terms of metadata, ```filename``` and ```page``` are also attached to the embeddings which are extracted during document parsing of the pdf documents.

## Retrieval
This workflow is an example of leveraging data stored in a vector database along with a large language model to enable retrieval-based Q&A off your data. The steps are:


 1.  **Embed query:** Given a user submitted query, the first step is to convert it into a common representation (an embedding) for subsequent use in identifying the most relevant stored content. Because of this, it is recommended to use the *same* embedding model to generate embeddings. In this sample, the query text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html), which is the same model  in the ingestion workflow.
 
 2.  **Retrieve relevant content:** Next, we use the embeddings representation of the query to make a retrieval request from the vector database, which in turn returns *relevant* entries (content) in it. The vector database therefore also acts as a retriever for fetching relevant information from the database.

*More information about embeddings and their retrieval [here](https://pub.aimind.so/llm-embeddings-explained-simply-f7536d3d0e4b)*
 
 *Find more information about Retrieval augmented generation with LangChain [here](https://python.langchain.com/docs/modules/data_connection/)*

## Response
**SambaNova Large language model (LLM):** Once the relevant information is retrieved, the content is sent to a SambaNova LLM to generate the final response to the user query. 

   - **Prompt engineering:** The user's query is combined with the retrieved content along with instructions to form the prompt before being sent to the LLM. This process involves prompt engineering, and is an important part in ensuring quality output. In this AI template, customized prompts are provided to the LLM to improve the quality of response for this use case.

   *Learn more about [Prompt engineering](https://www.promptingguide.ai/)*

# Customizing the template

The provided example template can be further customized based on the use case.

## Import Data

**PDF Format:** Different packages are available to extract text out of PDF files. They can be broadly categorized in two classes, as below
- OCR-based: [pytesseract](https://pypi.org/project/pytesseract/), [paddleOCR](https://pypi.org/project/paddleocr/), [unstructured](https://unstructured.io/)
- Non-OCR based: [pymupdf](https://pypi.org/project/PyMuPDF/), [pypdf](https://pypi.org/project/pypdf/), [unstructured](https://unstructured.io/)
Most of these packages have easy [integrations](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) with the Langchain library.

you can find examples of the usage of these loaders in the [Data extraction starter kit](../data_extraction/README.md)

This template includes by default three PDF loaders, you can change the default one to use in the [config.yaml](./config.yaml) file in the parameter `loader`

> unstructured, pypdf2, and fitz are the default avalible loaders in this template

Including a new loader can be done in the following location:
```
file: streamlit/app.py
function: get_data_for_splitting
```

## Split Data

You can experiment with different ways of splitting the data, such as splitting by tokens or using context-aware splitting for code or markdown files. LangChain provides several examples of different kinds of splitting [here](https://python.langchain.com/docs/modules/data_connection/document_transformers/).

The **RecursiveCharacterTextSplitter** inside the vectordb class, which is used for this template, can be further customized using the `chunk_size` and `chunk_overlap` parameters. For LLMs with a long sequence length, a larger value of `chunk_size` could be used to provide the LLM with broader context and improve performance. The `chunk_overlap` parameter is used to maintain continuity between different chunks.


This modification can be done in the following location:
file: [config.yaml](config.yaml)
```yaml
retrieval:
    "chunk_size": 1200
    "chunk_overlap": 240
    ...
```

## Embed data

There are several open-source embedding models available on HuggingFace. [This leaderboard](https://huggingface.co/spaces/mteb/leaderboard) ranks these models based on the Massive Text Embedding Benchmark (MTEB). A number of these models are available on SambaStudio and can be further fine-tuned on specific datasets to improve performance.

This modification can be done in the following location:
```
file: ai-starter-kit/vectordb/vector_db.py
function: load_embedding_model
```

## Store embeddings

The template can be customized to use different vector databases to store the embeddings generated by the embedding model. The [LangChain vector stores documentation](https://js.langchain.com/docs/modules/data_connection/vectorstores/integrations/) provides a broad collection of vector stores that can be easily integrated.

This modification can be done in the following location:
```
file: app.py
function: create_vector_store
```

> Find more information about the usage of SambaStudio hosted embedding models in the section *Use Sambanova's LLMs and Embeddings Langchain wrappers* [here](../README.md)

## Retrieval

Similar to the vector stores, a wide collection of retriever options is also available depending on the use case. In this template, the vector store was used as a retriever, but it can be enhanced and customized, as shown in some of the examples [here](https://js.langchain.com/docs/modules/data_connection/retrievers/).

This modification can be done in the following location:
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

You can test the performace of multiple models avalable in sambaverse, for changing the model in this template:

- Search in the available models in playground and select the three dots the click in show code, you should search the values of these two tags `modelName` and `select_expert` 
- Modify the parameters for calling the model, those are in *llm* in ```config,yaml``` file setting the values of `sambaverse_model_name` and `sambaverse_expert`, temperature and maximun generation token can aso be modified

**If using Sambastudio:**

The template uses the SN LLM model, which can be further fine-tuned to improve response quality. To train a model in SambaStudio, learn how to [prepare your training data](https://docs.sambanova.ai/sambastudio/latest/generative-data-prep.html), [import your dataset into SambaStudio](https://docs.sambanova.ai/sambastudio/latest/add-datasets.html) and [run a training job](https://docs.sambanova.ai/sambastudio/latest/training.html)
Modify the parameters for calling the model, those are in *llm* in ```config,yaml``` file, temperature and maximun generation token can be modified

### Prompt engineering

Finally, prompting has a significant effect on the quality of LLM responses. Prompts can be further customized to improve the overall quality of the responses from the LLMs. For example, in the given template, the following prompt was used to generate a response from the LLM, where ```question``` is the user query and ```context``` are the documents retrieved by the retriever.
```python
custom_prompt_template = """[INST]<<SYS>> You are a helpful assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the answer is not in the context, say that you don't know. Cross check if the answer is contained in provided context. If not than say \"I do not have information regarding this\". Do not use images or emojis in your answer. Keep the answer conversational and professional.<</SYS>>

{context}    

Question: {question}

Helpful answer: [/INST]"""

CUSTOMPROMPT = PromptTemplate(
template=custom_prompt_template, input_variables=["context", "question"]
)
```

This modification can be done in the following location:
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
