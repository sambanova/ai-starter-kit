<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Search Assistant
======================

<!-- TOC -->

- [Search Assistant](#search-assistant)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the account and config file](#set-up-the-account-and-config-file)
        - [Setup for SambaStudio](#setup-for-sambastudio)
        - [Setup for Sambaverse](#setup-for-sambaverse)
        - [Update the Embedding API information](#update-the-embedding-api-information)
- [Bring up the starter kit GUI](#bring-up-the-starter-kit-gui)
    - [Option 1: Use a virtual environment](#option-1-use-a-virtual-environment)
    - [Option 2: Deploy the starter kit in a Docker container](#option-2-deploy-the-starter-kit-in-a-docker-container)
    - [Run the demo](#run-the-demo)
- [Workflow overview](#workflow-overview)
    - [Answer and search workflow](#answer-and-search-workflow)
    - [Answer and scrape sites workflow](#answer-and-scrape-sites-workflow)
    - [Retrieval workflow](#retrieval-workflow)
- [Customizing the starter kit](#customizing-the-starter-kit)
    - [Use a custom serp tool](#use-a-custom-serp-tool)
    - [Customize website scraping](#customize-website-scraping)
    - [Customize document transformation](#customize-document-transformation)
    - [Customize data splitting](#customize-data-splitting)
    - [Customize data embedding](#customize-data-embedding)
    - [Customize embedding storage](#customize-embedding-storage)
    - [Customize retrieval](#customize-retrieval)
    - [Customize LLM usage](#customize-llm-usage)
        - [Sambaverse endpoint](#sambaverse-endpoint)
        - [SambaStudio endpoint](#sambastudio-endpoint)
    - [Experiment with prompt engineering](#experiment-with-prompt-engineering)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

This AI Starter Kit is an example of a semantic search workflow that can be built using the SambaNova platform to get answers to your questions using Google search information as the source. This kit includes:

 -   A configurable SambaStudio connector to run inference off a model deployed and trained on SambaNova hardware. 
 -   A configurable integration with a third-party vector database.
 -   An implementation of the semantic search workflow and prompt construction strategies.
 -   Configurable integrations with multiple SERP APIs
 -   An strategy for an instant question - search - answer workflow
 -   An strategy for a query - search - web-crawl - answer workflow

This example is ready to use. 

* Run the model following the steps in [Before you begin](#before-you-begin) and [Bring up the starter kit GUI](#bring-up-the-starter-kit-gui)
* Learn how the model works and look at resources in [Workflow overview](#workflow-overview).
* Customize the model to meet your organization's needs by looking at the [Customizing the starter kit](#customizing-the-starter-kit) section.

# Before you begin

You can use this model with Sambaverse or SambaStudio, but you have to do some setup first. 

## Clone this repository

Clone the start kit repo.

```
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the account and config file 

You can use the model with SambaStudio or Sambaverse. 

### Setup for SambaStudio

To perform SambaStudio setup, you must be a SambaNova customer with a SambaStudio account. 

1. Log in to SambaStudio and get your API authorization key. The steps for getting this key are described [here](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html#_acquire_the_api_key).
2. Select the LLM you want to use (e.g. Llama 2 70B chat) and deploy an endpoint for inference. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).
3. Update the `sn-ai-starter-kit/.env` config file in the root repo directory. Here's an example: 

    ```bash
    BASE_URL="https://api-stage.sambanova.net"
    PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
    API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
    ```

4. In the [config file](./config.yaml), set the variable `api` to `"sambastudio"`.

### Setup for Sambaverse

1. Create a Sambaverse account at [Sambaverse](sambaverse.sambanova.net) and select your model. 
2. Get your [Sambaverse API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) (from the user button).
3. In the repo root directory find the config file `sn-ai-starter-kit/.env` and specify the Sambaverse API key, as in the following example: 

    ```bash
    SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
    ```

4. In the [config file](./config.yaml), set the `api` variable to `"sambaverse"`.

### Update the Embedding API information

You have these options to specify the embedding API info: 

* **Option 1: Use a CPU embedding model**

    In the [config file](./config.yaml), set the variable `embedding_model:` to `"cpu"`

* **Option 2: Set a SambaStudio embedding model**

To increase inference speed, you can use SambaStudio E5 embedding model endpoint instead of using the default (CPU) Hugging Face embeddings, Follow [this guide](https://docs.sambanova.ai/sambastudio/latest/e5-large.html#_deploy_an_e5_large_v2_endpoint) to deploy your SambaStudio embedding model

NOTE: Be sure to set batch size model parameter to 32.

1. Update API information for the SambaNova embedding endpoint in the **`sn-ai-starter-kit/.env`** file in the root repo directory. For example:

    - Assume you have an endpoint with the URL
        "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
    - You can enter the following in the env file (with no spaces):

        ```bash
            SAMBASTUDIO_EMBEDDINGS_BASE_URL="https://api-stage.sambanova.net"
            SAMBASTUDIO_EMBEDDINGS_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
            SAMBASTUDIO_EMBEDDINGS_ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
            SAMBASTUDIO_EMBEDDINGS_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
        ```

2. In the [config file](./config.yaml), set the variable `embedding_model` to `"sambastudio"`

    > NOTE: Using different embedding models (cpu or sambastudio) may change the results, and change How the embedding model is set and what the parameters are.
    > 
    > You can see the difference in how they are set in the [vectordb.py file](../vectordb/vector_db.py)  (`load_embedding_model method`).


# Bring up the starter kit GUI

We recommend that you run the starter kit in a virtual environment or use a container. 

## Option 1: Use a virtual environment

If you want to use virtualenv or conda environment 

1. Install and update pip.

    ``` bash
    cd ai-starter-kit/web_crawled_data_retriever
    python3 -m venv search_assistant_env
    source search_assistant_env/bin/activate
    pip install -r requirements.txt
    ```

2. Set the serp tool to use. This kit provides 3 options of serp tool to use: [SerpAPI](https://serpapi.com/), [Serper](https://serper.dev/), [openSERP](https://github.com/karust/openserp).

- For [openSERP](https://github.com/karust/openserp) follow the docker usage [instructions](https://github.com/karust/openserp?tab=readme-ov-file#docker-usage---)

- For [SerpAPI](https://serpapi.com/) and [Serper](https://serper.dev/) create an account and follow the instructions to get your API_KEY for SambaStudio or Sambaverse. The add the key to in the environment variables file in the root repo directory `sn-ai-starter-kit/.env`. (`SERPER_API_KEY` or `SERPAPI_API_KEY`)

  > Setting more than of these tools it's optional you can set only one and run the kit with this, there are some pros and cons of each one of these tools


3. Run the following command:

    ```bash
    streamlit run streamlit/app.py --browser.gatherUsageStats false   
    ```

    You should see the following application user interface:

![capture of search_assistant_kit](./docs/search_assitant.png)

## Option 2: Deploy the starter kit in a Docker container 

If you want to use Docker:

1. Update the `SAMBASTUDIO_KEY`, `SNAPI`, `SNSDK` args in [docker-compose.yaml file](docker-compose.yaml)

2. Run the command:

    ```bash
    docker-compose up --build
    ```

You will be prompted to go to the link (http://localhost:8501/) in your browser where you will be greeted with the streamlit page shown in the screenshot above.

## Run the demo 

After the GUI is up and running, you can start making selections in the left pane of the GUI. 

1. Select the Search Tool to use. That's the tool that will search the internet. 
2. Select the search engine you want to use for retrieval.
3. Set the maximum number of search results to retrieve.
4. Select the method for retrieval
    - **Search and answer** Does a search for each query you pass to the search assistant, and uses the search result snippets to provide an answer.
    - **Search and Scrape Sites** Asks you for an initial query and searches and scrapes the sites. Creates a vector database from the result. You can then as other questions related to your initial query and the method uses the stored information to give an answer.
5. Click the **Set** button to start asking questions!

# Workflow overview

This AI starter kit implements two distinct workflows each with a series of operations.

## Answer and search workflow

1. **Search** Use the Serp tool to retrieve the search results, and use the snippets of the organic search results (Serper, OpenSerp) or the raw knowledge graph (Serpapi) as context.

2. **Answer** Call the LLM using the retrieved information as context to answer your question.

## Answer and scrape sites workflow

1. **Search** Use the Serp tool to retrieve the search results and get links of organic search result.

2. **Website crawling**  Scrape the HTML from the website using Langchain [AsyncHtmlLoader](https://python.langchain.com/docs/integrations/document_loaders/async_html) Document, which is built on top of the [requests](https://requests.readthedocs.io/en/latest/) and [aiohttp](https://docs.aiohttp.org/en/stable/) Python packages.

3. **Document parsing:** Document transformers are tools used to transform and manipulate documents. They take in structured documents as input and apply transformations to extract specific information or modify the documents' content. Document transformers can perform tasks such as extracting properties, generating summaries, translating text, filtering redundant documents, and more. Transformers process many documents efficiently and can be used to preprocess data before further analysis or to generate new versions of the documents with desired modifications.

   Depending on the required information you need to extract from websites, this step might require some customization.
   * Langchain Document Transformer [html2text](https://python.langchain.com/docs/integrations/document_transformers/html2text) is used to extract plain and clear text from the HTML documents. 
   * Other document transformers like the [BeautfulSoup transformer](https://python.langchain.com/docs/integrations/document_transformers/beautiful_soup) are available for plain text extraction from HTML and are included in the LangChain package. 
      
    If you want to retrieve remote files, this starter kit includes extra file type loading functionality. You can activate or deactivate these loaders listing the filetypes in the [config file](./config.yaml) in the parameter `extra_loaders`. Right now remote **PDF** loading is available

4. **Data splitting:** Due to token limits in LLMs, you need to split the data into chunks of text to be embedded and stored in a vector database after the data has been parsed and its content extracted. The size of the chunk of text depends on the context (sequence) length offered by the model. Generally, larger context lengths result in better performance. The method used to split text also has an impact on performance (for instance, making sure there are no word breaks, sentence breaks, etc.). The downloaded data is split using [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter).
    
5. **Data embedding:**  For each chunk of text from the previous step, we use an embeddings model to create a vector representation of it. These embeddings are used in the storage and retrieval of the most relevant content given a user's query. The split text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html).

   For more information about what an embeddings is click [here](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526)

6. **Embedding storage:**  Embeddings for each chunk, along with content and relevant metadata (such as source website), are stored in a vector database. The embedding acts as the index in the database. In this starter kit, we store information with each entry, which can be modified to suit your needs. Several vector database options are available, each with its own pros and cons. This starter kit uses [FAISS](https://github.com/facebookresearch/faiss) as the vector database because it's a free, open-source option with straightforward setup, but can easily be updated to use another database if desired. In terms of metadata, `website source`  is also attached to the embeddings which are stored during web scraping.

## Retrieval workflow

This workflow is an example of leveraging data stored in a vector database along with a large language model to enable retrieval-based Q&A of your data. This method is called [Retrieval Augmented Generation RAG](https://netraneupane.medium.com/retrieval-augmented-generation-rag-26c924ad8181), The steps are:

1. **Embed query:** The first step is to convert a user-submitted query into a common representation (an embedding) for subsequent use in identifying the most relevant stored content. Because of this, we recommend that you use the *same* embedding model to generate embeddings. In this sample, the query text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html), which is the same model in the ingestion workflow.
 
2. **Retrieve relevant content:** Next, we use the embeddings representation of the query to make a retrieval request from the vector database, which returns *relevant* entries (content). The vector database acts as a retriever for fetching relevant information from the database.
    
     Find more information about embeddings and their retrieval [here](https://pub.aimind.so/llm-embeddings-explained-simply-f7536d3d0e4b)
 
     Find more information about Retrieval augmented generation with LangChain [here](https://python.langchain.com/docs/modules/data_connection/)

# Customizing the starter kit

You can customize this starter kit based on your use case. 

## Use a custom serp tool

You can modify or change the behavior of the search step by including your custom method in [SearchAssistant](./src/search_assistant.py) class. Your method must be able to receive a query, have a `do_analysis` flag, and return a result and a list of retrieved URLS.

This modification can be done in the following location:
> file: [src/search_assistant.py](src/search_assistant.py)

## Customize website scraping

Different packages are available to crawl and extract from websites. This starter kit uses the [AsyncHtmlLoader](https://python.langchain.com/docs/integrations/document_loaders/async_html). Langchain also includes a couple of [HTML loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/html) that can be used.

This modification can be done in the following location:

> file: [src/search_assistant.py](src/search_assistant.py)
>
>function: `load_htmls`
>

The maximum number of sites in the scraping method is set to 20 scraped sited, but you can modify that limit and the web crawling behavior in the following location:

> file: [config.yaml](config.yaml)
>```yaml
>web_crawling:
>    "max_depth": 2
>    "max_scraped_websites": 20
>```

> file: [src/search_assistant.py](src/search_assistant.py)
>```
>function: web_crawl
>```

## Customize document transformation

Depending on the loader used for scraping the sites, you may want to use a transformation method to clean up the downloaded documents. You can do that in the following location:

> file: [src/search_assistant.py](src/search_assistant.py)
>
>function: `clean_docs`
>

[LangChain](https://python.langchain.com/docs/integrations/document_transformers) provides several document transformers that you can use.

## Customize data splitting

You can experiment with different ways of splitting the data, such as splitting by tokens or using context-aware splitting for code or markdown files. LangChain provides several examples of different kinds of splitting [here](https://python.langchain.com/docs/modules/data_connection/document_transformers/).


You can customize the **RecursiveCharacterTextSplitter** in the [kit src file](src/search_assistant.py), which is used by this starter kit by changing the `chunk_size` and `chunk_overlap` parameters. 
* For LLMs with a long sequence length, try using a larger value of `chunk_size` to provide the LLM with broader context and improve performance. 
* The `chunk_overlap` parameter is used to maintain continuity between different chunks.

This modification can be done in the following location:
> file: [config.yaml](config.yaml)
>```yaml
>retrieval:
>    "chunk_size": 1200
>    "chunk_overlap": 240
>    "db_type": "faiss"
>    "k_retrieved_documents": 4
>    "score_treshold": 0.5
>```

## Customize data embedding

Several open source embedding models are available on HuggingFace. [This leaderboard](https://huggingface.co/spaces/mteb/leaderboard) ranks these models based on the Massive Text Embedding Benchmark (MTEB). Several of these models are available on SambaStudio and can be used or further fine-tuned on specific datasets to improve performance.

This modification can be done in the following location:
> file: [../vectordb/vector_db.py](../vectordb/vector_db.py)
>
> function: `load_embedding_model`
>

> Find more information about the usage of SambaStudio hosted embedding models in the section *Use Sambanova's LLMs and Embeddings Langchain wrappers* [here](../README.md).

## Customize embedding storage

Customize search assistant to use a different vector database to store the embeddings generated by the embedding model. The [LangChain vector stores documentation](https://python.langchain.com/docs/integrations/vectorstores) provides a broad collection of vector stores that are easy to integrate.

This modification can be done in the following location:
> file: [../vectordb/vector_db.py](../vectordb/vector_db.py)
>
> function: `create_vector_store`
>

## Customize retrieval

Similar to the vector stores, a wide collection of retriever options is also available. This starter kit uses the vector store as a retriever, but it can be enhanced and customized, as shown in some of the examples [here](https://python.langchain.com/docs/integrations/retrievers).

This modification can be done in the following location:

file: [config.yaml](config.yaml)
```yaml
    "db_type": "chroma"
    "k_retrieved_documents": 3
    "score_treshold": 0.6
```

and
> file: [src/search_assistant.py](src/search_assistant.py)
>
>function: `retrieval_qa_chain`
>

## Customize LLM usage 

You can further customize the model itself. If you're using Sambaverse, you can also compare model performance for your use case. 

### Sambaverse endpoint

You can test the performance of multiple models available in Sambaverse. For changing the model used by this starter kit:

If you're using a Sambaverse endpoint, follow these steps:

1. In the playground, find the model you're interested in. 
2. Select the three dots and then **Show code** and note down the values of `modelName` and `select_expert`. 
3. Modify the parameters for calling the model. In the `config.yaml` file, set the values of `sambaverse_model_name` and `sambaverse_expert`. You can also modify temperature and maximum generation token.

### SambaStudio endpoint

The starter kit uses the SN LLM model, which can be further fine-tuned to improve response quality. 

To train a model in SambaStudio, learn how to: 
* [prepare your training data](https://docs.sambanova.ai/sambastudio/latest/generative-data-prep.html)
* [import your dataset into SambaStudio](https://docs.sambanova.ai/sambastudio/latest/add-datasets.html)
* [run a training job](https://docs.sambanova.ai/sambastudio/latest/training.html)

You can modify the parameters for calling the model and the temperature and maximum generation token in the `config,yaml` file.

## Experiment with prompt engineering

Prompting has a significant effect on the quality of LLM responses. Prompts can be further customized to improve the overall quality of the responses from the LLMs. For example, in this starter kit, the following prompt was used to generate a response from the LLM, where `question` is the user query and `context` are the documents retrieved by the search engine.
```yaml
template: |
          <s>[INST] <<SYS>>\nUse the following pieces of context to answer the question at the end.
          If the answer is not in context for answering, say that you don't know, don't try to make up an answer or provide an answer not extracted from provided context.
          Cross check if the answer is contained in provided context. If not than say \"I do not have information regarding this.\"\n
          context
          {context}
          end of context
          <</SYS>>/n
          Question: {question}
          Helpful Answer: [/INST]
)
```
Those modifications can be done in the following locations:

  > Prompt: retrieval Q&A chain  
  >
  > file: [prompts/llama7b-web_scraped_data_retriever.yaml](prompts/llama7b-web_scraped_data_retriever.yaml)

  > Prompt: Serpapi search and answer chain  
  >
  > file: [prompts/llama7b-llama70b-SerpapiSearchAnalysis.yaml](prompts/llama7b-llama70b-SerpapiSearchAnalysis.yaml)

  > Prompt: Serper search and answer chain  
  >
  > file: [prompts/llama7b-llama70b-SerpapiSearchAnalysis.yaml](prompts/llama7b-llama70b-SerpapiSearchAnalysis.yaml)

  > Prompt: OpenSerp search and answer chain  
  >
  > file: [prompts/llama7b-llama70b-SerpapiSearchAnalysis.yaml](prompts/llama7b-llama70b-SerpapiSearchAnalysis.yaml)

Learn more about prompt engineering [here](https://www.promptingguide.ai/)

# Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- streamlit (Version 1.32.2)
- streamlit-extras (Version 0.3.6)
- langchain (Version 0.2.1)
- langchain_community (Version 0.2.1)
- sentence_transformers (Version 2.2.2)
- instructorembedding (Version 1.0.1)
- faiss-cpu (Version 1.7.4)
- python-dotenv (Version 1.0.0)
- pydantic (Version 1.10.14)
- pydantic_core (Version 2.10.1)
- sseclient-py (Version 1.8.0)
- google-search-results (Version 2.4.2)
- html2text (Version 2024.2.26)
- unstructured[pdf] (Version 0.12.4)
