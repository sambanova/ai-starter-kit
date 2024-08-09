
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="./../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Agents with LangGraph
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

This AI Starter Kit is primarily intended to demonstrate agentic workflows.  For improved user experience, try the fast Llama 3 models in Samba-1 Turbo. The default is Llama 3 70B for a good compromise between model capability and throughput, however, the other variations could be tried as well.  Llama 3 8B for lower latency or 405B for maximum capability.  The kit includes:
 -   A configurable SambaStudio connector. The connector generates answers from a deployed model.
 -   A configurable integration with a third-party vector database.
 -   An implementation and guide of a complex semantic search workflow using numerous chains via LangGraph.
 -   An implementation and guide of a hierarchical team of agents with human in the loop supervision.

The original motivation and implementation examples come from [LangGraph](https://blog.langchain.dev/langgraph/).  The chat format has been simplified for clarity and ease of use with SambaNova Systems' models.  Many examples can be found [here](https://github.com/langchain-ai/langgraph/tree/main/examples).  

The implementations provided as part of this starterkit build on a number of provided components for retrieval augmented generation (RAG), code generation, and [Tavily](https://tavily.com/) internet search.  These components have been used to build agentic pipelines and teams.  One large pipeline will be code generation augmented corrective RAG.  There are many different steps for handling query ambiguity, compound queries, mathematical reasoning through code generation and execution, etc.  The other example uses the aforementioned [components](./../utils/) to build teams of agents.  These teams have specific workflows customized for sub tasks.  A supervisor manages the teams, given query and response history and the state of the system, including what the next step should be.  After each round of execution, the user is able to review the results and make adjustments and/or provide input as needed.  

A GUI frontend built with Streamlit is also provided for a user-friendly interface.  This frontend provides a simple means for human in the loop interactions for the hierarchical teams of agents application.  The application is quite simple: it assumes corrective RAG is the preferred method of generation unless internet or web search is explicitly mentioned.  If no documents are found from the vectorstore based on the query, the user will be prompted asking if they would like the supervisor to have the Tavily search team search the internet for answers.  If an affirmative is provided to the supervisor from the user, then routing to the Tavily team will be carried out and found contexts will be passed in for corrective RAG, instead of contexts from the vectorstore.  If a negative response is provided from the user, the system will simply route to END.  Although simple in nature, this example should demonstrate how to build complex, human in the loop applications if and when needed.

This sample is ready-to-use. We provide: 

* Instructions for setup with SambaStudio or Sambaverse. 
* Instructions for running the agent pipelines as is. 

# Before you begin

You have to set up your environment before you can run or customize the starter kit.

## Clone this repository

Clone the starter kit repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the account and config file

The next step is to set up your environment to use one of the models available from SambaNova. If you're a current SambaNova customer with access to SambaStudio, you'll want to follow those instructions. If you are not yet a SambaNova customer, you can self-service provision API endpoints using Sambaverse. Note that Sambaverse, although freely available to the public, is very rate limited and will not have fast RDU optimized inference speeds.

### Setup for Sambaverse users 

1. Create a Sambaverse account at [Sambaverse](https://sambaverse.sambanova.net) and select your model. 
2. Get your [Sambaverse API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) (from the user button).
3. In the repo root directory find the config file in `sn-ai-starter-kit/.env` and specify the Sambaverse API key (with no spaces), as in the following example: 

```yaml
    SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
```

4. In the [config file](./config.yaml), set the `api` variable to `"sambaverse"`.

### Setup for SambaStudio users

To perform this setup, you will be using a hosted endpoint that has been setup for this workshop.  In enterprise settings, you must be a SambaNova customer with a SambaStudio account. The endpoint information will be shared in the workshop.  For customers:

1. Log in to SambaStudio and get your API authorization key. The steps for getting this key are described [here](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html#_acquire_the_api_key).
2. Select the LLM you want to use (e.g. Llama 2 70B chat) and deploy an endpoint for inference. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).
3. Update the `sn-ai-starter-kit/.env` config file in the root repo directory. Here's an example: 

 ```
   BASE_URL="https://api-stage.sambanova.net"
   PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
   ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
   API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
   ```

4. Open the [config file](./config.yaml), set the variable `api` to `"sambastudio"`, and save the file

### Update the Embedding API information

You have these options to specify the embedding API info: 

* **Option 1: Use a CPU embedding model**

    In the [config file](./config.yaml), set the variable `embedding_model:` to `"cpu"` 

* **Option 2: Set a SambaStudio embedding model**

    To increase inference speed, you can use SambaStudio E5 embedding model endpoint instead of using the default (CPU) Hugging Face embeddings, Follow [this guide](https://docs.sambanova.ai/sambastudio/latest/e5-large.html#_deploy_an_e5_large_v2_endpoint) to deploy your SambaStudio embedding model.  For the workshop, we will provide the E5 endpoint with batch size 32 for inference.

    NOTE: Be sure to set batch size model parameter to 32.

    1. Update API information for the SambaNova embedding endpoint in the **`sn-ai-starter-kit/.env`** file in the root repo directory. For example: 

    * Assume you have an endpoint with the URL
        "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
    * You can enter the following in the env file (with no spaces):
        ```
        EMBED_BASE_URL="https://api-stage.sambanova.net"
        EMBED_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
        EMBED_ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
        EMBED_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
        ```
    2. In the [config file](./config.yaml), set the variable `embedding_model` to `"sambastudio"`

    > NOTE: Using different embedding models (cpu or sambastudio) may change the results, and change How the embedding model is set and what the parameters are. 
    > 
    > You can see the difference in how they are set in the [vectordb.py file](../vectordb/vector_db.py)  (`load_embedding_model method`).


# Deploy the starter kit GUI

We recommend that you run the starter kit in a virtual environment or use a container. 

## Workshop deployment: Use a virtual environment (3.11 preferred)

If you want to use virtualenv or conda environment:

1. Install and update pip.

```
cd ai_starter_kit/
python3 -m venv agent_env
source agent_env
pip install --upgrade pip
pip  install  -r  agent_workflows/requirements.txt
```

2. Run the following command:
```
cd agent_workflows/streamlit/
streamlit run app.py --browser.gatherUsageStats false 
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

This workflow is an example of leveraging data stored in a vector database along with two different styles of agentic frameworks to enable advanced retrieval-based Q&A off your data.

For detailed decriptions of each setup, please consult the following notebooks:

* [Code RAG](./notebooks/code_rag.ipynb)

* [Corrective RAG Team](./notebooks/corrective_rag_team.ipynb)


## Q&A workflow and scaling

Developers and users should iterate on the workflow they feel is most desirable.  Users can be brought into the loop via a shared state between the streamlit app, supervisor, and teams.  System prompts to the supervisor and team members can be used to guide the conversation and general behavior of the application and user experience.  Complexity should be added with user exposure and input, as they may adapt to the method of working and both parties can find a balance between user experience and development and maintenance costs that will likely grow.

# Third-party tools and data sources


- streamlit (version 1.37.0)
- langchain (version 0.2.11)
- langchain-community (version 0.2.10)
- langgraph (version 0.1.6)
- pyppeteer (version 2.0.0)
- sentence_transformers (version 2.2.2)
- InstructorEmbedding (version 1.0.1)
- chromadb (version 0.5.5)
- PyPDF2 (version 3.0.1)
- unstructured_inference (version 0.7.27)
- unstructured[pdf] (version 0.13.3)
- PyMuPDF (version 1.23.4)
- python-dotenv (version 1.0.1)

# Acknowledgements and References

The following work aims to demonstrate complex agentic chains, which should have a better user experience when using Samba-1 Turbo.  The work herein has been leveraged and adapted from the great folks at LangGraph.  Some of the adaptations of the original works also demonstrate how to modularize different components of the LangGraph setup and implement in Streamlit for rapid, early development.  The original tutorial can be found here:

[langgraph agent tutorial](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_rag_agent_llama3_local.ipynb)

We also used Uber and Lyft documents from Llama Index's tutorials.
