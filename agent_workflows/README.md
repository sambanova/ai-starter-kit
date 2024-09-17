
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="./../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Agents with LangGraph
======================

<!-- TOC -->

- [Agents with LangGraph](#agents-with-langgraph)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the inference endpoint, configs and environment variables](#set-up-the-inference-endpoint-configs-and-environment-variables)
    - [Update the Embeddings API information](#update-the-embeddings-api-information)
- [Deploy the starter kit GUI](#deploy-the-starter-kit-gui)
    - [Workshop deployment: Use a virtual environment 3.11 preferred](#workshop-deployment-use-a-virtual-environment-311-preferred)
- [Use the starter kit](#use-the-starter-kit)
    - [Ingestion workflow](#ingestion-workflow)
    - [Retrieval workflow](#retrieval-workflow)
    - [Q&A workflow and scaling](#qa-workflow-and-scaling)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)
- [Acknowledgements and References](#acknowledgements-and-references)

<!-- /TOC -->

# Overview

This AI Starter Kit is primarily intended to demonstrate agentic workflows.  For improved user experience, try the sncloud. The default is Llama 3 70B for a good compromise between model capability and throughput, however, the other variations could be tried as well.  Llama 3 8B for lower latency or 405B for maximum capability.  The kit includes:
 - Compatibility with sncloud.
 -   A configurable SambaStudio connector. The connector generates answers from a deployed model.
 -   A configurable integration with a third-party vector database.
 -   An implementation and guide of a complex semantic search workflow using numerous chains via LangGraph.
 -   An implementation and guide of a hierarchical team of agents with human in the loop supervision.

The original motivation and implementation examples come from [LangGraph](https://blog.langchain.dev/langgraph/).  The chat format has been simplified for clarity and ease of use with SambaNova Systems' models.  Many examples can be found [here](https://github.com/langchain-ai/langgraph/tree/main/examples).  

The implementations provided as part of this starterkit build on a number of provided components for retrieval augmented generation (RAG), code generation, and [Tavily](https://tavily.com/) internet search.  These components have been used to build agentic pipelines and teams.  One large pipeline will be code generation augmented corrective RAG.  There are many different steps for handling query ambiguity, compound queries, mathematical reasoning through code generation and execution, etc.  The other example uses the aforementioned [components](./../utils/) to build teams of agents.  These teams have specific workflows customized for sub tasks.  A supervisor manages the teams, given query and response history and the state of the system, including what the next step should be.  After each round of execution, the user is able to review the results and make adjustments and/or provide input as needed.  

A GUI frontend built with Streamlit is also provided for a user-friendly interface.  This frontend provides a simple means for human in the loop interactions for the hierarchical teams of agents application.  The application is quite simple: it assumes corrective RAG is the preferred method of generation unless internet or web search is explicitly mentioned.  If no documents are found from the vectorstore based on the query, the user will be prompted asking if they would like the supervisor to have the Tavily search team search the internet for answers.  If an affirmative is provided to the supervisor from the user, then routing to the Tavily team will be carried out and found contexts will be passed in for corrective RAG, instead of contexts from the vectorstore.  If a negative response is provided from the user, the system will simply route to END.  Although simple in nature, this example should demonstrate how to build complex, human in the loop applications if and when needed.

This sample is ready-to-use. We provide: 

* Instructions for setup with SambaStudio. 
* Instructions for running the agent pipelines as is. 

# Before you begin

You have to set up your environment before you can run or customize the starter kit.

## Clone this repository

Clone the starter kit repo.
```
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

```
cd ai_starter_kit/
python3 -m venv agent_env
source agent_env
pip install --upgrade pip
pip  install  -r  agent_workflows/requirements.txt
```

2. Run the following command:
```bash
streamlit run streamlit/app.py --browser.gatherUsageStats false
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

All the packages/tools are listed in the `requirements.txt` file in the project directory.

# Acknowledgements and References

The following work aims to demonstrate complex agentic chains, which should have a better user experience when using Samba-1 Turbo.  The work herein has been leveraged and adapted from the great folks at LangGraph.  Some of the adaptations of the original works also demonstrate how to modularize different components of the LangGraph setup and implement in Streamlit for rapid, early development.  The original tutorial can be found here:

[langgraph agent tutorial](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_rag_agent_llama3_local.ipynb)

We also used Uber and Lyft documents from Llama Index's tutorials.
