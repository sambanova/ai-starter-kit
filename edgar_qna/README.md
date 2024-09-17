
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

EDGAR Q&A 
======================

Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

Table of Contents:

<!-- TOC -->

- [EDGAR Q&A](#edgar-qa)
- [Overview](#overview)
- [Workflow overview](#workflow-overview)
    - [Perform ingestion](#perform-ingestion)
    - [Perform retrieval](#perform-retrieval)
    - [Answer questions](#answer-questions)
- [Use the AI starter kit](#use-the-ai-starter-kit)
    - [Clone the repo](#clone-the-repo)
    - [Set up the models and config file](#set-up-the-models-and-config-file)
        - [Set up the inference endpoint, configs and environment variables](#set-up-the-inference-endpoint-configs-and-environment-variables)
        - [Update the Embeddings API information](#update-the-embeddings-api-information)
    - [Deploy the starter kit](#deploy-the-starter-kit)
        - [Option 1: Run Streamlit UI from local install](#option-1-run-streamlit-ui-from-local-install)
        - [Option 2: Run a Multiturn-chat Streamlit UI from local install](#option-2-run-a-multiturn-chat-streamlit-ui-from-local-install)
        - [Option 3: Run a comparative-Q&A-chat Streamlit UI from local install](#option-3-run-a-comparative-qa-chat-streamlit-ui-from-local-install)
        - [Option 4: Run via Docker](#option-4-run-via-docker)
- [Customizing the starter kit](#customizing-the-starter-kit)
    - [Customize data import](#customize-data-import)
    - [Customize data splitting](#customize-data-splitting)
    - [Customize data embedding](#customize-data-embedding)
    - [Customize embedding storage](#customize-embedding-storage)
    - [Customize retrieval](#customize-retrieval)
    - [Finetune a SambaStudio model](#finetune-a-sambastudio-model)
    - [Experiment with prompt engineering](#experiment-with-prompt-engineering)
    - [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

This AI starter kit is an example of building semantic search workflow with the SambaNova platform. Edgar Q&A uses data from companies' 10-K annual reports to answer questions. It includes:
* A configurable SambaStudio connector to run inference off a model deployed in it.
* A configurable integration with a third-party vector database.
* An implementation of the semantic search workflow and prompt construction strategies.

This example is ready to use. 
* Select one of the options in the [Getting Started](#getting-started) section and follow the steps. 
* Customize the starter kit to your organization's needs, as discussed in the [Customizing the Template](#customizing-the-template) section.

# Workflow overview

This AI starter kit implements two distinct workflows that pipelines a series of operations.

## Perform ingestion

This workflow is an example of downloading and indexing data for subsequent Q&A. Follow these steps:
1. **Download data:** This workflow begins with pulling 10K reports from the EDGAR dataset to be chunked, indexed and stored for future retrieval. EDGAR data is downloaded using the [SEC-DATA-DOWNLOADER](https://pypi.org/project/sec-edgar-downloader/), which retrieves the filing report in XBRL format.
2. **Parse data:** After obtaining the report in XBRL format, we parse the document and extract only relevant text information. We're parsing the document using [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/), which is a great tool for web scraping. 
3. **Split data:** After the data has been downloaded, we need to split the data into chunks of text to be embedded and stored in a vector database. The size of the chunk of text depends on the context (sequence) length offered by the model. Generally, larger context lengths result in better performance. The method used to split text also has an impact on performance (for instance, making sure there are no word breaks, sentence breaks, etc.). The downloaded data is split using [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter).
4. **Embed data:** For each chunk of text from the previous step, we use an embeddings model to create a vector representation of it. These embeddings are used in the storage and retrieval of the most relevant content based on the user's query. The split text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html).
5. **Store embeddings:** Embeddings for each chunk, along with content and relevant metadata (such as source documents) are stored in a vector database. The embedding acts as the index in the database. In this starter kit, we store information with each entry, which can be modified to suit your needs. There are several vector database options available, each with their own pros and cons. This AI starter kit is set up to use the [chromadb](https://www.trychroma.com/) vector database because it is free, open-source options with straightforward setup. You can easily update the code to use another database if desired. 


## Perform retrieval

This workflow is an example of leveraging data stored in a vector database along with a large language model to enable retrieval-based Q&A off your data. The steps are:
1. **Embed the query:** Given a user-submitted query, the first step is to convert it into a common representation (an embedding) for subsequent use in identifying the most relevant stored content. Because of this, it is recommended to use the *same* model during ingestion and query embedding. In this example, the query text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html), which is was also used in the ingestion workflow.
2.  **Retrieve relevant content:** Next, we use the embeddings representation of the query to make a retrieval request from the vector database, which returns *relevant* entries (content). Therefore, the vector database also acts as a retriever for fetching relevant information from the database. If the retrieval engine uses memory, then it can remember the chat history between the user and the system and improve the user experience. 

## Answer questions

1. **Send content to SambaNova LLM:** After the relevant information is retrieved, the content is sent to a SambaNova LLM to generate the response to the user query. 
2. **Prompt engineering:** The user's query is combined with the retrieved content along with instructions and chat history (if using memory) to form the prompt before being sent to the LLM. This process involves prompt engineering, and is an important part of ensuring quality output. In this AI template, customized prompts are provided to the LLM to improve the quality of response for this use case.


# Use the AI starter kit

To use this AI starter kit without modifications, follow these steps. 

## Clone the repo

1. Clone the ai-starter-kit repo.
```
  git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the models and config file

### Set up the inference endpoint, configs and environment variables

The next step is to set up your environment variables to use one of the models available from SambaNova. If you're a current SambaNova customer, you can deploy your models with SambaStudio. If you are not a SambaNova customer, you can self-service provision API endpoints using SambaNova Cloud.

- If using **SambaNova Cloud** Please follow the instructions [here](../README.md#use-sambanova-cloud-option-1) for setting up your environment variables.
    Then in the [config file](./config.yaml) set the llm `api` variable to `"sncloud"` and set the `select_expert` config depending on the model you want to use.

- If using **SambaStudio** Please follow the instructions [here](../README.md#use-sambastudio-option-2) for setting up endpoint and your environment variables.
    Then, in the [config file](./config.yaml) set the llm `api` variable to `"sambastudio"`, set the `CoE` and `select_expert` configs if using a CoE endpoint.

### Update the Embeddings API information

You have these options to specify the embedding API info: 

* **Option 1: Use a CPU embedding model**

    In the [config file](./config.yaml), set the variable `type` in `embedding_model` to `"cpu"`

* **Option 2: Set a SambaStudio embedding model**

To increase inference speed, you can use a SambaStudio embedding model endpoint instead of using the default (CPU) Hugging Face embeddings.

1. Follow the instructions [here](../README.md#use-sambastudio-embedding-option-2) for setting up your environment variables.

2. In the [config file](./config.yaml), set the variable `type` `embedding_model` to `"sambastudio"` and set the configs `batch_size`, `coe` and `select_expert` according your sambastudio endpoint

    > NOTE: Using different embedding models (cpu or sambastudio) may change the results, and change How the embedding model is set and what the parameters are. 


## Deploy the starter kit

You have several options for using this starter kit. 

### Option 1: Run Streamlit UI from local install

Running from local install is the simplest option and includes a simple Streamlit-based UI for quick experimentation. 


> **Important:** When running through local install, no 10-Ks for organizations are preindexed, with 10-Ks being pulled and indexed on-demand. The workflow to do this has been implemented in this starter kit. To pull the latest 10-K from EDGAR, simply specify the company ticker in the sample UI and click `Submit`. This results in a one-time fetch of the latest 10-K from EDGAR, parsing the XBRL file downloaded, chunking, embedding and indexing it before making it available for Q&A. As a result, it takes some time for the data to be available the first time you ask a question for a new company ticker. As this is a one-time operation per company ticker, all subsequent Q&A off that company ticker is much faster, as this process does not need to be repeated.
![](docs/edgar_qa_demo.png)


1. Update pip and install dependencies. We recomment that you use virtual env or `conda` environment for installation.
```bash
cd ai_starter_kit/edgar_qna/
python3 -m venv edgar_env
source edgar_env/bin/activate
pip  install  -r  requirements.txt
```

2. Run the following command:

```bash
streamlit run streamlit/app_qna.py --browser.gatherUsageStats false 
```
This opens the demo in your default browser at port 8501. 

### Option 2: Run a Multiturn-chat Streamlit UI from local install

This option is a Streamlit-based UI for experimenting with a multiturn conversational AI assistant. 


1. Update pip and install dependencies. We recomment that you use virtual env or `conda` environment for installation.
```bash
cd ai_starter_kit/edgar_qna/
python3 -m venv edgar_env
source edgar_env/bin/activate
pip  install  -r  requirements.txt
```

2. Run the following command:

```bash
streamlit run app_chat.py --browser.gatherUsageStats false 
```
This will open the demo in your default browser at port 8501.

> **Important:** When running through local install, at least a 10-K for any organization has to be pre-indexed. You can follow the steps in Option 1 to create the index of a 10-k report of available organizations. The workflow to interact with the system starts by picking a data source, in which you write the path to a previously indexed vector store. Then, click on `Load` and the specified vector data base will be used. As you imagine, this process is very fast since the vector store already exists. Finally, you have an up-and-running chatbot assitant that is more than happy to help you with any questions that you may have about the filing report stored. 
![](docs/edgar_multiturn_demo.png)

### Option 3: Run a comparative-Q&A-chat Streamlit UI from local install

This option is a Streamlit-based UI for experimenting with a comparative question and answering assistant. 

> **Important:** The workflow to interact with the system starts by picking a pair of companies to compare with using the drop down boxes. Then, click on `Load` and a vector data base with both reports will be used. If the vector store was previously generated, it will load it quickly. In case you want to force a reload, there's an option to do that too. Finally, you have an up-and-running chatbot assitant that is more than happy to help you with any comparative questions that you may have about the filing report stored. 
![](docs/edgar_comparative_qa_demo.png)


1. Update pip and install dependencies. We recomment that you use virtual env or `conda` environment for installation.
```bash
cd ai_starter_kit/edgar_qna/
python3 -m venv edgar_env
source edgar_env/bin/activate
pip  install  -r  requirements.txt
```

2. Run the following command:
```bash
streamlit run app_comparative_chat.py --browser.gatherUsageStats false 
```
This will open the demo in your default browser at port 8501.

### Option 4: Run via Docker

Running through Docker is the most scalable approach for running this AI starter kit. This approach provides a path to production deployment.
In this example, you execute the comparative chat demo (as in option 3).

1. To run the container with the AI starter kit image, enter the following command:
```bash
docker-compose up --build
```

2. When prompted, run the link (http://0.0.0.0:8501/) in your browser. You will be greeted with a page that's identical to the page in Option 3.

# Customizing the starter kit

You can customize the starter kit based on your use case.

## Customize data import

Depending on the format of input data files (e.g., .pdf, .docx, .rtf), different packages can be used for conversion to plain text files.

This kit parses the information downloaded from SEC as xlbr file

To modify import, create separate methods based on the existing methods in the following location:

```
file: edgar_sec.py
methods: download_sec_data, parse_xbrl_data
```

## Customize data splitting

You can experiment with different ways of splitting the data, such as splitting by tokens or using context-aware splitting for code or for markdown files. LangChain provides several examples of different kinds of splitting [here](https://python.langchain.com/docs/modules/data_connection/document_transformers/).

The **RecursiveCharacterTextSplitter**, which is used in this example, can be further customized using the `chunk_size` and `chunk_overlap` parameters. For LLMs with a long sequence length, a larger value of `chunk_size` could be used to provide the LLM with broader context and improve performance. The `chunk_overlap` parameter maintains continuity between different chunks.

To modify data splitting, you have these options: 

* Set the retrieval parameters in the following location:

file: [config.yaml](./config.yaml)

parameters:
```yaml
retrieval:
    "chunk_size": 500
    "chunk_overlap": 50
```

* Or modify the following method

```
file: edgar_sec.py
function: create_load_vector_store
```

## Customize data embedding

Several open-source embedding models are available on Hugging Face. [This leaderboard](https://huggingface.co/spaces/mteb/leaderboard) ranks these models based on the Massive Text Embedding Benchmark (MTEB). Several of these models are available in SambaStudio. You can fine tune one of these models on specific datasets to improve performance.


To customize data embeddings, make modifications in the following location:
```
file: edgar_sec.py
function: create_load_vector_store
```

For more information about the usage of SambaStudio hosted embedding models, see *Use Sambanova's LLMs and Embeddings Langchain wrappers* [here](../README.md)

## Customize embedding storage

You can customize the example to use different vector databases to store the embeddings that are generated by the embedding model. The [LangChain vector stores documentation](https://js.langchain.com/docs/modules/data_connection/vectorstores/integrations/) provides a broad collection of vector stores that can be easily integrated.

To customize where the embeddings are stored, go to the following location:

```
file: edgar_sec.py
function: create_load_vector_store
```

## Customize retrieval

Similar to the vector stores, a wide collection of retriever options are available depending on the use case. In this template, the vector store was used as a retriever, but it can be enhanced and customized, as shown in some of the examples. [here](https://js.langchain.com/docs/modules/data_connection/retrievers/).


This modification can be done in a separate retrieval method in the following location:

```
file: src/edgar_sec.py
methods: retrieval_qa_chain, retrieval_conversational_chain, retrieval_comparative_process
```

and their parameteres can be updated in the following location
file: [config.yaml](./config.yaml)
```yaml
retrieval:
    "db_type": "chroma"
    "n_retrieved_documents": 3
```


## Finetune a SambaStudio model

If you're a SambaNova customer, you can fine-tune any of the SambaStudio models to improve response quality. 

To train a model in SambaStudio, learn how to [prepare your training data](https://docs.sambanova.ai/sambastudio/latest/generative-data-prep.html), [import your dataset into SambaStudio](https://docs.sambanova.ai/sambastudio/latest/add-datasets.html) and [run a training job](https://docs.sambanova.ai/sambastudio/latest/training.html)

## Experiment with prompt engineering

Prompting has a significant effect on the quality of LLM responses. Customize prompts to improve the overall quality of the responses from the LLMs. For example, with this starter kit, the following prompt uses meta-tags for Llama LLM models and generates a response from the LLM, 
* `question` is the user query and `context` are the documents retrieved by the retriever.
```python
custom_prompt_template = """<s>[INST] <<SYS>>\nYou're a helpful assistant\n<</SYS>>
Use the following pieces of context about company annual/quarterly report filing to answer the question at the end. If the answer to the question cant be extracted from given CONTEXT than say I do not have information regarding this.

Context:
{context}

Question: {question}

Helpful Answer: [/INST]"""
CUSTOMPROMPT = PromptTemplate(
template=custom_prompt_template, input_variables=["context", "question"]
)
```
You can do this modification in the following location:
```
file: edgar_qna/prompts
```

## Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory.
