<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Post Call Analysis
======================
<!-- TOC -->

- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the account and config file](#set-up-the-account-and-config-file)
        - [Setup for Sambaverse users](#setup-for-sambaverse-users)
        - [Setup for SambaStudio users](#setup-for-sambastudio-users)
    - [Set up a virtual environment (Recommended)](#set-up-a-virtual-environment-recommended)
- [Deploy the starter kit in a Docker container](#deploy-the-starter-kit-in-a-docker-container)
- [Deploy the starter kit](#deploy-the-starter-kit)
- [Use post-call analysis with your own data](#use-post-call-analysis-with-your-own-data)	
- [How the starter kit works](#how-the-starter-kit-works)
- [Customizing the starter kit](#customizing-the-starter-kit)
    - [Customize the model](#customize-the-model)
        - [Sambaverse](#sambaverse)
        - [SambaStudio](#sambastudio)
    - [Improve results with prompt engineering](#improve-results-with-prompt-engineering)
    - [Customize the factual accuracy analysis](#customize-the-factual-accuracy-analysis)
    - [Customize call quality assessment](#customize-call-quality-assessment)
    - [Customize batch inference](#customize-batch-inference)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview


This AI starter kit illustrates a systematic approach to post-call analysis including Automatic Speech Recognition (ASR), diarization, large language model analysis, and retrieval augmented generation (RAG) workflows. All workflows are built using the SambaNova platform. 

This starter kit provides:

* A customizable SambaStudio connector that facilitates LLM inference from deployed models.
* A configurable SambaStudio connector that enables ASR pipeline inference from deployed models.
* Implementation of a RAG workflow that includes prompt construction strategies which are tailored for call analysis, including:
    - Call summarization
    - Classification
    - Named entity recognition
    - Sentiment analysis
    - Factual accuracy analysis
    - Call quality assessment

This example is ready to use. We provide: 
* Instructions for setup. 
* Instructions for running the model as is. 
* Instructions for customization. 

# Before you begin

This starter kit automatically uses the SambaNova `snapi` CLI to create an ASR pipeline project and run batch inference jobs for doing speech recognition steps. You only have to set up your environment first. 

## Clone this repository

Clone the start kit repo.

```
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the account and config file 

The next step depends on whether you're a SambaStudio user or want to try the publicly available Sambaverse. 

### Setup for Sambaverse users

1. Create a Sambaverse account at [Sambaverse](sambaverse.sambanova.net) and select your model. 
2. Get your [Sambaverse API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) (from the user button).
3. In the repo root directory find the config file `sn-ai-starter-kit/.env` and specify the Sambaverse API key, as in the following example: 

```yaml
    SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
```

4. In the [config file](./config.yaml), set the `api` variable to `"sambaverse"`.

### Setup for SambaStudio users

To perform this setup, you must be a SambaNova customer with a SambaStudio account. 

1. Log in to SambaStudio and get your API authorization key. The steps for getting this key are described [here](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html#_acquire_the_api_key).
2. Select the LLM you want to use (e.g. Llama 2 70B chat) and deploy an endpoint for inference. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).
3. Update the `sn-ai-starter-kit/.env` config file in the root repo directory. Here's an example: 

```yaml
BASE_URL="https://api-stage.sambanova.net"
PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
SAMBASTUDIO_KEY="1234567890abcdef987654321fedcba0123456789abcdef"
```
4. Update the [config file](./config.yaml): 
*  Set the variable `api` to `"sambastudio"`.
* Run `snapi app list`, search for the `ASR With Diarization` section in the output, and set `asr_with_diarization_app_id` in the `apps` section of the config file to the app ID.

5. Follow the instructions in this [guide](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html) for installing SambaStudio SNSDK and SNAPI, (you can omit the *Create a virtual environment* step if you plan to create a `post_call_analysis_env` environment in the next step.)


## Set up a virtual environment (Recommended)

We recommend that you use virtualenv or conda environment for installation and update pip.

```bash
    cd ai-starter-kit/post_call_analysis
    python3 -m venv post_call_analysis_env
    source post_call_analysis_env/bin/activate
    pip install -r requirements.txt
```

# Deploy the starter kit in a Docker container 

To run this with docker

- 1. update the `SAMBASTUDIO_KEY`, `SNAPI`, `SNSDK` args in [docker-compose.yaml file](docker-compose.yaml)

- 2. run the command:

    ```bash
        docker-compose up --build
    ```
You will be prompted to go to the link (http://localhost:8501/) in your browser where you will be greeted with the streamlit page as above.


# Deploy the starter kit

To run the starter kit, run the following command

```
streamlit run streamlit/app.py --browser.gatherUsageStats false  
```

You will see the following Streamlit user interface

![capture of post_call_analysis_demo](./docs/post_call_analysis_base.png)

#  Use post-call analysis with your own data

1. Pick your source audio (.wav file) or transcript (CSV file containing the call transcription with diarization) . 

2. In the GUI, select either *Audio input* or *Text transcript input* and upload the file. If the input is an audio file, the processing step could take a couple of minutes to initialize the bash inference job in SambaStudio. Then you will see the following output structure.

![capture of post_call_analysis_demo](./docs/post_call_analysis_audio.png)

    Be sure to have at least 3 RDUs available in your SambaStudio environment.

3. In the GUI, select **Analysis Settings** and set the analysis parameters. You can define a list of classes for classification or specify entities for extraction.

    Provide the input path containing your facts and procedures knowledge bases, or you include a list of urls that you want to scrape to include it as facts and procedures knowledge bases.

    By default, this starter kit performs factual and procedures check only on .txt and PDF files or urls.  

4. Click the **Analyse transcription** button to run the analysis steps over the transcription. After a short time (a few minutes) you will see the following output structure.

![capture of post_call_analysis_demo](./docs/post_call_analysis_analysis.png)

# How the starter kit works

This section discusses how the start kit works and which tasks it performs with each step. 

* **Audio processing:** Audio processing is performed by the batch inference pipeline for ASR and Diarization and is composed of these steps:

    - **Transcription:** The Transcription step involves converting the audio data into text format. This step utilizes Automatic Speech Recognition (ASR) technology to transcribe spoken words into written text.

    - **Diarization:** The Diarization process distinguishes between different speakers in the conversation. It segments the audio data based on speaker characteristics, identifies each speaker's audio segments, and enables further analysis on a per-speaker basis.

    This pipeline retrieves a CSV file that contains the timestamped audio segments with speaker labels and transcription assigned to each segment.

* **Analysis:** Analysis consists of several steps. 

    - **Transcript Reduction:** Transcript reduction involves condensing the transcribed text to eliminate redundancy and shorten it enough to fit within the context length of the LLM. This results in a more concise representation of the conversation. This process uses the [reduce](https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.reduce.ReduceDocumentsChain.html) langchain chain and the [`reduce prompt template`](./prompts/reduce.yaml), which iteratively takes chunks of the conversation and compresses them while preserving key ideas and entities.

    - **Summarization:** Summarization generates a summary of the conversation, capturing its key points and main themes. This process uses the [`summarization prompt template`](./prompts/sumarization.yaml)

    - **Classification:** Classification categorizes the call based on its content or purpose by assigning it to a list of predefined classes or categories. This zero-shot classification uses the [`classification prompt template`](./prompts/topic_clasification.yaml). The reduced call transcription and a list of possible classes is passed to the langchain [list output parser](https://python.langchain.com/docs/modules/model_io/output_parsers/types/csv) to get a list structure as result.

    - **Named entity recognition:** Named Entity Recognition (NER) identifies and classifies named entities mentioned in the conversation, such as names of people, organizations, locations, and other entities of interest. We pass (1) a list of entities and (2) the reduced conversation to the [`NER prompt template`](./prompts/ner.yaml). The output is parsed with the langchain [Structured Output parser](https://python.langchain.com/docs/modules/model_io/output_parsers/types/structured), which returns a JSON structure containing a list of extracted values for each entity.

    - **Sentiment Analysis:** Sentiment Analysis determines the overall sentiment expressed in the conversation. This can help gauging the emotional tone of the interaction. This process uses the [`sentiment analysis prompt template`](./prompts/sentiment_analysis.yaml).

    - **Factual Accuracy Analysis:** Factual Accuracy Analysis evaluates the factual correctness of statements made during the conversation. This process uses a RAG (Retrieval Augmented Generation) methodology:
        - A series of documents are loaded, chunked, embedded, and stored in a vectorstore database.
        - Relevant documents for factual checks and procedures are retrieved and contrasted with the call transcription using the [`Factual Accuracy Analysis prompt template`](./prompts/factual_accuracy_analysis.yaml) and a [retrieval](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create_retrieval_chain.html) langchain chain. 
        - The output is then parsed using the langchain [Structured Output parser](https://python.langchain.com/docs/modules/model_io/output_parsers/types/structured), which converts it into a JSON structure containing a `correctness` field, an `error` field containing a description of the errors in the transcription, and a `score` field.

    - **Procedure Analysis:** Procedure Analysis evaluates whether the agent follows predefined procedures during the conversation and ensures that the agent's procedures correspond with procedural guidelines. This process uses the [`procedures analysis prompt template`](./prompts/procedures_analysis.yaml).

        The output is then parsed using the langchain [Structured Output parser](https://python.langchain.com/docs/modules/model_io/output_parsers/types/structured), which converts it into a JSON structure containing a `correctness` field, an `error` field containing a description of the errors evidenced in the transcription, and a `score` field.

    - **Call Quality Assessment:** Call Quality Assessment evaluates agent accuracy in the call. It helps identify areas for improvement in call handling processes. In this starter kit, a basic analysis is performed alongside the [Factual Accuracy Analysis](#factual-accuracy-analysis) and [Procedure Analysis](#procedure-analysis) steps. The analysis assigns a score based on the errors made by the agent in the call, and predicts the NPS score that the user might give. This is achieved using the `get_call_quallity_assessment()` method in the [Analysis script](./src/analysis.py).

# Customizing the starter kit

You can customize this starter kit in many ways. 

## Customize the model 

The precise process for customizing the model depends on whether you're using Sambaverse or SambaStudio. 

### Sambaverse

With Sambaverse, you can test and compare the performance of several models. 

To change the model that this starter kit is using: 

1. Log in to Sambaverse. 
2. Find the model you want to use in the playground, select the three dots, click **Show code**, and find the values of `modelName` and `select_expert`.
3. To modify the parameters for calling the model, open `config.yaml` and set the values of `sambaverse_model_name` and `sambaverse_expert`. You can also modify `temperature` and `maximum generation token` to experiment with that. 

### SambaStudio

If you're using a SambaStudio model, you can fine tune that model to improve response quality. 

1. Learn how to [prepare your training data](https://docs.sambanova.ai/sambastudio/latest/generative-data-prep.html), [import your dataset into SambaStudio](https://docs.sambanova.ai/sambastudio/latest/add-datasets.html) and [run a training job](https://docs.sambanova.ai/sambastudio/latest/training.html)

2. Modify the parameters for calling the model. In the `config.yaml` file, you can modify  `temperature` and `maximum generation token`.

### Improve results with prompt engineering

Prompting has a significant effect on the quality of LLM responses. All prompts used in [Analysis section](#analysis) can be further customized to improve the overall quality of the responses from the LLMs. For example, the following prompt was used to generate a response from the LLM, where `question` is the user query and `context` are the documents retrieved.
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
> Learn more about prompt engineering [here](https://www.promptingguide.ai/).


### Customize the factual accuracy analysis

You can customize or add specific document loaders in the `load_files` method, which can be found in the [vectordb class](../vectordb/vector_db.py). We have several examples of document loaders for different formats with specific capabilities in the [data extraction starter kit](../data_extraction/README.md).

### Customize call quality assessment

You can customize the basic example in this starter kit by including your own metrics in the evaluation steps. You can also modify the output parsers to obtain extra data or structures in the [analysis script](./src/analysis.py) methods.

### Customize batch inference

You can customize batch inference by modifying methods in the [analysis](./notebooks/analysis.ipynb), [asr](./notebooks/asr.ipynb) notebooks, and [analysis](./src/analysis.py), [asr](./src/asr.py) scripts.


# Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory. Some of the main packages are listed below:

- langchain (version 0.1.2)
- python-dotenv (version 1.0.1)
- requests (2.31.0)
- pydantic (1.10.14)
- unstructured (0.12.4)
- sentence_transformers (2.2.2)
- instructorembedding (1.0.1)
- faiss-cpu (1.7.4)
- streamlit (1.31.1)
- streamlit-extras (0.3.6)
- watchdog (4.0.0)
- sseclient (0.0.27)
- plotly (5.19.0)
- nbformat (5.9.2)
- librosa (0.10.1)
- streamlit_javascript (0.1.5)
