
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Benchmarking
======================

<!-- TOC -->

- [Overview](#overview)
- [Getting started](#getting-started)
    - [Get access to your model](#get-access-to-your-model)
    - [Integrate your model](#integrate-your-model)
    - [Run the starter kit](#run-the-starter-kit)
    - [Use the starter kit](#use-the-starter-kit)
- [Workflow: Performance evaluation](#workflow-performance-evaluation)
    - [Select the LLM model](#1-select-the-llm-model)
    - [Analyze results](#)
- [Workflow: Performance on chat](#workflow-performance-on-chat)
    - [Select the LLM model](#1-select-the-llm-model-1)
    - [Interact with LLM](#)

- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

This AISK performs a performance evaluation on different LLM models hosted in SambaStudio Composition of Experts deployment. It allows users to configure various LLMs with diverse parameters, enabling experiments to not only generate different outputs, but measurement metrics simultanously. The Kit includes:
- A configurable SambaStudio COE connector. The connector generates answers from a deployed model.
- An app with two functionalities:
    - A performance evaluation process with configurable options that users will utilize to obtain and compare different results. 
    - A chat interface with configurable options that users will set to interact and get performance metrics.

This sample is ready-to-use. We provide two options:
* [Getting Started](#getting-started) help you run the kit by following a few simple steps.
* [Performance evaluation](#workflow-performance-evaluation) serves as a starting point for customizing the available parameters and obtain interesting performance results based on cocurrent processes.
* [Performance on chat](#workflow-performance-on-chat) provides a set of options and a chat interface to interact with and analyze multiple performance metrics.
   
# Getting started

## Get access to your model

First, you need access to a SambaStudio Composition of Experts (COE) model. Therefore, it's mandatory to have all necessary experts downloaded on the SambaStudio platform, specially those that will be used with this Kit. [Follow these steps](https://docs.sambanova.ai/sambastudio/latest/model-hub.html#_download_models_using_the_gui) to download the experts or check if  they're already downloaded.

Deploy the **Samba-1.0** model to an endpoint for inference in SambaStudio, either through the GUI or CLI. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html) for more information. It might take some minutes, so please be patient.

## Integrate your model

To integrate your LLM with this AI starter kit, follow these steps:
1. Clone the ai-starter-kit repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

2. Update the SambaStudio COE LLM API information in your target SambaNova application. 

    Update the environment variables file in the root repo directory `ai-starter-kit/.env` to point to the SambaStudio endpoint. For example, for an endpoint with the URL "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef012 update the env file (with no spaces) as:
   ```
    BASE_URL="https://api-stage.sambanova.net"
    PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
    API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
   ```

3. (Recommended) Use a `venv` or `conda` environment for installation, and do a `pip install`. 
```
cd ai-starter-kit/benchmarking
python3 -m venv benchmarking_env
source benchmarking_env/bin/activate
pip  install  -r  requirements.txt
```
## Run the starter kit
To run the demo, execute the following command:
```
streamlit run streamlit/app.py --browser.gatherUsageStats false 
```

After deploying the starter kit, you'll see the following user interface:

![capture of enterprise_knowledge_retriever_demo](./imgs/performance_eval.png)

## Use the starter kit 

More details will come in the following sections [Workflow: Performance evaluation](#workflow-performance-evaluation) and [Workflow: Performance on chat](#workflow-performance-on-chat). However, the general usage of both is the comming bullets:

1. In the left side bar, pick one of the two app functionalities: `Performance evaluation` or `Performance on chat`.

2. Select the LLM model/expert and configure each of the parameters related to each functionality.

3. Press the `Run` button, wait and see results in the middle of the screen. In the case of `Performance on chat` functionality, users are able to interact with the LLM in a chat interface.  

# Workflow: Performance evaluation 

Choose the option `Performance on chat` on the left side bar, the following interface shows up: 

![capture of enterprise_knowledge_retriever_demo](./imgs/performance_eval.png)

In order to use this functionality, please follow the steps below:

## 1. Select the LLM model
## 2. Choose parameter values
## 3. Run the performance evaluation process
## 4. See and analyze results

# Workflow: Performance on chat 

Choose the option `Performance on chat` on the left side bar, the following interface shows up: 

![capture of enterprise_knowledge_retriever_demo](./imgs/performance_on_chat.png)

In order to use this functionality, please follow the steps below:

## 1. Select the LLM model

Under section `Set up the LLM`, users need to select the LLM model that will be used as engine for chatting interactions. The app provides a diverse list of LLMs to choose:
- DonutLM-v1
- Lil-c3po
- llama-2-7b-chat-hf
- LlamaGuard-7b
- Mistral-7B-Instruct-v0.2
- Mistral-T5-7B-v1
- Rabbit-7B-DPO-Chat
- Snorkel-Mistral-PairRM-DPO
- v1olet_merged_dpo_7B
- zephyr-7b-beta

## 2. Choose LLM parameter values

Different LLM parameters are available for experimentation, directly related to the previously chosen LLM. The app provides toggles and sliders to facilitate the configuration of all these parameters. Users can use the default values or modify them as needed.

- Do sample (True, False)
- Max tokens to generate (from 50 to 4096)
- Repetition penalty (from 1 to 10)
- Temperature (from 0.01 to 1)
- Top k (from 1 to 1000)
- Top p (from 0.01 to 1)

## 3. Set up the LLM model

After selecting the desired LLM and configuring the parameters, users have to press the `Run` button on the bottom. It will automatically set up the chosen LLM and activate the chat interface for upcoming interactions.

## 4. Ask anything and see results

Users are able to ask anything and get a generated answer of their questions, as showed in the image bellow. In addition to the back and forth conversations between the user and the LLM, there is a expander option that users can click to see the following metrics per each LLM response:
- Latency (s)
- Throughput (tokens/s)
- Time to first token (s)
- Time per output token (ms)

![capture of enterprise_knowledge_retriever_demo](./imgs/performance_on_chat_results.png)


# Third-party tools and data sources 

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- streamlit (version 1.34.0)
- st-pages (version 0.4.5)
- ray (version 2.22.0)
- transformers (version 4.40.1)
- python-dotenv (version 1.0.0)
- Requests (version 2.31.0)
- seaborn (version 0.12.2)