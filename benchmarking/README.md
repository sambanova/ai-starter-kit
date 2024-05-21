
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
    - [Deploy the starter kit](#run-the-starter-kit)
    - [Use the starter kit](#use-the-starter-kit)
- [Workflow: Performance evaluation](#workflow-performance-evaluation-wip)
    - [Set parameters](#)
    - [Analyze results](#)
- [Workflow: Performance on chat](#workflow-performance-on-chat-wip)
    - [Set parameters](#)
    - [Interact with LLM](#)

- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

This AISK performs a performance evaluation on different LLM models hosted in SambaStudio Center of Experts deployment. It allows users to configure various LLMs with diverse parameters, enabling experiments to not only generate different outputs, but measurement metrics simultanously. The Kit includes:
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

First, you need access to a SambaStudio Center of Experts (COE) model. Therefore, it's mandatory to have all necessary experts downloaded on the SambaStudio platform, specially those that will be used with this Kit. [Follow these steps](https://docs.sambanova.ai/sambastudio/latest/model-hub.html#_download_models_using_the_gui) to download the experts or check if  they're already downloaded.

Deploy the **Samba-1.0** model to an endpoint for inference in SambaStudio, either through the GUI or CLI. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html) for more information. It might take some minutes, so please be patient.

## Integrate your model

To integrate your LLM with this AI starter kit, follow these steps:
1. Clone the ai-starter-kit repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

2. Update the SambaStudio COE LLM API information in your target SambaNova application. 

    Update the environment variables file in the root repo directory `ai-starter-kit-snova/.env` to point to the SambaStudio endpoint. For example, for an endpoint with the URL "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef012 update the env file (with no spaces) as:
   ```
    SAMBASTUDIO_BASE_URL="https://api-stage.sambanova.net"
    SAMBASTUDIO_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    SAMBASTUDIO_ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
    SAMBASTUDIO_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
   ```

3. (Recommended) Use a `venv` or `conda` environment for installation, and do a `pip install`. 
```
cd ai-starter-kit-snova/benchmarking
python3 -m venv benchmarking_env
source benchmarking_env/bin/activate
pip  install  -r  requirements.txt
```
## Run the starter kit
To run the demo, execute the following command:
```
cd streamlit
streamlit run app.py --browser.gatherUsageStats false 
```

After deploying the starter kit, you'll see the following user interface:

![capture of enterprise_knowledge_retriever_demo](./imgs/performance_eval.png)

## Use the starter kit 

More details will come in the following sections [Workflow: Performance evaluation](#workflow-performance-evaluation) and [Workflow: Performance on chat](#workflow-performance-on-chat). However, the general usage of both is the comming bullets:

1. In the left side bar, pick one of the two app functionalities: `Performance evaluation` or `Performance on chat`.

2. Select the LLM model/expert and configure each of the parameters related to each functionality.

3. Press the `Run` button, wait and see the results in the middle of the screen. In the case of `Performance on chat` functionality, users are able to interact with the LLM in a chat interface.  

# Workflow: Performance evaluation (WIP)

# Workflow: Performance on chat (WIP)

# Third-party tools and data sources 

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- streamlit (version 1.34.0)
- st-pages (version 0.4.5)
- ray (version 2.22.0)
- transformers (version 4.40.1)
- python-dotenv (version 1.0.0)
- Requests (version 2.31.0)
- seaborn (version 0.12.2)