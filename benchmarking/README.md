
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

Deploy the **Samba-1.1** model to an endpoint for inference in SambaStudio, either through the GUI or CLI. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html#_create_a_coe_endpoint_using_the_gui) for more information. It might take some minutes, so please be patient.

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

There are two options that users can choose. First one is running the performance evaluation from terminal and the other is using streamlit app.

## Using terminal

1. Open the file `run.sh` and configure the following parameters in there:

   - model: model name to be used. If it's a COE model, add "COE/" prefix to the name. Example: "COE/Meta-Llama-3-8B-Instruct"
   - mean-input-tokens: average number of input tokens. It's recommended to choose no more than 1024 tokens to avoid long waitings. Default value: 1000.
   - stddev-input-tokens: standard deviation of input tokens. It's recommended to choose no more than 50% the amount of input tokens. Default value: 10.
   - mean-output-tokens: average number of output tokens. It's recommended to choose no more than 1024 tokens to avoid long waitings. Default value: 1000.
   - stddev-output-tokens: standard deviation of output tokens. It's recommended to choose no more than 50% the amount of output tokens. Default value: 10.
   - max-num-completed-requests: maximum number of completed requests. Default value: 32 
   - timeout: time when the process will stop. Default value: 600 seconds
   - num-concurrent-requests: number of concurrent workers. Currently, using just 1 is suggested since all performance metrics will be available.
   - results-dir: path to the results directory. Default value: "./data/results/llmperf"
   - llm-api: currently only supporting Sambanova's. Default value: "sambanova"
   - mode: whether the generation is in stream or batch mode. Default value: "stream" 
   - additional-sampling-params: additional params for LLM. Default value: '{}'

2. Run the following command in terminal. Performance evaluation process will start running and a progress bar will be shown until it is done.

```
sh run.sh
```

3. Review results and further customization. Results will be saved in `results-dir` location, and the name of the output files will depend on the model name, number of mean input/output tokens, number of concurrent workers, and generation mode. Besides, for each run, two files are generated with the following suffixes: `individual_responses` and `summary`.

- Individual responses file (WIP)
- Summary (WIP)

## Using streamlit app

Choose the option `Performance evaluation` on the left side bar, the following interface shows up: 

![capture of enterprise_knowledge_retriever_demo](./imgs/performance_eval.png)

In order to use this functionality, please follow the steps below:

### 1. Introduce the LLM model

Under section `Configuration`, users need to introduce the LLM model that will be used for the performance evaluation process. Please, go to the model card in SambaStudio and search for the list of experts that the model supports. Choose one of them and introduce the same name in here.

### 2. Choose parameter values

Different LLM parameters are available for experimentation, directly related to the previously chosen LLM. The app provides toggles and sliders to facilitate the configuration of all these parameters. Users can use the default values or modify them as needed.

- Number of input tokens: average number of input tokens. Default value: 250.
- Standard deviation of input tokens: standard deviation of input tokens. Default value: 50.
- Number of output tokens: average number of output tokens. Default value: 250.
- Standard deviation of output tokens: standard deviation of output tokens. Default value: 50.
- Number of total requests: maximum number of completed requests. Default value: 50 
- Number of concurrent requests: number of concurrent workers. Currently, using just 1 is suggested since all performance metrics will be available.
- Timeout: time when the process will stop. Default value: 600 seconds

### 3. Run the performance evaluation process

Click on `Run!` button. It will automatically start the process. Depending on the previous parameter configuration, it should take between 1 min and 20 min  

### 4. See and analyze results

- Scatter plots (WIP)
- Box plots (WIP)

# Workflow: Performance on chat 

Choose the option `Performance on chat` on the left side bar, the following interface shows up: 

![capture of enterprise_knowledge_retriever_demo](./imgs/performance_on_chat.png)

In order to use this functionality, please follow the steps below:

### 1. Introduce the LLM model

Under section `Set up the LLM`, users need to introduce the LLM model that will be used for chatting. Please, go to the model card in SambaStudio and search for the list of experts that the model supports. Choose one of them and introduce the same name in here.

## 2. Choose LLM parameter values

Different LLM parameters are available for experimentation depending on the LLM model deployed. These are directly related to the previously chosen LLM. The app provides toggles and sliders to facilitate the configuration of all these parameters. Users can use the default values or modify them as needed.

<!-- - Do sample (True, False) -->
- Max tokens to generate (from 50 to 2048)
<!-- - Repetition penalty (from 1 to 10)
- Temperature (from 0.01 to 1)
- Top k (from 1 to 1000)
- Top p (from 0.01 to 1.00) -->

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