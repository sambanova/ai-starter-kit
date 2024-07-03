
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
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the account and config file](#set-up-the-account-and-config-file)
    - [Create the (virtual) environment](#create-the-virtual-environment)
- [Deploy the starter kit GUI](#deploy-the-starter-kit-gui)
- [Use the starter kit](#use-the-starter-kit)
    - [Performance evaluation workflow](#performance-evaluation-workflow)
        - [Using streamlit app](#using-streamlit-app)
        - [Using terminal](#using-terminal)
    - [Performance on chat workflow](#performance-on-chat-workflow)
- [Batching vs non-batching benchmarking](#batching-vs-non-batching-benchmarking)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

This AI Starter Kit evaluates the performance of different LLM models hosted in SambaStudio. It allows users to configure various LLMs with diverse parameters, enabling experiments to not only generate different outputs but also measurement metrics simultaneously. The Kit includes:
- A configurable SambaStudio connector. The connector generates answers from a deployed model.
- An app with two functionalities:
    - A performance evaluation process with configurable options that users will utilize to obtain and compare different results 
    - A chat interface with configurable options that users will set to interact and get performance metrics
- A bash process that is the core of the performance evaluation and provides more flexibility to users

This sample is ready-to-use. We provide:
- Instructions for setup with SambaStudio
- Instructions for running the model as-is
- Instructions for customizing the model
   
# Before you begin

To perform this setup, you must be a SambaNova customer with a SambaStudio account. You also have to set up your environment before you can run or customize the starter kit. 

_These steps assume a Mac/Linux/Unix shell environment. If using Windows, you will need to adjust some commands for navigating folders, activating virtual environments, etc._

## Clone this repository

Clone the starter kit repo.
```bash
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the account and config file

1. Log in to SambaStudio, select the LLM you want to use (e.g. COE/Meta-Llama-3-8B-Instruct), and deploy an endpoint for inference. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html) for a general reference, and the [Dynamic batching documentation](https://docs.sambanova.ai/sambastudio/latest/dynamic-batching.html#_create_a_dynamic_batching_coe_endpoint) for more information on how to deploy a dynamic batching endpoint.  

2. Update the `ai-starter-kit/.env` config file in the root repo directory with information on this endpoint:
    ```env
    # SambaStudio endpoint URLs follow the format:
    # <BASE_URL>/api/predict/generic/<PROJECT_ID>/<ENDPOINT_ID>
    # Both the endpoint URL and the endpoint API key can be found by clicking into an endpoint's details page

    BASE_URL="https://yoursambastudio.url"
    PROJECT_ID="your-samba-studio_model-projectid"
    ENDPOINT_ID="your-samba-studio-model-endpointid"
    API_KEY="your-samba-studio-model-apikey"
    ```

3. (Optional) If you are planning to use the `run.sh` bash process. More details about the bash process will be covered later.

## Create the (virtual) environment
1. (Recommended) Create a virtual environment and activate it: 
    ```bash
    python<version> -m venv <virtual-environment-name>
    source <virtual-environment-name>/bin/activate
    ```

2. Install the required dependencies:
    ```bash
    cd benchmarking # If not already in the benchmarking folder
    pip install -r requirements.txt
    ```

# Deploy the starter kit GUI

Ensure you are in the `benchmarking` folder and run the following command:
```
streamlit run streamlit/app.py --browser.gatherUsageStats false 
```

After deploying the starter kit, you will see the following user interface:

![perf_eval_image](./imgs/performance_eval.png)

## Use the starter kit 

After you've deployed the GUI, you can use the starter kit. More details will come in the following sections, however the general usage is described in the comming bullets: 

1. In the left side bar, select one of the two app functionalities: `Performance evaluation` or `Performance on chat`.

2. If the LLM deployed is a Composition of Experts, introduce the LLM expert and configure each of the parameters related to each functionality. Otherwise, just do the latter.

3. Press the `Run` button, wait and analyze results in the middle of the screen. In the case of `Performance on chat` functionality, users are able to interact with the LLM in a chat interface.  

### Performance evaluation workflow

There are two options that users can choose from. The first one is running the performance evaluation process using the terminal, while the other is using the Streamlit app.

#### Using streamlit app

Choose the option `Performance evaluation` on the left side bar, the following interface shows up: 

![perf_eval_image](./imgs/performance_eval.png)

In order to use this functionality, please follow the steps below:

1. Introduce the LLM model

Under the section `Configuration`, users need to introduce the LLM model that will be used for the performance evaluation process. If it's a CoE model, add a `COE/` prefix to the name (for example, `COE/Meta-Llama-3-8B-Instruct`). If you're not sure about the name of the model/expert you want to choose, please go to the model card in SambaStudio and search for the model/expert name.

2. Choose parameter values

Different LLM parameters are available for experimentation, directly related to the previously introduced LLM. The app provides toggles and sliders to facilitate the configuration of all these parameters. Users can use the default values or modify them as needed.

- Number of input tokens: average number of input tokens. Default value: 1000.
- Input tokens standard deviation: standard deviation of input tokens. Default value: 10.
- Number of output tokens: average number of output tokens. Default value: 1000.
- Output tokens standard deviation: standard deviation of output tokens. Default value: 10.
- Number of total requests: maximum number of completed requests. Default value: 32. 
- Number of concurrent workers: number of concurrent workers. Default value: 1. 
- Timeout: time when the process will stop. Default value: 600 seconds

3. Run the performance evaluation process

Click on the `Run!` button. It will automatically start the process. Depending on the previous parameter configuration, it should take between 1 min and 20 min. Some diagnostic/progress information will be displayed in the terminal shell.

4. See and analyze results

    _Note: Not all model endpoints currently support the calculation of server-side statistics. Depending on your choice of endpoint, then, you may see either client and server information, or you may see just the client-side information._

    **Bar plots**

    Results are composed by four bar plots. (Performance metric values are based on a SambaTurbo endpoint.)

    - Time to First Token bar plot: users should expect to see a value around 1.2 seconds from Client side numbers. If the endpoint supports dynamic batch size, this plot will show the TTFT per batch.
    - End-to-End Latency bar plot: users should expect to see a value around 8 seconds from Client side numbers. If the endpoint supports dynamic batch size, this plot will show the latency per batch.
    - Throughput bar plot: users should expect to see a value around 140 tokens/second from Client side numbers. If the endpoint supports dynamic batch size, this plot will show the throughput per request, per batch.
    - Total throughput per batch: users should expect to see a value around 140 tokens/second from Client side numbers. If the endpoint supports dynamic batch size, this plot will show the number of total throughput per batch.

#### Using terminal

Users have this option if they want to experiment using values that are beyond the limits specified in the Streamlit app parameters.

1. Open the file `run.sh` and configure the following parameters in there:

   - model: model name to be used. If it's a COE model, add "COE/" prefix to the name. Example: "COE/Meta-Llama-3-8B-Instruct"
   - mean-input-tokens: average number of input tokens. It's recommended to choose no more than 2000 tokens to avoid long waitings. Default value: 1000.
   - stddev-input-tokens: standard deviation of input tokens. It's recommended to choose no more than 50% the amount of input tokens. Default value: 10.
   - mean-output-tokens: average number of output tokens. It's recommended to choose no more than 2000 tokens to avoid long waitings. Default value: 1000.
   - stddev-output-tokens: standard deviation of output tokens. It's recommended to choose no more than 50% the amount of output tokens. Default value: 10.
   - max-num-completed-requests: maximum number of completed requests. Default value: 32 
   - num-concurrent-workers: number of concurrent workers. Default value: 1. 
   - timeout: time when the process will stop. Default value: 600 seconds
   - results-dir: path to the results directory. Default value: "./data/results/llmperf"
   - additional-sampling-params: additional params for LLM. Default value: '{}'

2. Run the following command in terminal. The performance evaluation process will start running and a progress bar will be shown until it is done.

```
sh run.sh
```

3. Review and analyze results. Results will be saved in `results-dir` location, and the name of the output files will depend on the model name, number of mean input/output tokens, number of concurrent workers, and generation mode, like the following:

```
<MODEL_NAME>_{NUM_INPUT_TOKENS}_{NUM_OUTPUT_TOKENS}_{NUM_CONCURRENT_WORKERS}_{MODE}
```

For each run, two files are generated with the following suffixes in the output file names: `_individual_responses` and `_summary`.

- Individual responses file 

This output file contains the number of input and output tokens, number of total tokens, Time To First Token (TTFT), End-To-End Latency (E2E Latency) and Throughput from Server (if available) and Client side, for each individual request sent to the LLM. Users can use this data for further analysis. We provide this notebook `notebooks/analyze-token-benchmark-results.ipynb` with some charts that they can use to start.

![individual_responses_image](./imgs/perf_eval_individual_responses_output.png)

- Summary file

This file includes various statistics such as percentiles, mean and standard deviation to describe the number of input and output tokens, number of total tokens, Time To First Token (TTFT), End-To-End Latency (E2E Latency) and Throughput from Client side. It also provides additional data points that bring more information about the overall run, like inputs used, number of errors, and number of completed requests per minute. 

![summary_output_image](./imgs/perf_eval_summary_output.png)

### Performance on chat workflow

Choose the option `Performance on chat` on the left side bar, the following interface shows up: 

![perf_on_chat_image](./imgs/performance_on_chat.png)

In order to use this functionality, please follow the steps below:

1. Introduce the LLM model

Under section `Set up the LLM`, users need to introduce the LLM model that will be used for chatting. If it's a COE model, add "COE/" prefix to the name. Example: "COE/Meta-Llama-3-8B-Instruct". If you're not sure about the name of the model/expert you want to choose, please go to the model card in SambaStudio and search for the model/expert name.

2. Choose LLM parameter values

Different LLM parameters are available for experimentation depending on the LLM model deployed. These are directly related to the previously introduced LLM. The app provides various options to facilitate the configuration of all these parameters. Users can use the default values or modify them as needed.

<!-- - Do sample (True, False) -->
- Max tokens to generate (from 50 to 2048)
<!-- - Repetition penalty (from 1 to 10)
- Temperature (from 0.01 to 1)
- Top k (from 1 to 1000)
- Top p (from 0.01 to 1.00) -->

3. Set up the LLM model

After introducing the LLM and configuring the parameters, users have to press the `Run` button on the bottom. It will automatically set up the introduced LLM and activate the chat interface for upcoming interactions.

4. Ask anything and see results

Users are able to ask anything and get a generated answer of their questions, as showed in the image bellow. In addition to the back and forth conversations between the user and the LLM, there is a expander option that users can click to see the following metrics per each LLM response:
- Latency (s)
- Throughput (tokens/s)
- Time to first token (s)

![perf_on_chat_image](./imgs/performance_on_chat_results.png)

# Batching vs non-batching benchmarking

This kit also supports [SambaNova Studio models with Dynamic Batch Size](https://docs.sambanova.ai/sambastudio/latest/dynamic-batching.html), which improves the model performance significantly. 

In order to use a batching model, first users need to set up the proper endpoint supporting this feature, please [look at this section](#set-up-the-account-and-config-file) for reference. Additionally, users need to specify `number of workers > 1`, either using [the streamlit app](#using-streamlit-app) or [the terminal](#using-terminal). Since the current maximum batch size is 16, it's recomended to choose a value for `number of workers` equal or greater than that. 

About results, users will notice an increase in latency per request, but a higher number of completed requests per minute, compared against non-batching models. Moreover, overall throughput will be higher, and end-to-end latency of the whole process lower.

Here's an example using a Meta-Llama-3-8B-Instruct endpoting with dynamic batching size.

Non-batching setup
- Parameters:
  - Number of requests: 32
  - Number of concurrent workers: 1
- Results:
  - Overall Output Throughput: 321.23 tokens/s
  - Completed Requests Per Minute: 19.27 

Batching setup
- Parameters:
  - Number of requests: 32
  - Number of concurrent workers: 32
- Results:
  - Overall Output Throughput: 810.48 tokens/s
  - Completed Requests Per Minute: 48.63

# Third-party tools and data sources 

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- streamlit (version 1.34.0)
- st-pages (version 0.4.5)
- [llmperf framework](](https://github.com/ray-project/llmperf))
- transformers (version 4.40.1)
- python-dotenv (version 1.0.0)
- Requests (version 2.31.0)
- seaborn (version 0.12.2)