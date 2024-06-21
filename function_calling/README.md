
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Function Calling kit
======================

# Overview

This function calling kit is an example of tools calling implementation and a generic function calling module that can be used inside your application workflows.

# Before you begin

To use this in your application you need an instruction model, we recommend to use the Meta Llama3 70B or Llama3 8B, either from Sambaverse or from SambaStudio CoE.

## Clone this repository

Clone the starter kit repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the account and config file for the LLM 

The next step sets you up to use one of the models available from SambaNova. It depends on whether you're a SambaNova customer who uses SambaStudio or you want to use the publicly available Sambaverse. 

### Setup for SambaStudio users

To perform this setup, you must be a SambaNova customer with a SambaStudio account.

1. Log in to SambaStudio and get your API authorization key. The steps for getting this key are described [here](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html#_acquire_the_api_key).
2. Select the model you want to use (e.g. CoE containing Meta-Llama-Guard-2-8B) and deploy an endpoint for inference. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).
3. In the repo root directory create an env file in  `sn-ai-starter-kit/.env`, and update it with your Sambastudio endpoint variables ([view your endpoint information](https://docs.sambanova.ai/sambastudio/latest/endpoints.html#_view_endpoint_information)), Here's an example:

    ``` bash
    BASE_URL="https://api-stage.sambanova.net"
    PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
    API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
    ```

### Setup for Sambaverse users 

1. Create a Sambaverse account at [Sambaverse](sambaverse.sambanova.net) and select your model. 
2. Get your [Sambaverse API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) (from the user button).
3. In the repo root directory create an env file in `sn-ai-starter-kit/.env` and specify the Sambaverse API key (with no spaces), as in the following example:

    ``` bash
        SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
    ```

###  Install dependencies

We recommend that you run the starter kit in a virtual environment or use a container. 

NOTE: python 3.10 or higher is required to use this kit.

1. Install the python dependencies in your project environment.

    ```bash
    cd ai_starter_kit/function_calling
    python3 -m venv function_calling_env
    source function_calling_env/bin/activate
    pip  install  -r  requirements.txt
    ```

# Use the Funtion Calling kit 

We provide a simple module for using the Function Calling LLM, for this you will need:

1. Create your set of tools:

    You should create a set of tools that you want the model to be able to use, those tools, should be langchain tools.

    We provide an example of different langchain integrated tools and implementation of custom tools in [src/tools.py](src/tools.py), and in the [step by step notebook](./notebooks/function_calling_guide.ipynb).

    > See more in [langchain tools](https://python.langchain.com/v0.1/docs/modules/tools/)

2. Instantiate your Function calling LLM, passing the model to use (either Sambaverse or Sambastudio), and the required tools as argument, then you can invoke the function calling pipeline using the `function_call_llm` method passing the user query

    ``` python
        from function_calling.src.function_calling  import FunctionCallingLlm
        
        ### Define your tools
        from function_calling.src.tools import get_time, calculator, python_repl
        tools = [get_time, calculator, python_repl]

        fc = FunctionCallingLlm("sambaverse", tools)

        fc.function_call_llm("<user query>", max_it=5, debug=True)
    ```

    we provide an [usage notebook](notebooks/usage.ipynb) that you can use as a guide for using the function calling module

# Customizing the Funtion Calling module

The example module can be further customized based on the use case.

The complete tools generation, methods, prompting and parsing for implementing function calling, can be found and further customized for your specific use case following the [Guide notebook](function_calling_guide.ipynb)  

# Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory. Some of the main packages are listed below:

* python-dotenv (version 1.0.1)
* langchain (version 0.2.3)
* langchain-community (version 0.2.4)
* langchain-experimental (version 0.0.6)
* sseclient-py (version 1.8.0)