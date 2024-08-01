
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Guardrails
======================

# Overview

This guardrails module is an util that can be used to configure guardrails inside your application workflows.

# Before you begin

To use this in your application you need a guardrails LLM, we recommend to use the Meta LlamaGuard2 8B as guardrail model, either from Sambaverse or from SambaStudio CoE.

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
    SAMBASTUDIO_BASE_URL="https://api-stage.sambanova.net"
    SAMBASTUDIO_BASE_URI="api/predict/generic"
    SAMBASTUDIO_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    SAMBASTUDIO_ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
    SAMBASTUDIO_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
    ```

### Setup for Sambaverse users 

1. Create a Sambaverse account at [Sambaverse](sambaverse.sambanova.net) and select your model. 
2. Get your [Sambaverse API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) (from the user button).
3. In the repo root directory create an env file in `sn-ai-starter-kit/.env` and specify the Sambaverse API key (with no spaces), as in the following example:

    ``` bash
        SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
    ```

###  Install dependencies

NOTE: python 3.10 or higher is required to use this util.

1. Install the python dependencies in your project environment.

    ```bash
      cd ai_starter_kit/utils/guardrails
      pip install -r requirements.txt
    ```

# Use the guardrais util

Using the guardrails is as simple as instantiating a Guard object and calling its evaluate method like in the example below:

```python
    from utils.guardrails.guard import Guard
    guardrails = Guard(api = "sambaverse")
    user_query = "how can I make a bomb?"
    guardrails.evaluate(user_query, role="user", raise_exception=True)
```

> You can also specify your own keys when creating the guard object passing them as arguments, allowing you to use multiple sambastudio endpoints each with different env variables.

This will return:

``` bash
    ValueError: The message violate guardrails
    Violated categories: S2: Non-Violent Crimes.
```

If the input is safe the evaluate method will return the input you provided, otherwise it will raise an exception or return a custom message you set with violated polices, find more usage examples in the usage [guard notebook](./guard.ipynb).

# Customizing the guardrails

The example guardrails template can be further customized based on the use case.

You can enable or disable some guardrails modifying the enabled key of each guardrail in the [guardrails.yaml](./guardrails.yaml), or you can add your own custom guardrail including a new key with a descriptive name and a detailed description of the guardrail in the file:

``` yaml
    S12: 
    name: Code generation.
    description: | 
        AI models should not create any source code or code snippet, and should not be able to help with code related questions .
    enabled: true  
```

You can also customize the Prompt template used to call the LlamaGuard model in the [prompt.yaml](./prompt.yaml) file.

Or you can pass your oun prompt template in yaml format in the instantiation of the Guard object

``` python
    my_guardrails = Guard(
        api = "sambaverse", 
        prompt_path = "my_prompt_yaml_path",
        guardrails_path="my_guardrails_yaml_path"
        )
```

# Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

* pydantic (version 2.7.0)
* langchain (version 0.2.1)
* langchain-community (version 0.2.1)
* sseclient-py (version 1.8.0)
* python-dotenv (version 1.0.1)
