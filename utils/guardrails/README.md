
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>

Guardrails
======================

<!-- TOC -->

- [Guardrails](#guardrails)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the models and config file](#set-up-the-models-and-config-file)
        - [Set up the inference endpoint, configs and environment variables](#set-up-the-inference-endpoint-configs-and-environment-variables)
        - [Install dependencies](#install-dependencies)
- [Use the guardrais util](#use-the-guardrais-util)
- [Customizing the guardrails](#customizing-the-guardrails)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

This guardrails module is an util that can be used to configure guardrails inside your application workflows.

# Before you begin

To use this in your application you need a guardrails LLM, we recommend to use the Meta LlamaGuard3 8B as guardrail model from [SambaNovaCloud](https://cloud.sambanova.ai/) models.

## Clone this repository

Clone the starter kit repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the models and config file

### Set up the inference endpoint, configs and environment variables

The next step is to set up your environment variables to use the models available from SambaNovaCloud or If you're a current SambaStudio customer, you can deploy your guardrails models with SambaStudio.

- If using **SambaNovaCloud** Please follow the instructions [here](../../README.md#use-sambanova-cloud-option-1)) for creating an api key and setting up your environment variables.
    Then when setting the [Guard instance](guard.ipynb) set the llm `api` variable to `"sncloud"`, and set `model` config.

- If using **SambaStudio** Please follow the instructions [here](../../README.md#use-sambastudio-option-2) for setting up endpoint and your environment variables.
    Then when setting the [Guard instance](guard.ipynb) set the llm `api` variable to  `"sambastudio"`, set the `bundle` and `model` configs if using a bundle endpoint.

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
    guardrails = Guard(api = "sncloud", model 'Meta-Llama-3.1-8B-Instruct')
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
        api = "sncloud", 
        prompt_path = "my_prompt_yaml_path",
        guardrails_path="my_guardrails_yaml_path"
        )
```

# Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

* pydantic (version 2.7.0)
* langchain (version 0.3.9)
* langchain-community (version 0.3.9)
* sseclient-py (version 1.8.0)
* python-dotenv (version 1.0.1)