
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Code Copilot
======================

<!-- TOC -->

- [Code Copilot](#code-copilot)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Set up the account for using the LLM](#set-up-the-account-for-using-the-llm)
        - [Setup for Sambanova Cloud users](#setup-for-sambanova-cloud-users)
        - [Setup for SambaStudio users](#setup-for-sambastudio-users)
    - [Get Continue Extension/plugin](#get-continue-extensionplugin)
- [Setting up](#setting-up)
    - [Setup Continue](#setup-continue)
    - [Set the custom SambaNova integration](#set-the-custom-sambanova-integration)
- [Usage](#usage)
    - [Ask question to the LLM in your IDE](#ask-question-to-the-llm-in-your-ide)
    - [Ask about selected code](#ask-about-selected-code)
    - [Edit code](#edit-code)
    - [Understand terminal errors](#understand-terminal-errors)
    - [Custom commands](#custom-commands)
- [Customizing the connector](#customizing-the-connector)
    - [Modify the parameters and the model to use](#modify-the-parameters-and-the-model-to-use)
    - [Add custom commands](#add-custom-commands)
- [Acknowledgments](#acknowledgments)

<!-- /TOC -->

# Overview

This AI Starter Kit is a demonstration of how to use Sambanova's models as coding assistants leveraging the [Continue](https://www.continue.dev/) extension for VSCode and JetBrains.  The Kit includes:

- A configurable SambaNova Cloud - Continue connector. The connector generates answers from a SambaNova cloud hosted model.

- A configurable SambaStudio - Continue connector. The connector generates answers from a deployed LLM.

- An installation, setup, and usage guide.

# Before you begin

For this starter kit, you will need access to an SambaNova Cloud account or a SambaStudio Environment.
> You can use a LLM of your choice, either from SambaNova Cloud or from SambaStudio, but is highly recommended to use Meta-Llama-3-8B-Instruct

## Set up the account for using the LLM 

The next step sets you up to use one of the models available from SambaNova. It depends on whether you're a SambaNova customer who uses SambaStudio or you want to use the publicly available SambaNova Cloud.

### Setup for Sambanova Cloud users 

1. Create a SambaNova Cloud account at [SambaNova Cloud](https://cloud.sambanova.ai).

2. Get your [SambaNova Cloud API key](https://cloud.sambanova.ai/apis) 

### Setup for SambaStudio users

To perform this setup, you must be a SambaNova customer with a SambaStudio account.

1. Log in to SambaStudio and select the LLM to use (e,g. Bundle1.1 with Llama 3 8B instruct) and deploy an endpoint for inference. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).

2. Get your endpoint API authorization key. as shown [here](https://docs.sambanova.ai/sambastudio/latest/endpoints.html#_endpoint_api_keys).

## Get Continue Extension/plugin

you will need to install the [Continue](https://www.continue.dev/) [VSCode](https://marketplace.visualstudio.com/items?itemName=Continue.continue) or [JetBrains](https://plugins.jetbrains.com/plugin/22707-continue) Extension. *this can be done directly from the extensions/plugins section of you IDE searching ***Continue****

# Setting up

## Setup Continue

After installing Continue you will need to do the basic setup

- First you will be prompted to select a provider Select SambaNova Cloud, and enter your SambaNovaCLoud api key, then click **Connect** and close the ***Continue** window

## Add SambaNovaCloud default models

You should set the ***Continue*** config.json file. press ⌘+l this will open a new ***Continue*** session, Click in the gear ⚙ bottom right button, the the json file will open, replace the contents of this file with the contents of the [config.json](config.json) kit provided file, and update the `apiKey` model fields with your SambaNovaCloud API Key

This will add to ***Continue*** the following models

- `Meta-Llama-3.1-8B-Instruct`
- `Meta-Llama-3.1-70B-Instruct`
- `Meta-Llama-3.1-405B-Instruct`

And will add `Meta-Llama-3.2-1B-Instruct` as tab autocompletion model.

### Add SambaStudio models

If you want to use models deployed in your SambaStudio environment see the steps in [customizing the connector section](#set-the-custom-sambanova-integration-optional)

# Usage

## Ask question to the LLM in your IDE

You can interact with the LLM directly using ⌘+l command this will open a ***Continue*** session in a new extension window then ask anything to the to the model!

## Ask about selected code

You can ask information of a selected snippet of code to the LLM directly selecting your text and then pressing ⌘+l, this will open a ***Continue*** session in a new extension window with the code snippet as context, then ask anything related with your code!

## Edit code

You can ask your LLM to modify your code, add functionalities, documentation etc, over a selected code snippet, for this first select the code snippet you want the model to modify, then press ⌘+i, this will open an input bar in the top of your IDE, then write your desired changes and press enter, this will generate the modified code for you and you can edit it, accept or reject the proposed changes.

## Understand terminal errors

You can ask the model to inspect your terminal error outputs to explain you the error and give you some suggestions, for this after getting an error in your terminal only press ⌘+shit+r, this will open a ***Continue*** session in a new extension window with the error explanation!

## Code autocompletion
check in your IDE bottom bar the ***Continue*** button and click on Enable Autocomplete, you will see a tick mark indication is enabled, then in all files you will see auto completion suggestions you can accept pressing tab.

## Custom Actions 

You can execute your custom actions/prompts selecting a code snippet and then pressing ⌘+l to open a new ***Continue*** session then write `/<yourCommand>` to generate, see how to create your custom commands [here](#add-custom-actions)

> See more about ***Continue*** extension usage [here](https://docs.continue.dev/how-to-use-continue)

# Customizing the connector

The example template can be further customized based on the use case.

## Add custom actions

You can add your custom commands adding them to the [config.json file](config.json)
 
A custom command should have the following structure

```json
{
  "name": "yourCommand",
  "prompt": "{{{ input }}} \n\n custom prompt",
  "description": "Description of your custom action"
}
```

## Set the custom SambaNova integration (Optional)

After the basic installation you can set the custom SambaNovaCloud or the SambaStudio ***Continue*** connectors

First you should modify the ***Continue*** `config.ts` file

- Open the `config.ts` file in  `~/.continue/` folder and replace the contents with the contents of the [config.ts](config.ts) kit provided file.

- If you are using SambaNova Cloud: replace the sambanova_api_key variable  with your previously generated SambaNova Cloud api key

    For example, for an api key `123456ab-cdef-0123-4567-890abcdef` update the first section in the config.ts file as:

    ```ts
    //SambaNova Cloud usage
    const sambanova_api_key = "123456ab-cdef-0123-4567-890abcdef"
    ```

- If you are using SambaStudio: replace the sambastudio_base_url, sambastudio_project_id, sambastudio_endpoint_id and sambastudio_api_key variables with your SambaStudio endpoint keys:

    For example, for an endpoint with the URL `https://api-stage.sambanova.net/api/predict/generic/stream/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef012` with api key `123456ab-cdef-0123-4567-890abcdef` update the first section in the config.ts file as:

    ```ts
    //SambaStudio usage 
    const sambastudio_base_url = "https://api-stage.sambanova.net";
    const sambastudio_project_id = "12345678-9abc-def0-1234-56789abcdef0";
    const sambastudio_endpoint_id = "456789ab-cdef-0123-4567-89abcdef012";
    const sambastudio_api_key = "123456ab-cdef-0123-4567-890abcdef"
    ```

In the [~/.continue/config.ts](~/.continue/config.ts) script you can add new SambaNovaCloud and SambaStudio models with the possibility of customize parameters like temperature, top_p, top_k, do_sample and others, even call a more complex service using sambaNova Models in behind.


## Modify the parameters and the model to use

You can change the default SambaNova Cloud model or the default model used with SambaStudio Bundle models modifying the model and expert selection in [config.ts](config.ts) file:

- If using SambaNova Cloud:
    ```ts
        const sambanova_model = "llama3-405b";
    ```

- If using SambaStudio Bundle:
    ```ts
        const sambastudio_use_bundle = true;
        const sambastudio_model = "Meta-Llama-3-8B-Instruct";
    ```

Also you can modify model parameters modifying the `body` params of the SambaNova Cloud model and Sambastudio model in the `SambastudioModel` an `SambaNovaCloudModel` definitions of [config.ts file](config.ts).

# Acknowledgments

This kit work aims to show the integration of SambaNova Systems RDU acceleration models, The work here has been leveraged and adapted from the [Continue](https://docs.continue.dev/) documentation. The original docs can be found in [https://docs.continue.dev/](https://docs.continue.dev/)