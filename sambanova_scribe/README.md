<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

# SambaNova scribe
======================

Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

Table of Contents:

<!-- TOC -->

- [SambaNova scribe](#sambanova-scribe)
- [Overview](#overview)
- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Install System Dependencies](#install-system-dependencies)
    - [Set up the models, environment variables and config file](#set-up-the-models-environment-variables-and-config-file)
        - [Set up the transcription model](#set-up-the-transcription-model)
        - [Set up the generative model](#set-up-the-generative-model)
- [Deploy the starter kit GUI](#deploy-the-starter-kit-gui)
- [Use the starter kit](#use-the-starter-kit)
- [Customizing the starter kit](#customizing-the-starter-kit)
    - [Transcription params](#transcription-params)
    - [Generation model parameters](#generation-model-parameters)
            - [Experiment with prompt engineering](#experiment-with-prompt-engineering)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

This AI Starter Kit is a simple example of a audio transcription and processing workflow. You send your audio file or a youtube link to app and the audio will be send to the SambaNova platform, and get the transcription and the bullet point summary of the PROVIDED audio.

This sample is ready-to-use. We provide:

- Instructions for setup with SambaNova Cloud.
- Instructions for running the application as is.
- Instructions for customizing the application.

# Before you begin

You have to set up your environment before you can run or customize the starter kit.

## Clone this repository

Clone the starter kit repo.

```bash
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Install System Dependencies

this kit requires you to have installed in your system ffmpeg:

- On macOS, you can manually install them using Homebrew:
```bash
    brew install ffmpeg
```
3. On Linux (Ubuntu/Debian), you can install them manually:
```bash
    sudo apt-get update && sudo apt-get install -y ffmpeg
```
4. On Windows, you may need to install these dependencies manually from the [ffmpeg site](https://ffmpeg.org/download.html) and ensure they are in your system PATH.

## Set up the models, environment variables and config file

### Set up the transcription model

The next step is to set up your environment variables to use one of the transcription models available from SambaNova. You can obtain a free API key through SambaNova Cloud.

- **SambaNova Cloud**: To set up your environment variables.

For more information and to obtain your API key, visit the [SambaNova Cloud webpage](https://cloud.sambanova.ai).

To integrate SambaNova Cloud Transcription models with this AI starter kit, update the API information by configuring the environment variables in the `ai-starter-kit/.env` file:

- Create the `.env` file at `ai-starter-kit/.env` if the file does not exist.
- Enter the transcription SambaNova Cloud API url and key in the `.env` file, for example:

```bash
TRANSCRIPTION_BASE_URL = "https://api.sambanova.ai/v1"
TRANSCRIPTION_API_KEY = "456789abcdef0123456789abcdef0123"
```

### Set up the generative model

The next step is to set up your environment variables to use one of the inference models available from SambaNova. You can obtain a free API key through SambaNova Cloud. Alternatively, if you are a current SambaNova customer, you can deploy your models using SambaStudio.

- **SambaNova Cloud (Option 1)**: Follow the instructions [here](../README.md#use-sambanova-cloud-option-1) to set up your environment variables.
    Then, in the [config file](./config.yaml), set the llm `type` variable to `"sncloud"` and set the `select_expert` config depending on the model you want to use.

# Deploy the starter kit GUI

We recommend that you run the starter kit in a virtual environment or use a container. We also recommend using Python >= 3.10 and < 3.12.

1. Install and update pip.

```bash
    cd ai_starter_kit/sambanova_scribe
    python3 -m venv sambanova_scribe_env
    source sambanova_scribe_env/bin/activate
    pip  install  -r  requirements.txt
```

2. Run the following command:

```bash
    streamlit run streamlit/app.py --browser.gatherUsageStats false 
```

After deploying the starter kit you see the following user interface:

![capture of sambanova scribe demo](./docs/sambanova_scribe_app.png)

# Use the starter kit 

After you've deployed the GUI, you can use the starter kit. Follow these steps:

1. Depending if you have set your env variables you will be prompted or not to set them in the set up bar.

2. In the main panel select the input method either a youtube link or a file.
    > Audios should be mp3, mp4 or wav format
    > Either from youtube download or audio file can not exceed 25MB

3. Click on the Transcribe button this will download the youtube audio or upload your file and generate the transcription of the audio

4. Click on the create summary button to get a bullet point summary of the recording

# Customizing the starter kit

You can further customize the starter kit based on the use case.

## Transcription params

The transcription parameters can be customized in the [config.yaml](./config.yaml) file, the the `audio_model` section you can change the `model` you want to use to transcribe, the `temperature` and the language.

## Generation model parameters

The llm parameters can be customized in the [config.yaml](./config.yaml) file, the the `llm` section you can change the model you want to use in `select_expert` parameter, also parameters like `temperature` or `max_tokens_to_generate`

#### Experiment with prompt engineering

Prompting has a significant effect on the quality of LLM responses. Prompts can be further customized to improve the overall quality of the responses from the LLMs. For example, in this starter kit, the following prompt template was used to generate a bullet point summary from the LLM, where `text` is the transcription of the audio.

```yaml
template: |
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a knowledge base assistant chatbot powered by Sambanova's AI chip accelerator, designed to answer questions based on user-uploaded documents. 
    Use the following pieces of retrieved context to answer the question. Each piece of context includes the Source for reference. If the question references a specific source, then filter out that source and give a response based on that source. 
    If the answer is not in the context, say: "This information isn't in my current knowledge base." Then, suggest a related topic you can discuss based on the available context.
    Maintain a professional yet conversational tone. Do not use images or emojis in your answer.
    Prioritize accuracy and only provide information directly supported by the context. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    \n ------- \n
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

You can make modifications to the prompt template in the following file:

```
    file: prompts/summary.yaml
```

# Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory.
