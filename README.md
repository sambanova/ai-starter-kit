<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="./images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

# SambaNova AI Starter Kits

# Overview

SambaNova AI Starter Kits are a collection of open-source examples and guides to facilitate the deployment of AI-driven use cases in the enterprise.

To run these examples, you need access to a SambaStudio environment with your models deployed to endpoints. Most code examples are written in Python, though the concepts can be applied in any language.

Questions? Just <a href="https://discord.gg/XF5Sf2sa" target="_blank">message us</a> on Discord <a href="https://discord.gg/XF5Sf2sa" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!


# AI Starter Kits - Ecosystem

## Data Ingestion & Preparation
* 🔍 Data Extraction
* 📊 Web-crawled Data Retriever
* 🧠 Synthetic Q&A Pair Creation

## Model Development & Optimization
* 🚀 YoDA - Your Data, Your Model Recipe
* 🎯 Q&A Fine-tuning
* 💾 SQL Model Fine-tuning
* 🔬 Fine-tuning Embeddings

## Intelligent Information Retrieval
* 📚 Enterprise Knowledge Retriever
* 🔎 Search Assistant
* 🖼️ Image Search
* 🖼️ Multi-Modal Knowledge Retriver

## Advanced AI Capabilities
* 🧭 CoE Routing
* 💡 Prompt Engineering
* 🔀 Performance Benchmarking
* 🔧 Function Calling

## Deployment & Integration
* 🖥️ Endpoint Creation
* 🔗 Samba Apps Integration
* 🐳 Docker Images


# 
# Available AI Starter Kits

<table style="width: 100%;">
<tbody>

<tr>
<td width="25%"><a href="benchmarking/README.md">Benchmarking</a></td>
<td width="75%">This kit evaluates the performance of multiple LLM models hosted in SambaStudio. It offers various performance metrics and configuration options. Users can also see these metrics within a chat interface. </td>
</tr>

<tr>
<td width="25%"><a href="code_copilot/README.md">Code Copilot</a></td>
<td width="75%">This example guide shows a simple integration with Continue VSCode and JetBrains extension using SambaNova platforms, to use Sambanova's hosted models as your custom coding assistant. </td>
</tr>

<tr>
<td width="25%"><a href="workshops/genai_summit/complex_rag/README.md">Complex RAG</a></td>
<td width="75%"> Sample implementation of a complex RAG workflow using the SambaNova platform to get answers to questions about your documents. Includes a runnable demo. </td>
</tr>

<tr>
<td width="25%"><a href="CoE_jump_start/README.md">CoE jump start</a></td>
<td width="75%">This kit demonstrates how to call <a href=https://coe-1.cloud.snova.ai/>SambaNova CoE</a> models using the Langchain framework. The script offers different approaches for calling CoE models, including using Sambaverse, using SambaStudio with a named expert, and using SambaStudio with routing.</td>
</tr>

<tr>
<td width="25%"><a href="data_extraction/README.md">Data Extraction</a></td>
<td width="75%">Series of notebooks that demonstrate methods for extracting text from documents in different input formats.</td>
</tr>
<tr>
<td width="25%"><a href="edgar_qna/README.md">EDGAR Q&A</a></td>
<td width="75%">Example workflow that uses the SambaNova platform to answer questions about organizations using their 10-K annual reports. Includes a runnable local demo and a Docker container to simplify remote deployment.</td>
</tr>

<tr>
<td width="25%"><a href="enterprise_knowledge_retriever/README.md">Enterprise Knowledge Retrieval</a></td>
</td>
<td width="75%">Sample implementation of the semantic search workflow using the SambaNova platform to get answers to questions about your documents. Includes a runnable demo.</td>
</tr>

<tr>
<td width="25%"><a href="fine_tuning_embeddings/README.md"> Fine tuning embeddings</a></td>
<td width="75%">Example workflow for fine-tuning embeddings from unstructured data, leveraging Large Language Models (LLMs) and open-source embedding models to enhance NLP task performance.</td>
</tr>

<tr>
<td width="25%"><a href="fine_tuning_sql/README.md"> Fine tuning SQL</a></td>
<td width="75%">Example workflow for fine-tuning an SQL model for Question-Answering purposes, leveraging Large Language Models (LLMs) and open-source embedding models to enhance SQL generation task performance.</td>
</tr>

<tr>
<td width="25%"><a href="function_calling/README.md"> Function Calling</a></td>
<td width="75%">Example of tools calling implementation and a generic function calling module that can be used inside your application workflows.</td>
</tr>
<td width="25%"><a href="image_search/README.md">Image Search</a></td>
<td width="75%">This example workflow shows a simple approach to image search by image description or image similarity. All workflows are built using the SambaNova platform. </td>
</tr>

<tr>
<td width="25%"><a href="multimodal_knowledge_retriever/README.md">Multimodal Knowledge Retriever</a></td>
<td width="75%"> Sample implementation of the semantic search workflow leveraging the SambaNova platform to get answers using text, tables, and images to questions about your documents. Includes a runnable demo. </td>
</tr>

<tr>
<td width="25%"><a href="post_call_analysis/README.md">Post Call Analysis</a></td>
<td width="75%">Example workflow that shows a systematic approach to post-call analysis including Automatic Speech Recognition (ASR), diarization, large language model analysis, and retrieval augmented generation (RAG) workflows. All workflows are built using the SambaNova platform. </td>
</tr>

<tr>
<td width="25%"><a href="prompt_engineering/README.md">Prompt Engineering</a></td>
</td>
<td width="75%">Starting point demo for prompt engineering using SambaNova's API to experiment with different use case templates. Provides useful resources to improve prompt crafting, making it an ideal entry point for those new to this AISK.</td>
</tr>

<tr>
<td width="25%"><a href="search_assistant/README.md">Search Assistant</a></td>
<td width="75%">Sample implementation of the semantic search workflow built using the SambaNova platform to get answers to your questions using search engine snippets, and website crawled information as the source. Includes a runnable demo.</td>
</tr>

<tr>
<td width="25%"><a href="web_crawled_data_retriever/README.md">Web Crawled Data Retrieval</a></td>
<td width="75%">Sample implementation of a semantic search workflow built using the SambaNova platform to get answers to your questions using website crawled information as the source. Includes a runnable demo.</td>
</tr>

<tr>
<td width="25%"><a href="yoda/README.md">YoDA: Your Data Your model</a></td>
<td width="75%">Sample training recipe to train a Language Model (LLM) using a customer's private data. </td>
</tr>

</tbody>
</table>

# Get started with SambaNova AI starter kit

## Setting your model

### Use Sambaverse models (Option 1)

Sambaverse allows you to interact with multiple open-source models. You can view the list of available models and interact with them in the [playground](https://sambaverse.sambanova.ai/playground).

Please note that Sambaverse's free offering is performance-limited. Companies that are ready to evaluate the production tokens-per-second performance, volume throughput, and 10x lower total cost of ownership (TCO) of SambaNova should [contact us](https://sambaverse.sambanova.ai/contact-us) for a non-limited evaluation instance.

Begin by creating a [Sambaverse](https://sambaverse.sambanova.net) account, then [get your API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) from the username button. Use the available models.

### Deploy your model in SambaStudio (Option 2)

Begin by deploying your LLM of choice (e.g. Llama 2 13B chat, etc) to an endpoint for inference in SambaStudio. Use either the GUI or CLI, as described in the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).

## Integrate your model in the starter kit

Integrate your LLM deployed on SambaStudio with this AI starter kit in two simple steps:

### 1. Clone this repo

```
  git clone https://github.com/sambanova/ai-starter-kit.git
```

### 2. Update API information for the SambaNova LLM

These are represented as configurable variables in the environment variables file in `sn-ai-starter-kit/.env`.

#### SambaStudio deployed model

For example, enter an endpoint with the URL
"https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
in the env file (with no spaces) as:

```
BASE_URL="https://api-stage.sambanova.net"
PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
```

#### Sambaverse model

Enter a Sambaverse API key, for example
"456789ab-cdef-0123-4567-89abcdef0123",
in the env file (with no spaces) as:

```
SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
```

### 3. Update API information for SambaNova Embeddings model (optional).

You can use SambaStudio E5 embedding model endpoint instead of using default in cpu HugginFace embeddings to increase inference speed, follow [this guide](https://docs.sambanova.ai/sambastudio/latest/e5-large.html#_deploy_an_e5_large_v2_endpoint) to deploy your SambaStudio embedding model

> _be sure to set batch size model parameter to 32_

Update API information for the SambaNova embedding endpoint. These are represented as configurable variables in the environment variables file in the root repo directory **`sn-ai-starter-kit/.env`**. For example, an endpoint with the URL
"https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
would be entered in the env file (with no spaces) as:

```
EMBED_BASE_URL="https://api-stage.sambanova.net"
EMBED_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
EMBED_ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
EMBED_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
```

> Note that using different embedding models (cpu or sambastudio) may change the results, and change the way they are set and their parameters
>
> with **CPU Huggingface embeddings**:
>
> ```python
>            embeddings = HuggingFaceInstructEmbeddings(
>                model_name="hkunlp/instructor-large",
>                embed_instruction="",
>                query_instruction="Represent this sentence for searching relevant passages:",
>                encode_kwargs={"normalize_embeddings": True},
>            )
> ```
>
> with **Sambastudio embeddings**:
>
> ```pyhton
> embeddings = SambaNovaEmbeddingModel()
> ```

### 4. Run the desired starter kit

Go to the `README.md` of the starter kit you want to use and follow the instructions. See [Available AI Starter Kits](#available-ai-starter-kits).

## Use Sambanova's LLMs and **Langchain** wrappers

### LLM Wrappers

Set your environment as shown in [integrate your model](#integrate-your-model-in-the-starter-kit).

#### Using Sambaverse LLMs

1. Import the **samabanova_endpoint** langchain wrapper in your project and define your **SambaverseEndpoint** LLM:

```python
from utils.sambanova_endpoint import SambaverseEndpoint

load_dotenv('.env')

llm = SambaverseEndpoint(
    sambaverse_model_name="Meta/llama-2-7b-chat-hf",
    model_kwargs={
      "do_sample": False,
      "temperature": 0.0,
      "max_tokens_to_generate": 512,
      "select_expert": "llama-2-7b-chat-hf"
      },
)
```

2. Use the model

```python
llm.invoke("your prompt")
```

#### Using Sambastudio LLMs

1. Import the **samabanova_endpoint** langchain wrapper in your project and define your **SambaNovaEndpoint** LLM:

```python
from utils.sambanova_endpoint import SambaNovaEndpoint

load_dotenv('.env')

llm = SambaNovaEndpoint(
    model_kwargs={
      "do_sample": False,
      "max_tokens_to_generate": 512,
      "temperature": 0.0
      },
)
```

2. Use the model

```python
llm.invoke("your prompt")
```

See [utils/usage.ipynb](./utils/usage.ipynb) for an example.

### Embedding Wrapper

1. Import the **samabanova_endpoint** langchain wrapper in your project and define your **SambaNovaEmbeddingModel** embedding:

```python
from utils.sambanova_endpoint import SambaNovaEmbeddingModel

load_dotenv('.env')

embedding = SambaNovaEmbeddingModel()
```

2. Use your embedding model in your langchain pipeline

See [utils/usage.ipynb](./utils/usage.ipynb) for an example.

---

## Setting Up Your Environment

There are two approaches to setting up your environment for the AI Starter Kits:

1. **Individual Kit Setup (Traditional Method)**
2. **Base Environment Setup (Recommended)**

### 1. Individual Kit Setup

Each starter kit has its own `README.md` and `requirements.txt` file. You can set up a separate virtual environment for each kit by following the instructions in their respective directories. This method is suitable if you're only interested in running a single kit or prefer isolated environments for each project.

To use this method:
1. Navigate to the specific kit's directory
2. Create a virtual environment
3. Install the requirements
4. Follow the kit-specific instructions

### 2. Base Environment Setup 

For users who plan to work with multiple kits or prefer a unified development environment, we recommend setting up a base environment. This approach uses a Makefile to automate the setup of a consistent Python environment that works across all kits.

Benefits of the base environment approach:
- Consistent Python version across all kits
- Centralized dependency management
- Simplified setup process
- Easier switching between different kits

#### Prerequisites

- **Poetry**: The Makefile will attempt to install Poetry if it's not already installed.
- **pyenv**: The Makefile will attempt to install pyenv if it's not already installed.

#### What the Base Setup Does

1. Installs pyenv and Poetry if they are not already installed.
2. Sets up a Python virtual environment using a specified Python version (default is 3.11.3).
3. Installs all necessary dependencies for the base environment.
4. Sets up the parsing service required by some kits.


#### Setting Up the Base Environment


1. **Install and Set Up the Base Environment:**

```bash
make all
```
This command will set up the base ai-starter-kit environment, including installing all necessary tools and dependencies.

2. **Activate the Base Environment:**

```bash
source .venv/bin/activate
```

3. **Navigate to Your Chosen Starter Kit:**
```bash
cd path/to/starter_kit
```
Within the starter kit there will be instructions on how to start the kit. You can skip the virtual environment creation 
part in the kits README.md as we've done it here.


4. **Enterprise Knowledge Retriever One Click Deploy:**
For the EKR Kit specifically we have enabled a one click deploy via makefile


### Enterprise Knowledge Retriever (EKR) Setup

To set up and run the Enterprise Knowledge Retriever:

1. **Set Up and Start EKR:**

```bash
make ekr
```
This command sets up the EKR environment, starts the parsing service, and launches the EKR application.


### Parsing Service Management
For certain kits, we utilise a standard parsing service. To work with this service, following the steps in this section.

- **Start Parsing Service:**
```bash
make start-parsing-service
```

- **Stop Parsing Service:**
```bash
make stop-parsing-service
```

- **Check Parsing Service Status:**
```bash
make parsing-status
```

- **View Parsing Service Logs:**
```bash
make make parsing-log
```

### Cleanup

To clean up all virtual environments created by the makefile and stop parsing services run the following command:
```bash
make clean
```
This command removes all virtual environments created with the makefile, stops the parsing service, and cleans up any temporary files.


## Troubleshooting

If you encounter issues while setting up or running the AI Starter Kit, here are some common problems and their solutions:

### Lock file mismatch

If you see an error message like this:

```
ERROR
Installing dependencies... Installing dependencies from lock file
pyproject.toml changed significantly since poetry.lock was last generated. Run `poetry lock [--no-update]` to fix the lock file. make: *** [install] Error 1
```

This means there's a mismatch between your `pyproject.toml` and `poetry.lock` files. To resolve this:

1. Run the following command to update the lock file:
   ```
   poetry lock
   ```

2. Then, try running `make all` again.

### Python version issues

If you're having problems with Python versions:

1. Ensure you have pyenv installed: `make ensure-pyenv`
2. Install the required Python versions: `make install-python-versions`
3. If issues persist, check your system's Python installation and PATH settings.

### Dependency conflicts

If you're experiencing dependency conflicts:

1. Try cleaning your environment: `make clean`
2. Update the lock file: `poetry lock --no-update`
3. Reinstall dependencies: `make install`

### Parsing service issues

If the parsing service isn't starting or is behaving unexpectedly:

1. Check its status: `make parsing-status`
2. View its logs: `make parsing-log`
3. Try stopping and restarting it: `make stop-parsing-service` followed by `make start-parsing-service`

### General troubleshooting steps

1. Ensure all prerequisites (Python, pyenv, Poetry) are correctly installed.
2. Try cleaning and rebuilding the environment: `make clean all`
3. Check for any error messages in the console output and address them specifically.
4. Ensure your `.env` file is correctly set up in the ai-starter-kit root with all necessary environment variables.

If you continue to experience issues, please [open an issue](https://github.com/sambanova/ai-starter-kit/issues/new) with details about your environment, the full error message, and steps to reproduce the problem.

### Important Notes for Users

- Ensure you have sufficient permissions to install software on your system.
- The setup process may take several minutes, especially when installing Python versions or large dependencies.
- If you encounter any issues during setup, check the error messages and ensure your system meets all prerequisites.
- Always activate the base environment before navigating to and running a specific starter kit.
- Some kits may require additional setup steps. Always refer to the specific README of the kit you're using.


**Note:** These AI Starter Kit code samples are provided "as-is," and are not production-ready or supported code. Bugfix/support will be on a best-effort basis only. Code may use third-party open-source software. You are responsible for performing due diligence per your organization policies for use in your applications.