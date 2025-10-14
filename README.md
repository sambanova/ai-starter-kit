<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="./images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>

# SambaNova AI Starter Kits

# Overview

SambaNova AI Starter Kits are a collection of open-source examples and guides designed to facilitate the deployment of AI-driven use cases for both developers and enterprises.

To run these examples, you can obtain a free API key using [SambaNova Cloud](https://cloud.sambanova.ai). Alternatively, if you are a current SambaNova customer, you can deploy your models using [SambaStudio](https://docs.sambanova.ai/sambastudio/latest/index.html). Most of the code examples are written in Python, although the concepts can be applied to any programming language.

Questions? Just <a href="https://community.sambanova.ai/latest" target="_blank">message us</a> on SambaNova Community <a href="https://community.sambanova.ai/latest" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Community" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

# Available AI Starter Kits

The table below lists the available kits, which are grouped into four categories: 1) Data Ingestion & Preparation, 2) Model Development & Optimization, 3) Intelligent Information Retrieval, and 4) Advanced AI Capabilities.

For functionalities related to third-party integrations, find a list in our [Integrations Repository](https://github.com/sambanova/integrations) and [Integrations Docs](https://docs.sambanova.ai/cloud/docs/integrations).

**Note**: For each kit, we specify whether it is compatible with SambaNova Cloud, SambaStudio, or both.

<table style="width: 100%;">
<thead>
<tr>
<th width="20%">Name</th>
<th width="45%">Kit Description</th>
<th width="20%">Compatible APIs</th>
<th width="15%">Category</th>
  
</tr>
</thead>
  
<tbody>

<tr>
<td width="20%"><a href="data_extraction/README.md">Data Extraction</a></td>
<td width="40%">Series of notebooks that demonstrate methods for extracting text from documents in different input formats.</td>
<td width="20%"> SambaNova Cloud, SambaStudio</td>
<td width="20%"> Data Ingestion & Preparation </td>  
</tr>

<tr>
<td width="20%"><a href="e2e_fine_tuning/README.md">End to End Fine-tuning</a></td>
<td width="40%"> Recipe to upload and train a Language Model (LLM) using your own data in SambaStudio platform. </td>
<td width="20%"> SambaStudio </td>
<td width="20%"> Data Ingestion & Preparation </td>
</tr>

<tr>
<td width="20%"><a href="enterprise_knowledge_retriever/README.md">Enterprise Knowledge Retrieval</a></td>
<td width="40%">Sample implementation of the semantic search workflow using the SambaNova platform to get answers to questions about your documents. Includes a runnable demo.</td>
<td width="20%"> SambaNova Cloud, SambaStudio</td>
<td width="20%"> Intelligent Information Retrieval </td>  
</tr>

<tr>
<td width="20%"><a href="multimodal_knowledge_retriever/README.md">Multimodal Knowledge Retriever</a></td>
<td width="40%"> Sample implementation of the semantic search workflow leveraging the SambaNova platform to get answers using text, tables, and images to questions about your documents. Includes a runnable demo. </td>
<td width="20%"> SambaNova Cloud, SambaStudio</td>
<td width="20%"> Intelligent Information Retrieval </td>   
</tr>

<tr>
<td width="20%"><a href="eval_jumpstart/README.md">RAG Evaluation Kit</a></td>
<td width="40%">A tool for evaluating the performance of LLM APIs using the RAG Evaluation methodology.</td>
<td width="20%">SambaStudio</td>
<td width="20%"> Intelligent Information Retrieval </td>  
</tr>

<tr>
<td width="20%"><a href="search_assistant/README.md">Search Assistant</a></td>
<td width="40%">Sample implementation of the semantic search workflow built using the SambaNova platform to get answers to your questions using search engine snippets, and website crawled information as the source. Includes a runnable demo.</td>
<td width="20%">SambaNova Cloud, SambaStudio</td>
<td width="20%"> Intelligent Information Retrieval </td>   
</tr>

<tr>
<td width="20%"><a href="benchmarking/README.md">Benchmarking</a></td>
<td width="40%">This kit evaluates the performance of multiple LLM models hosted in SambaStudio. It offers various performance metrics and configuration options. Users can also see these metrics within a chat interface.</td>
<td width="20%"> SambaNova Cloud, SambaStudio</td>
<td width="20%"> Advanced AI Capabilities </td>    
</tr>

<tr>
<td width="20%"><a href="https://github.com/sambanova/integrations/tree/main/continue">Code Copilot</a></td>
<td width="40%">This example guide shows a simple integration with Continue VSCode and JetBrains extension using SambaNova platforms, to use Sambanova's hosted models as your custom coding assistant. </td>
<td width="20%"> SambaNova Cloud, SambaStudio</td>
<td width="20%"> Integrations </td>  
</tr>

<tr>
<td width="20%"><a href="financial_assistant/README.md"> Financial Assistant</a></td> 
<td width="40%">This app demonstrates the capabilities of LLMs in extracting and analyzing financial data using function calling, web scraping, and RAG.</td>
<td width="20%"> SambaNova Cloud, SambaStudio</td>
<td width="20%"> Advanced AI Capabilities </td>
</tr>

<tr>
<td width="20%"><a href="function_calling/README.md"> Function Calling</a></td>
<td width="40%">Example of tools calling implementation and a generic function calling module that can be used inside your application workflows.</td>
<td width="20%"> SambaNova Cloud, SambaStudio</td>
<td width="20%"> Advanced AI Capabilities </td>
</tr>

</tbody>
</table>

# Getting Started 

Go to [SambaNova Cloud Quickstart Guide](./quickstart/README.md) If is your first time using the AI State Kits and you want to try out simple examples. Follow the next steps to read more detailed instructions or if you ar a SambaStudio user. 

## Getting a SambaNova API key and setting your generative models

Currently, there are two ways to obtain an API key from SambaNova. You can get a free API key using SambaNova Cloud. Alternatively, if you are a current SambaNova customer, you can deploy your models using SambaStudio. 

### Use SambaNova Cloud (Option 1)

For more information and to obtain your API key, visit the [SambaNova Cloud webpage](https://cloud.sambanova.ai).

To integrate SambaNova Cloud LLMs with this AI starter kit, update the API information by configuring the environment variables in the `ai-starter-kit/.env` file:

- Create the `.env` file at `ai-starter-kit/.env` if the file does not exist.
- Enter the SambaNova Cloud API key in the `.env` file, for example:
  
```bash
SAMBANOVA_API_KEY = "456789abcdef0123456789abcdef0123"
```

### Use SambaStudio (Option 2)

Begin by deploying your LLM of choice (e.g., Llama 3 8B) to an endpoint for inference in SambaStudio. Use either the GUI or CLI, as described in the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).

To integrate your LLM deployed on SambaStudio with this AI starter kit, update the API information by configuring the environment variables in the `ai-starter-kit/.env` file:

- Create the `.env` file at `ai-starter-kit/.env` if the file does not exist.
- Set your SambaStudio variables. For example, an endpoint with the URL
"https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
is entered in the `.env` file as:

``` bash
SAMBASTUDIO_URL="https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
SAMBASTUDIO_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
```

## Setting your embedding models

Currently, you can set your embedding models on SambaNova Cloud, SambaStudio, or CPU.

### Use SambaNovaCloud embeddings (Option 1)

To integrate your embedding model deployed on SambaStudio with this AI starter kit, update the API information by configuring the environment variables in the `ai-starter-kit/.env` file:

- Create the `.env` file at `ai-starter-kit/.env` if the file does not exist.
- Set your SambaNova Cloud variables in `.env` file as:

``` bash
SAMBANOVA_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
```

### Use CPU embedding (Option 2)

You can run the Hugging Face embedding models locally on CPU. In this case, no information is needed in the `.env` file.


### Use SambaStudio embedding (Option 3)

Alternatively, you can use SambaStudio embedding model endpoints instead of the CPU-based HugginFace embeddings to increase inference speed. Please follow [this guide](https://docs.sambanova.ai/sambastudio/latest/e5-large.html#_deploy_an_e5_large_v2_endpoint) to deploy your SambaStudio embedding model.

To integrate your embedding model deployed on SambaStudio with this AI starter kit, update the API information by configuring the environment variables in the `ai-starter-kit/.env` file:

- Create the `.env` file at `ai-starter-kit/.env` if the file does not exist.
- Set your SambaStudio variables. For example, an endpoint with the URL
`"https://api-stage.sambanova.net/api/predict/generic/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"`
is entered in the `.env` file as:

``` bash
SAMBASTUDIO_URL="https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
SAMBASTUDIO_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
```

## Run the desired starter kit

Go to the `README.md` of the starter kit you want to use and follow the instructions. See [Available AI Starter Kits](#available-ai-starter-kits).

## Additional information

<details>
<summary>Use Sambanova's LLMs and Langchain chat model wrappers</summary>

### LLM Wrappers

Set your environment as shown in [integrate your model](#integrate-your-model-in-the-starter-kit).

#### Using Sambastudio LLMs

1. Import the **SambaStudio** langchain integration in your project and define your **SambaStudio* ChatModel:

``` bash
pip install langchain-sambanova
```

- If using a Bundle endpoint:

```python
from langchain_sambanova import ChatSambaStudio

load_dotenv('.env')

llm = ChatSambaStudio(
  max_tokens_to_generate = 512,
  temperature = 0.0,
  model = "Meta-Llama-3-8B-Instruct"
)
```

- If using a single model endpoint

```python
from langchain_sambanova import ChatSambaStudio

load_dotenv('.env')

llm = ChatSambaStudio(
  max_tokens_to_generate = 512,
  temperature = 0.0
)
```

2. Use the model

```python
llm.invoke("your prompt")
```

See [utils/usage.ipynb](./utils/usage.ipynb) for an example.

### Using SambaNova Cloud LLMs

1. Import our **SambaNovaCloud** langchain integration in your project and define your **SambaNovaCloud* ChatModel:


``` bash
pip install langchain-sambanova
```

```python
from langchain_sambanova import ChatSambaNovaCloud

load_dotenv('.env')

llm = ChatSambaNovaCloud(model='Meta-Llama-3.3-70B-Instruct')
```

2. Use the model

```python
llm.invoke("your prompt")
```

See [utils/usage.ipynb](./utils/usage.ipynb) for an example.

### Embedding Wrapper

1. Import the **SambaStudioEmbedding** langchain integration in your project and define your **SambaStudioEmbedding*  embedding:

``` bash
pip install langchain-sambanova
```

- If using a Bundle endpoint

```python
from langchain_sambanova import SambaStudioEmbeddings

load_dotenv('.env')

embedding = SambaStudioEmbeddings(
              batch_size=1,
              model_kwargs = {
                  "select_expert":"e5-mistral-7b-instruct"
                  }
              )
```

- If using a single embedding model endpoint

```python
from langchain_sambanova import SambaStudioEmbeddings

load_dotenv('.env')

embedding = SambaStudioEmbeddings(batch_size=32)
```

> Note that using different embedding models (cpu or sambastudio) may change the results, and change the way they are set and their parameters

2. Use your embedding model in your langchain pipeline

See [utils/usage.ipynb](./utils/usage.ipynb) for an example.

### Javascript Example

1. Before running the code, ensure that you have Node.js installed on your system. You can download the latest version from the official Node.js [website](https://nodejs.org/en).

2. Set Up the Environment. To set up the environment, run the following commands in your terminal:

``` bash
npm init -y
```

``` bash
npm install @langchain/openai @langchain/core
```

> These commands will create a new package.json file and install the required dependencies.

3. Create a new file named `app.js` and add the following code:

```javascript
import { ChatOpenAI } from "@langchain/openai";

const SambaNovaCloudBaseURL = "https://api.sambanova.ai/v1";
const apiKey = "your-api-key";

const SambaNovaCloudChatModel = new ChatOpenAI({
  temperature: 0.9,
  model: "Meta-Llama-3.3-70B-Instruct",
  configuration: {
    baseURL: SambaNovaCloudBaseURL,
    apiKey: apiKey,
  },
});

const response = await SambaNovaCloudChatModel.invoke("Hi there, tell me a joke!");
console.log(response.content);
```

4. To run the app, execute the following command in your terminal:

``` bash
node app.js
```

---

</details>

<details>
<summary>Setting up your virtual environment</summary><br/>

There are two approaches to setting up your virtual environment for the AI Starter Kits:

1. **Individual Kit Setup (Traditional Method)**
2. **Base Environment Setup**

### 1. Individual Kit Setup

Each starter kit (see table [above](#available-ai-starter-kits)) has its own `README.md` and `requirements.txt` file. You can set up a separate virtual environment for each kit by following the instructions in their respective directories. This method is suitable if you're only interested in running a single kit or prefer isolated environments for each project.

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

- **pyenv**: The Makefile will attempt to install pyenv if it's not already installed.
- **Docker**: (Optional) If you want to use the Docker-based setup, ensure Docker is installed on your system.

#### What the Base Setup Does

1. Installs pyenv and Poetry if they are not already installed.
2. Sets up a Python virtual environment using a specified Python version (default is 3.11.3).
3. Installs all necessary dependencies for the base environment.
4. Sets up the parsing service required by some kits.
5. Installs system dependencies like Tesseract OCR and Poppler.
6. Provides Docker-based setup options for consistent environments across different systems.

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


### Parsing Service Management
For certain kits, we utilise a standard parsing service. By Default it's started automatically with the base environment. To work with this service in isolation, following the steps in this section.

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
make parsing-log
```

### Docker-based Setup

To use the Docker-based setup:

1. Ensure Docker is installed on your system.
2. Build the Docker image:

```bash
make docker-build
```

3. Run a specific kit in the Docker container:
```bash
make docker-run-kit KIT=<kit_name>
```
Replace `<kit_name>` with the name of the starter kit you want to run (e.g., `function_calling`).

4. To open a shell in the Docker container:
```bash
make docker-shell
```

### Cleanup

To clean up all virtual environments created by the makefile and stop parsing services run the following command:
```bash
make clean
```
This command removes all virtual environments created with the makefile, stops the parsing service, and cleans up any temporary files.
</details>

<details>
<summary>Troubleshooting</summary><br/>

If you encounter issues while setting up or running the AI Starter Kit, here are some common problems and their solutions:

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

### pikepdf installation issues

If you encounter an error while installing `pikepdf`, such as:

```
ERROR: Failed building wheel for pikepdf
Failed to build pikepdf
```

This is likely due to missing `qpdf` dependency. The Makefile should automatically install `qpdf` for you, but if you're still encountering issues:

1. Ensure you have proper permissions to install system packages.
2. If you're on macOS, you can manually install `qpdf` using Homebrew:
   ```bash
   brew install qpdf
   ```
3. On Linux, you can install it using your package manager, e.g., on Ubuntu:
   ```
   sudo apt-get update && sudo apt-get install -y qpdf
   ```
4. After installing `qpdf`, try running `make install` again.

If you continue to face issues, please ensure your system meets all the requirements for building `pikepdf` and consider checking the [pikepdf documentation](https://pikepdf.readthedocs.io/en/latest/installation.html) for more detailed installation instructions.

### Parsing service issues

If the parsing service isn't starting or is behaving unexpectedly:

1. Check its status: `make parsing-status`
2. View its logs: `make parsing-log`
3. Try stopping and restarting it: `make stop-parsing-service` followed by `make start-parsing-service`

### System Dependencies Issues

If you encounter issues related to Tesseract OCR or Poppler:

1. Ensure the Makefile has successfully installed these dependencies.
2. On macOS, you can manually install them using Homebrew:
 ```bash
   brew install tesseract poppler
   ```
3. On Linux (Ubuntu/Debian), you can install them manually:
 ```bash
   sudo apt-get update && sudo apt-get install -y tesseract-ocr poppler-utils
   ```
4. On Windows, you may need to install these dependencies manually and ensure they are in your system PATH.

### Docker-related Issues

If you're using the Docker-based setup and encounter issues:

1. Ensure Docker is properly installed and running on your system.
2. Try rebuilding the Docker image: `make docker-build`
3. Check Docker logs for any error messages.
4. Ensure your firewall or antivirus is not blocking Docker operations.

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
</details>

### API Reference

- Find more information about SambaNova Cloud [here](https://docs.sambanova.ai/cloud/docs/get-started/overview)

- Find more information about SambaStudio [here](https://docs.sambanova.ai/sambastudio/latest/index.html)

**Note:** These AI Starter Kit code samples are provided "as-is," and are not production-ready or supported code. Bugfix/support will be on a best-effort basis only. Code may use third-party open-source software. You are responsible for performing due diligence per your organization policies for use in your applications.

