<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

# SambaNova Llama Stack Integration

## Overview

Llama Stack is a framework that standardizes core building blocks to simplify AI application development. It encapsulates best practices across the Llama ecosystem, making it easier to build and deploy AI solutions efficiently. Llama Stack consists of two main components:

- Server – A running distribution of Llama Stack that hosts various adaptors.
- Client – A consumer of the server's API, interacting with the hosted adaptors.

> An adaptor is a provider component in Llama Stack, such as an inference model, a safety guardrails model, or an agent. A distribution is a bundle of multiple adaptors deployed on the server, providing a unified interface for AI workflows.

SambaNova offers its own Llama Stack distribution, which includes the following adaptors:

- Inference:
    - remote::sambanova
    - inline::sentence-transformers
- Vector I/O:
    - inline::faiss
    - remote::chromadb
    - remote::pgvector
- Safety:
    - inline::llama-guard
- Agents:
    - inline::meta-reference
- Telemetry:
    - inline::meta-reference
- Tool Runtime:
    - remote::brave-search
    - remote::tavily-search
    - inline::code-interpreter
    - inline::rag-runtime
    - remote::model-context-protocol
    - remote::wolfram-alpha

For more details on Llama Stack, refer to the official [documentation](https://llama-stack.readthedocs.io/en/latest/index.html).

## Setting up the Environment

To get started, you need to create a virtual environment and install the base Llama Stack framework. This will allow you to build the distribution template and integrate SambaNova's adaptors.

1. Create a Virtual Environment
Run the following commands to set up a virtual environment:

``` bash
    python -m venv llamastackenv
    source llamastackenv/bin/activate
```

2. Install Dependencies
Once the environment is activated, install the necessary packages:

``` bash
    pip install uv
    pip install llama-stack
```

with the environment set now you can build the distribution

## Building the SambaNova Distribution  

You can build the SambaNova Llama Stack distribution using either a virtual environment (**venv**), a conda environmnet or a Docker image.  

### Build with venv (Recommended)  

To build the distribution within the currently activated virtual environment, run:

```bash
llama stack build --template sambanova --image-type venv
```

### Build with Docker  

To build the distribution as a Docker container, use the following command:  

```bash
llama stack build --template sambanova --image-type container  
```

After the build process is complete, verify that the image was created by listing available Docker images:  

```bash
docker image list
```

Example Output:

``` bash
REPOSITORY                        TAG       IMAGE ID       CREATED          SIZE
distribution-sambanova            0.1.6     4f70c8f71a21   5 minutes ago    1.3GB
```


### Build with conda

To build the distribution in a conda environment, run:

``` bash
llama stack build --template sambanova --image-type conda
```

Now The distribution is built the Llama Stack server can be deployed

## Running the SambaNova Distribution

Before deploying the distribution, set the required environment variables:

```bash
export INFERENCE_MODEL="meta-llama/Llama-3.3-70B-Instruct"  # Change this to the desired default model
export LLAMA_STACK_PORT=8321
export SAMBANOVA_API_KEY="12345678abcdef87654321fe"  # Replace with your SambaNova Cloud API key
```

Create the Llama Stack Directory

```bash
mkdir -p ~/.llama
```

### Deploy with venv (Recommended)

Run the following command to start the distribution using a virtual environment:

``` bash
llama stack run --image-type venv ~/.llama/distributions/sambanova/sambanova-run.yaml \
    --port $LLAMA_STACK_PORT \
    --env INFERENCE_MODEL=$INFERENCE_MODEL \
    --env SAMBANOVA_API_KEY=$SAMBANOVA_API_KEY
```

### Deploy with Docker

To deploy using Docker, run:

```bash
docker run -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  distribution-sambanova:0.1.6 \  # Change this to match the tag of your built image
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env SAMBANOVA_API_KEY=$SAMBANOVA_API_KEY
```

### Deploy with Conda

For Conda-based deployment, use:

```bash
llama stack run --image-type conda ~/.llama/distributions/sambanova/sambanova-run.yaml \
    --port $LLAMA_STACK_PORT \
    --env INFERENCE_MODEL=$INFERENCE_MODEL \
    --env SAMBANOVA_API_KEY=$SAMBANOVA_API_KEY
```

## Usage:

We provide a series of [notebooks](./notebooks/) that demonstrate how to use the SambaNova Llama Stack distribution:

1. [Quickstart](./notebooks/quickstart.ipynb) – A simple client usage covering: Listing available models, Using the SambaNova inference adaptor to interact with cloud-based LLM chat models, Using the safety adaptor.

2. [Image Chat](./notebooks/image_chat.ipynb) – Demonstrates how to use the inference adaptor to interact with SambaNova cloud-based vision models.

3. Loop Chat – Shows how to implement a loop conversation for chat using the SambaNova inference adaptor.

4. Tool Calling – Demonstrates tool invocation using the SambaNova inference adaptor and tool runtime adaptors.

5. RAG Agent – Provides an example of a simple Retrieval-Augmented Generation (RAG) agent using The SambaNova inference adaptor, Vector I/O adaptors, Inline embeddings,The agent adaptor.

6. ReAct Agent – Implements a simple Reasoning + Acting (ReAct) agent with tool invocation capabilities.
