<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>


# SambaNova API QuickStart Guide

This guide walks through setting up an API key, performing a few sample queries with and without LangChain, and shares example applications to bootstrap application development for common AI use cases with open-source Python code on the SambaNova GitHub page. Let's get started!

## Setting up SambaNova API Key

1. Create an account on the [SambaNova Developer Portal](https://cloud.sambanova.ai/) to get an API key.
2. Once logged in, navigate to the API section and generate a new key. 
3. Set your API key as an environment variable:
   ```shell
   export SAMBANOVA_API_KEY="your-api-key-here"
   ```

## Supported Models

Access Meta, Deepseek and Qwen families of models at **full precision** via the SambaNova Cloud API!

**Model details for DeepSeek family**:
1. DeepSeek R1 671B:
   - Model ID: `DeepSeek-R1`
   - Context length: 4k, 8k
2. DeepSeek R1 Distill Llama 70B:
   - Model ID: `DeepSeek-R1-Distill-Llama-70B`
   - Context length: 4k, 8k, 16k, 32k
3. DeepSeek V3:
   - Model ID: `DeepSeek-V3-0324`
   - Context length: 4k, 8k

**Model details for Llama 4 family**:
1. Llama 4 Scout:
   - Model ID: `Llama-4-Scout-17B-16E-Instruct`
   - Context length: 4k, 8k
2. Llama 4 Maverick:
   - Model ID: `Llama-4-Maverick-17B-128E-Instruct`
   - Context length: 4k, 8k

**Model details for Llama 3.3 family**:
1. Llama 3.3 70B:
   - Model ID: `Meta-Llama-3.3-70B-Instruct`
   - Context length: 4k, 8k, 16k, 32k, 64k, 128k

**Model details for Llama 3.2 family**:
1. Llama 3.2 1B:
   - Model ID: `Meta-Llama-3.2-1B-Instruct`
   - Context length: 16k
2. Llama 3.2 3B:
   - Model ID: `Meta-Llama-3.2-3B-Instruct`
   - Context length: 4k

**Model details for Llama 3.1 family**:
1. Llama 3.1 8B:
   - Model ID: `Meta-Llama-3.1-8B-Instruct`
   - Context length: 4k, 8k, 16k, 32k, 64k, 128k
3. Llama 3.1 405B:
   - Model ID: `Meta-Llama-3.1-405B-Instruct`
   - Context length: 4k, 8k, 16k
   
**Model details for Qwen  family**
1. QwQ 32B:
    - Model ID: `QwQ-32B`
    - Context length: 8k, 16k
2. Qwen2 Audio 7B:
    - Model ID: `Qwen2-Audio-7B-Instruct`
    - Context length: 4k

**Model details for Llama Guard family**:
1. Llama Guard 3 8B:
   - Model ID: `Meta-Llama-Guard-3-8B`
   - Context length: 8k

**Model details for E5 Embeddings family**:
1. E5 Mistral 7B:
   - Model ID: `E5-Mistral-7B-Instruct`
   - Context length: 8k

> You can also get the full list of models runing in your teminal:

``` bash
curl https://api.sambanova.ai/v1/models
```

## Query the API

Install the OpenAI Python library:
```shell  
pip install openai
```

Perform a chat completion:

```python
from openai import OpenAI
import os

api_key = os.environ.get("SAMBANOVA_API_KEY")

client = OpenAI(
    base_url="https://api.sambanova.ai/v1/",
    api_key=api_key,  
)

model = "Meta-Llama-3.1-405B-Instruct"
prompt = "Tell me a joke about artificial intelligence."

completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user", 
            "content": prompt,
        }
    ],
    stream=True,
)

response = ""
for chunk in completion:
    response += chunk.choices[0].delta.content or ""

print(response)
```

## Using SambaNova APIs with Langchain

Install `langchain-sambanova`:
```shell  
pip install -U langchain-sambanova
```

Here's an example of using SambaNova's APIs with the Langchain library:

```python
import os
from langchain_sambanova import ChatSambaNovaCloud

api_key = os.environ.get("SAMBANOVA_API_KEY")

llm = ChatSambaNovaCloud(
    api_key=api_key,
    streaming=True,
    model="Meta-Llama-3.3-70B-Instruct",
)

response = llm.invoke('What is the capital of France?')
print(response.content)
```

This code snippet demonstrates how to set up a Langchain `ChatSambaNovaCloud` instance with SambaNova's APIs, specifying the API key, streaming option, and model. You can then use the `llm` object to generate completions by passing in prompts.

## Starter Applications

[SambaNova AI Starter Kits](../README.md) help you build fast, bootstrapping application development for common AI use cases with open-source Python code on a SambaNova GitHub repository. They let you see how the code works and customize it to your needs, so you can prove the business value of AI. Here are some of the most popular kits:

| Application                        | Description                                                                                                                                                                                                                                                  | Demo                                                           | Gradio                                                                                                                  | Source Code                                                                                         |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Multi Modal Retriever              | Chart, Image, and Figure Understanding – Unlock insights from complex PDFs and images with advanced retrieval and answer generation that combines both visual and textual data.                                                                              | [Live Demo](https://aiskmmr.cloud.snova.ai/)                   | None                                                                                                                         | [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/multimodal_knowledge_retriever)  |
| Financial Assistant                | Enterprise-grade accuracy – Generate sophisticated, complex, and accurate responses by employing multiple agents in a chain to focus/decompose queries, generate multi-step answers, summarize them, and double-check accuracy.                                 | [Live Demo](http://aiskfinancialassistant.cloud.snova.ai/)     | None                                                                                                                         | [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/financial_assistant)             |
| SambaAI Workspaces Integration     | Seamless LLM Integration in Google Workspace – Enhance your productivity by integrating powerful language models directly into Google Docs and Sheets via App Scripts.                                                                                       | None                                                           | None                                                                                                                         | [Source Code](https://github.com/sambanova/ai-starter-kit/blob/main/google_integration/README.md)    |
| Llama 3.1 Instruct-o1              | Enhanced Reasoning with Llama 3.1 405B – Experience advanced thinking capabilities with Llama 3.1 Instruct-o1, hosted on Hugging Face Spaces.                                                                                                               | None                                                           | [Gradio Demo](https://huggingface.co/spaces/sambanovasystems/Llama3.1-Instruct-O1)                                             | None                                                                                                |
| Function Calling                   | Tools calling implementation and generic function calling module – Enhance your AI applications with powerful function calling capabilities.                                                                                                                 | [Live Demo](https://aiskfunctioncalling.cloud.snova.ai/)       | None                                                                                                                         | [Source Code](https://github.com/sambanova/ai-starter-kit/blob/main/function_calling/README.md)      |
| Enterprise Knowledge Retrieval     | Document Q&A on PDF, TXT, DOC, and more – Bootstrap your document Q&A application with this sample implementation of a Retrieval Augmented Generation semantic search workflow using the SambaNova platform, built with Python and a Streamlit UI.           | [Live Demo](https://aiskekr.cloud.snova.ai/)                   | [Gradio Demo](https://huggingface.co/spaces/sambanovasystems/enterprise_knowledge_retriever)                                   | [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/enterprise_knowledge_retriever)  |
| Search Assistant                   | Include web search results in responses – Expand your application's knowledge with this implementation of the semantic search workflow and prompt construction strategies, with configurable integrations with multiple SERP APIs.                            | [Live Demo](https://aisksearchassistant.cloud.snova.ai/)       | None                                                                                                                         | [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/search_assistant)                |
| Benchmarking                       | Compare model performance – Quickly determine which models meet your speed and quality needs by comparing model outputs, Time to First Token, End-to-End Latency, Throughput, Latency, and more with configuration options in a chat interface.                | [Live Demo](https://aiskbenchmarking.cloud.snova.ai/)          | None                                                                                                                         | [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/benchmarking)                    |


## Get Help

- Check out the [SambaNova support documentation](https://sambanova.ai/developer-resources) and [SambaNova Cloud documentation](https://docs.sambanova.ai/cloud/docs/get-started/overview) for additional help
- Find answers and post questions in the [SambaNova Community](https://community.sambanova.ai/latest)
- Let us know your most wanted features and challenges via the channels above
- More inference models, longer context lengths, and embeddings models are coming soon!

  
## Contribute

Building something cool? We welcome contributions to the SambaNova Quickstarts repository! If you have ideas for new quickstart projects or improvements to existing ones, please [open an issue](https://github.com/sambanova/ai-starter-kit/issues/new) or submit a [pull request](https://github.com/sambanova/ai-starter-kit/pulls) and we'll respond right away.
