<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="./images/SambaNova-dark-logo-1.png" height="60">
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

| Model | Context Length | Output Length | Dtype / Precision |
|-------|----------------|---------------|-------|
| Meta-Llama-3.1-8B-Instruct | 4096 | 1000 | BF16 |  
| Meta-Llama-3.1-8B-Instruct-8k | 8192 | 1000 | BF16 |  
| Meta-Llama-3.1-70B-Instruct | 4096 | 1000 | BF16 |
| Meta-Llama-3.1-70B-Instruct-8k | 8192 | 1000 | BF16 |
| Meta-Llama-3.1-405B-Instruct | 4096 | 1000 | BF16 |
| Meta-Llama-3.1-405B-Instruct-8k | 8192 | 1000 | BF16 |

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

Install `langchain-openai`:
```shell  
pip install -U langchain-openai
```

Here's an example of using SambaNova's APIs with the Langchain library:

```python
import os
from langchain_openai import ChatOpenAI

api_key = os.environ.get("SAMBANOVA_API_KEY")

llm = ChatOpenAI(
    base_url="https://api.sambanova.ai/v1/",  
    api_key=api_key,
    streaming=True,
    model="Meta-Llama-3.1-70B-Instruct",
)

response = llm.invoke('What is the capital of France?')
print(response.content)
```

This code snippet demonstrates how to set up a Langchain `ChatOpenAI` instance with SambaNova's APIs, specifying the API key, base URL, streaming option, and model. You can then use the `llm` object to generate completions by passing in prompts.

## Starter Applications

[SambaNova AI Starter Kits](https://community.sambanova.ai/t/ai-starter-kits/160) help you build fast, bootstrapping application development for common AI use cases with open-source Python code on a SambaNova GitHub repository. They let you see how the code works and customize it to your needs, so you can prove the business value of AI. Here are some of the most popular kits:

| Application | Description | Demo | Source Code |
|-------------|-------------|------|-------------|
| Enterprise Knowledge Retrieval Chatbot | Build a retrieval-augmented generation (RAG) chatbot using your enterprise documents | [Live Demo](https://sambanova-ai-starter-kits-ekr.replit.app/) | [Source Code](https://github.com/sambanova/ai-starter-kit/blob/main/enterprise_knowledge_retriever/README.md) |
| Search Assistant | Semantic search using search engine snippets | [Live Demo](https://sambanova-ai-starter-kits-search-assistant.replit.app/) | [Source Code](https://github.com/sambanova/ai-starter-kit/blob/main/search_assistant/README.md) |
| Financial Assistant | Agentic finance assistant built on our API | [Live Demo](https://sambanova-ai-starter-kits-financial-assistant.replit.app/) | [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/financial_insights) |
| Function Calling | Tools calling implementation and generic function calling module | [Live Demo](https://sambanova-ai-starter-kits-function-calling.replit.app/) | [Source Code](https://github.com/sambanova/ai-starter-kit/blob/main/function_calling/README.md) |
| Benchmarking Kit | Evaluates performance of multiple LLM models in SambaStudio | [Live Demo](https://sambanova-ai-starter-kits-benchmarking.replit.app/)  | [Source Code](https://github.com/sambanova/ai-starter-kit/blob/main/benchmarking/README.md) |

## Get Help

- Check out the [SambaNova support documentation](https://sambanova.ai/developer-resources) for additional help
- Find answers and post questions in the [SambaNova Community](https://community.sambanova.ai/latest)
- Let us know your most wanted features and challenges via the channels above
- More inference models, longer context lengths, and embeddings models are coming soon!

  
## Contribute

Building something cool? We welcome contributions to the SambaNova Quickstarts repository! If you have ideas for new quickstart projects or improvements to existing ones, please [open an issue](https://github.com/sambanova/ai-starter-kit/issues/new) or submit a [pull request](https://github.com/sambanova/ai-starter-kit/pulls) and we'll respond right away.
