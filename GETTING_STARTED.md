[![SambaNova logo](./images/SambaNova-dark-logo-1.png)](https://sambanova.ai/)

# QuickStart Guide: Getting Started with SambaNova APIs

Welcome to the **SambaNova QuickStart Guide**. This guide will help you set up your API key, run a "Hello World" example, explore a few starter applications, and integrate with Langchain to jumpstart your journey with SambaNova's APIs. Let's get started!

## Setting up Your SambaNova API Key

To get started, follow these steps:

1. Create an account on the [SambaNova Developer Portal](https://sambanova.ai/fast-api) to get your API key.
2. Once logged in, navigate to the API section and generate a new key. 
3. Set your API key as an environment variable:
   ```shell
   export SAMBANOVA_API_KEY=<your-api-key-here>
   ```

## Support Models

For model names, you can pick from the list below:

| Model | Context Length | Output Length | Dtype |
|-------|----------------|---------------|-------|
| Meta-Llama-3.1-8B-Instruct | 8192 | 1000 | BF16 |  
| Meta-Llama-3.1-70B-Instruct | 8192 | 1000 | BF16 |
| Meta-Llama-3.1-405B-Instruct | 4096 | 1000 | BF16 |

## Hello World: Your First API Request

Install the OpenAI Python library:
```shell  
pip install openai
```

Perform a chat completion:

```python
from openai import OpenAI
api_key = os.environ.get("SAMBANOVA_API_KEY")

client = OpenAI(
    base_url="https://fast-api.snova.ai/v1/",
    api_key=api_key,  
)

model = "llama3-405b"
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

Here's an example of using SambaNova's APIs with the Langchain library:

```python
import os
from langchain_openai import ChatOpenAI

api_key = os.environ.get("SAMBANOVA_API_KEY")

llm = ChatOpenAI(
    base_url="https://fast-api.snova.ai/v1/",  
    api_key=api_key,
    streaming=True,
    model="llama3-70b",
)

llm('What is the capital of France?')
```

This code snippet demonstrates how to set up a Langchain `ChatOpenAI` instance with SambaNova's APIs, specifying the API key, base URL, streaming option, and model. You can then use the `llm` object to generate completions by passing in prompts.

## Starter Applications

| Application | Description | Demo | Source Code |
|-------------|-------------|------|-------------|
| Enterprise Knowledge Retrieval Chatbot | Build a retrieval-augmented generation (RAG) chatbot using your enterprise documents | [Live Demo](https://sambanova-ai-starter-kits-ekr.replit.app/) | [Source Code](https://github.com/sambanova/ai-starter-kit/blob/main/enterprise_knowledge_retriever/README.md) |
| Conversational Search Assistant | Semantic search using search engine snippets | [Live Demo](https://sambanova-ai-starter-kits-search-assistant.replit.app/) | [Source Code](https://github.com/sambanova/ai-starter-kit/blob/main/search_assistant/README.md) |
| Financial Assistant | Agentic finance assistant built on our API | - | [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/financial_insights) |
| Function Calling | Tools calling implementation and generic function calling module | - | [Source Code](https://github.com/sambanova/ai-starter-kit/blob/main/function_calling/README.md) |
| Benchmarking Kit | Evaluates performance of multiple LLM models in SambaStudio | [Live Demo](https://sambanova-ai-starter-kits-benchmarking.replit.app/)  | [Source Code](https://github.com/sambanova/ai-starter-kit/blob/main/benchmarking/README.md) |

## Contributing

We welcome contributions to the SambaNova Quickstarts repository! If you have ideas for new quickstart projects or improvements to existing ones, please [open an issue](https://github.com/sambanova/ai-starter-kit/issues/new) or submit a [pull request](https://github.com/sambanova/ai-starter-kit/pulls).

## Next Steps

### Community and Support

- Join our [SambaNova Discord community](https://discord.gg/54bNAqRw) for discussions and support  
- Check out the [SambaNova support documentation](https://sambanova.ai/developer-resources) for additional help

### FAQs

* More models will be available soon.
* We are working on increasing the context length.
* For now, our API supports inference only; embedding functionality is not available yet.