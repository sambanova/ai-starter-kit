<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/light-logo.png" height="100">
  <img alt="SambaNova logo" src="../images/dark-logo.png" height="100">
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

SambaCloud currently supports the following models for all developer and enterprise accounts: [View supported models](https://docs.sambanova.ai/docs/en/models/sambacloud-models).

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

model = "Meta-Llama-3.3-70B-Instruct"
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
from langchain_sambanova import ChatSambaNova

api_key = os.environ.get("SAMBANOVA_API_KEY")

llm = ChatSambaNova(
    api_key=api_key,
    streaming=True,
    model="Meta-Llama-3.3-70B-Instruct",
)

response = llm.invoke('What is the capital of France?')
print(response.content)
```

This code snippet demonstrates how to set up a Langchain `ChatSambaNova` instance with SambaNova's APIs, specifying the API key, streaming option, and model. You can then use the `llm` object to generate completions by passing in prompts.

## Get Help

- Check out the [SambaNova Developer Guide](https://docs.sambanova.ai/cloud/docs/get-started/overview) for additional help
- Find answers and post questions in the [SambaNova Community](https://community.sambanova.ai/latest)
- Let us know your most wanted features and challenges via the channels above
- More inference models, longer context lengths, and embeddings models are coming soon!

  
## Contribute

Building something cool? We welcome contributions to the SambaNova Quickstarts repository! If you have ideas for new quickstart projects or improvements to existing ones, please [open an issue](https://github.com/sambanova/ai-starter-kit/issues/new) or submit a [pull request](https://github.com/sambanova/ai-starter-kit/pulls) and we'll respond right away.
