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

### 1. Enterprise Knowledge Retrieval Chatbot

Build a retrieval-augmented generation (RAG) chatbot using your enterprise documents. 

- [Live Demo](https://sambanova-ai-starter-kits-search-assistant.replit.app/)
- [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/enterprise-knowledge-retrieval) 
- To run locally:
  ```shell
  git clone https://github.com/sambanova/ai-starter-kit.git
  cd ai-starter-kit/enterprise-knowledge-retrieval
  pip install -r requirements.txt
  streamlit run app.py
  ```

### 2. Conversational AI Assistant 

Create an AI-powered conversational assistant that understands context and generates human-like responses.

- [Live Demo](https://sambanova-ai-starter-kits-conversational-ai.replit.app/)  
- [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/conversational-ai)
- To run locally: 
  ```shell
  git clone https://github.com/sambanova/ai-starter-kit.git  
  cd ai-starter-kit/conversational-ai
  pip install -r requirements.txt
  streamlit run app.py
  ```

### 3. Text Summarization Tool

Automatically generate concise summaries of long articles or documents.

- [Live Demo](https://sambanova-ai-starter-kits-text-summarizer.replit.app/)
- [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/text-summarization) 
- To run locally:
  ```shell 
  git clone https://github.com/sambanova/ai-starter-kit.git
  cd ai-starter-kit/text-summarization  
  pip install -r requirements.txt
  streamlit run app.py
  ```

### 4. Code Generation Assistant

Streamline your coding with an AI-powered code generation assistant.

- [Live Demo](https://sambanova-ai-starter-kits-code-generation.replit.app/)
- [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/code-generation)
- To run locally:  
  ```shell
  git clone https://github.com/sambanova/ai-starter-kit.git
  cd ai-starter-kit/code-generation
  pip install -r requirements.txt 
  streamlit run app.py
  ```

### 5. Content Creation Toolkit 

Automate content creation tasks like writing articles, generating product descriptions, and more.

- [Live Demo](https://sambanova-ai-starter-kits-content-creation.replit.app/) 
- [Source Code](https://github.com/sambanova/ai-starter-kit/tree/main/content-creation)
- To run locally:
  ```shell
  git clone https://github.com/sambanova/ai-starter-kit.git
  cd ai-starter-kit/content-creation  
  pip install -r requirements.txt
  streamlit run app.py 
  ```

## Contributing

We welcome contributions to the SambaNova Quickstarts repository! If you have ideas for new quickstart projects or improvements to existing ones, please [open an issue](https://github.com/sambanova/ai-starter-kit/issues/new) or submit a [pull request](https://github.com/sambanova/ai-starter-kit/pulls).


## Next Steps

### Community and Support

- Join our [SambaNova Discord community](https://discord.gg/54bNAqRw) for discussions and support  
- Check out the [SambaNova support documentation](https://sambanova.ai/developer-resources) for additional help


### FAQs

* More models will be available soon.
* We are working on increasing the context length.
* For now, our API supports inference only ; embedding functionality is not available yet.
