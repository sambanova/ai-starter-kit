# Weave @SambaNova

This document demonstrates how to use [W&B Weave](https://weave-docs.wandb.ai/){:target="_blank"} with [SambaNova](https://sambanova.ai/){:target="_blank"} as your fastest LLM provider of choice for open source models.

`Weights & Biases (W&B) Weave` is a framework for tracking, experimenting with, evaluating, deploying, and improving LLM-based applications. Designed for flexibility and scalability, Weave supports every stage of your LLM application development workflow:

- Tracing & Monitoring: Track LLM calls and application logic to debug and analyze production systems.
- Systematic Iteration: Refine and iterate on prompts, datasets, and models.
- Experimentation: Experiment with different models and prompts in the LLM Playground.
- Evaluation: Use custom or pre-built scorers alongside our comparison tools to systematically assess and enhance application performance.
- Guardrails: Protect your application with pre- and post-safeguards for content moderation, prompt safety, and more.

In order to use `Weave` @`SambaNova`, you need to set the environment variable `SAMBANOVA_API_KEY`: your API 
 for accessing the SambaNova Cloud. You can create your API 
 [here](https://cloud.sambanova.ai/apis){:target="_blank"}.

1. To get started, simply call `weave.init()` at the beginning of your script, with the project name as attribute.

2. `Weave` ops make results reproducible by automatically versioning code as you experiment.
Simply create a function decorated with `@weave.op()` that invokes each completion function and `Weave` will track the inputs and outputs of the function for you. 

3. By using the `weave.Model` class, you can capture and organize the experimental details of your app like your system prompt or the model that you are using. This helps organize and compare different iterations of your app.

## Pre-requisites: 
1. Create a [SambaNova Cloud](https://cloud.sambanova.ai/){:target="_blank"} account and get an API key.

2. Install the packages recommended in the `requirements.txt` file.
```bash
cd integrations/weave
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or install them separately:
```bash
pip install dotenv
pip install langchain-sambanova  # If you want to use our SambaNova LangChain Chat object
pip install litellm  # If you want to use the liteLLM chat object
pip install openai  # If you want to use the OpenAI SDK
pip install weave
```

## Notebooks
The `Weave@SambaNova.ipynb` notebook show how to use `Weave` with `Sambanova` using the three connectors:
- SambaNova LangChain chat object.
- liteLLM.
- OpenAI SDK.

## Setup
```python
import os
from typing import Any, Optional

import weave
from dotenv import load_dotenv

load_dotenv()

# If you have SAMBANOVA_API_KEY in your .env file
SAMBANOVA_API_KEY = os.getenv('SAMBANOVA_API_KEY')

# Choose your model
model = 'Meta-Llama-3.3-70B-Instruct'
```

## Via LangChain

`Weave` is designed to make tracking and logging all calls made through the `LangChain` Python library effortless, after `weave.init()` is called.

You can access all the features of the `LangChain` + `Weave` integration, by using our `LangChain` chat object, `langchain_sambanova.ChatSambaNovaCloud`.

For more details on all the `Weave` features supported by `LangChain`, please refer to [Weave @LangChain](https://weave-docs.wandb.ai/guides/integrations/langchain/).

```python
from langchain_core.prompts import PromptTemplate
from langchain_sambanova import ChatSambaNovaCloud

# Initialize Weave project
weave.init('weave_integration_sambanova_langchain')

# The LangChain SambaNova Chat object
llm = ChatSambaNovaCloud(
    model=model,
    temperature=0.7,
    top_p=0.95,
)

# The prompt template
prompt = PromptTemplate.from_template('1 + {number} = ')

# The LLM chain
llm_chain = prompt | llm
```

### Simple call
```python
# Invoke the LLM on the prompt
output = llm_chain.invoke({'number': 2})

print(output.content)
```

### Tracking Call Metadata
To track metadata from your `LangChain` calls, you can use the `weave.attributes` context manager. This context manager allows you to set custom metadata for a specific block of code, such as a chain or a single request.

```python
# The LLM chain with Weave attributes
with weave.attributes({'number_to_increment': 'value'}):
    output = llm_chain.invoke({'number': 2})

print(output.content)
```

## Via LiteLLM

`Weave` automatically tracks and logs LLM calls made via LiteLLM, after `weave.init()` is called.

You can access all the features of the `Weave` + `LiteLLM` integration, by specifying the `SambaNova` model name in the `LiteLLM` constructor, as explained in [LiteLLM @SambaNova](https://docs.litellm.ai/docs/providers/sambanova).

For more details on all the `Weave` features supported by `LiteLLM`, please refer to [Weave @LiteLLM](https://weave-docs.wandb.ai/guides/integrations/litellm).

```python
import litellm

# Initialize Weave project
weave.init('weave_integration_sambanova_litellm')
model_litellm = 'sambanova/' + model
```

### Simple call
```python
# Tranlsate
response = litellm.completion(
    model=model_litellm,
    messages=[{'role': 'user', 'content': "Translate 'Hello, how are you?' to French."}],
    max_tokens=1024,
)

print(response.choices[0].message.content)
```

### @weave.op
```python
# Define a translation function
@weave.op()
def translate_litellm(model: str, text: str, target_language: str) -> Any:
    response = litellm.completion(
        model=model, messages=[{'role': 'user', 'content': f"Translate '{text}' to {target_language}"}], max_tokens=1024
    )
    return response.choices[0].message.content

# Translate
translate_litellm(model_litellm, 'Hello, how are you?', 'French')
```

### weave.Model
```python
# Translator model
class TranslatorModel(weave.Model):  # type: ignore
    model: str
    temperature: float

    @weave.op()  # type: ignore
    def predict(self, text: str, target_language: str) -> Any:
        """Translate the given text to target language."""
        
        response = litellm.completion(
            model=self.model,
            messages=[
                {'role': 'system', 'content': f'You are a translator. Translate the given text to {target_language}.'},
                {'role': 'user', 'content': text},
            ],
            max_tokens=1024,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

# Create an instance of the translator weave.Model
translator = TranslatorModel(model=model_litellm, temperature=0.3)

# Translate
english_text = 'Hello, how are you today?'
french_text = translator.predict(english_text, 'French')

print(french_text)
```

## Via the OpenAI SDK

`SambaNova` supports the `OpenAI` SDK compatibility ([docs](https://docs.sambanova.ai/cloud/docs/capabilities/openai-compatibility)), which `Weave` automatically detects and integrates with.

To use the SambaNova API, simply switch out the `api_key` to your SambaNova API key, `base_url` to https://api.sambanova.ai/v1, and `model` to one of our chat models.

```python
from openai import OpenAI

# Initialize Weave project
weave.init('weave_integration_sambanova_openai_sdk')

# SambaNova URL, e.g. https://api.sambanova.ai/v1
SAMBANOVA_URL = os.getenv('SAMBANOVA_URL')

# Set the sambanova client
sambanova_client = OpenAI(base_url=SAMBANOVA_URL, api_key=SAMBANOVA_API_KEY)
```

### Simple call
```python
# Correct grammar
response = sambanova_client.chat.completions.create(
    model=model,
    messages=[
        {'role': 'system', 'content': 'You are a grammar checker, correct the following user input.'},
        {'role': 'user', 'content': 'That was so easy, it was a piece of pie!'}],
    temperature=0,
)

print(response.choices[0].message.content)
```

### @weave.op
```python
# Define the function for grammar correction 
@weave.op()
def correct_grammar(model: str, system_prompt: str, user_prompt: str) -> Optional[str]:
    """Correct the grammar of a text."""
    
    response = sambanova_client.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

# Correct grammar
response = correct_grammar(model, 'You are a grammar checker, correct the following user input.', 'That was so easy, it was a piece of pie!')

print(response)
```

### weave.Model
```python
# Grammar corrector model
class GrammarCorrectorModel(weave.Model):
    model: str
    system_message: str

    @weave.op()
    def predict(self, user_input: str) -> Optional[str]:
        """Correct the grammar of a text."""

        response = sambanova_client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': user_input}],
            temperature=0,
        )
        return response.choices[0].message.content

# Correct grammar
corrector = GrammarCorrectorModel(
    model=model, system_message='You are a grammar checker, correct the following user input.'
)
result = corrector.predict('That was so easy, it was a piece of pie!')
print(result)
```

