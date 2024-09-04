# Quickstart

Get up and running with the Sambanova API in a few minutes.

## Create an API Key

Please obtain your API key from [here](https://sambanova.ai/fast-api).

## Set up your API Key (recommended)

Configure your API key as an environment variable for enhanced security and ease of use:

```shell
export SAMBANOVA_API_KEY=<your-api-key-here>
```

## Requesting your first chat completion

Install the OpenAI Python library:

```shell
pip install openai
```

Performing a Chat Completion:

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

Now that you have successfully received a chat completion, you can explore other endpoints in the Sambanova API.

## Support Models

For model names, you can pick from the list below:

| Model | Context Length | Output Length | Dtype |
|-------|----------------|---------------|-------|
| Meta-Llama-3.1-8B-Instruct | 8192 | 1000 | BF16 |
| Meta-Llama-3.1-70B-Instruct | 8192 | 1000 | BF16 |
| Meta-Llama-3.1-405B-Instruct | 4096 | 1000 | BF16 |



## Important Notes

* More models will be available soon.
* We are working on increasing the context length.
* For now, our API supports inference only ; embedding functionality is not available yet.


## Next Steps

* Explore the Sambanova API [documentation](https://docs.sambanova.ai/home/latest/index.html) for more advanced features
* Join the Sambanova developer community for support and discussions [Discord](https://discord.gg/bbjYDeRx46) 
* Experiment with different models and parameters to optimize your results
* Consider integrating Sambanova's API into your projects for enhanced AI capabilities
