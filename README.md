<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="./images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

SambaNova AI Starter Kits
====================

# Overview
SambaNova AI Starter Kits are a collection of open-source examples and guides to facilitate the deployment of AI-driven use cases in the enterprise.

To run these examples, you need access to a SambaStudio environment with your models deployed to endpoints. Most code examples are written in Python, though the concepts can be applied in any language.

If you have any issues with the examples or to provide feedback, [create an issue](https://github.com/sambanova/ai-starter-kit/issues/new/choose) in GitHub.

# Available AI Starter Kits
|  Starter Kit | About |
| ------------ | ------------ |
| [Data Extraction](data_extraction/README.md) | A series of notebooks that demonstrates various methods for extracting text from documents in different input formats. |
| [EDGAR Q&A](edgar_qna/README.md)  |  An example workflow that uses the SambaNova platform to answer questions about organizations using their 10-K annual reports. Includes a runnable local demo and a Docker container to simplify remote deployment.  |
| [Enterprise Knowledge Retrieval](enterprise_knowledge_retriever/README.md) | A sample implementation of the semantic search workflow using the SambaNova platform to get answers to questions about your documents. Includes a runnable demo.  |
| [Fine tuning embeddings](fine_tuning_embeddings/README.md) |  This kit guides users through fine-tuning embeddings from unstructured data, leveraging Large Language Models (LLMs) and open-source embedding models to enhance NLP task performance.  |
| [Fine Tuning: SQL model](fine_tuning_sql/README.md) | A sample training recipe to build a fine-tuned SQL model over a Llama 7B base. |
| [Post Call Analysis](post_call_analysis/README.md) |  This kit exemplifies a systematic approach to post-call analysis starting with Automatic Speech Recognition (ASR), diarization, large language model analysis, and retrieval augmented generation (RAG) workflows that are built using the SambaNova platform.  |
| [Prompt Engineering](prompt_engineering/README.md)  |  An starting point demo for prompt engineering using Sambanova's API to experiment with diferent use case templates. It also provides useful resources to improve prompt crafting, making it an ideal entry point for those new to this AISK. |
| [Web Crawled Data Retrieval](web_crawled_data_retriever/README.md) | A sample implementation of the semantic search workflow built using the SambaNova platform to get answers to your questions using website crawled information as the source. Includes a runnable demo.  |
| [YoDA: Your Data Your model](yoda/README.md) | A sample training recipe to train a Language Model (LLM) using customer's private data. |

# Get started with SambaNova AI starter kit

## Setting your model

### (Option 1) Use Sambaverse models
Begin by creating a [Sambaverse](sambaverse.sambanova.net) account, then [get your API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) from the username button. Use the available models.

### (Option 2) Deploy your model in SambaStudio
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
> *be sure to set batch size model parameter to 32*

Update API information for the SambaNova embedding endpoint.  These are represented as configurable variables in the environment variables file in the root repo directory **```sn-ai-starter-kit/.env```**. For example, an endpoint with the URL
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
> ```python
>            embeddings = HuggingFaceInstructEmbeddings(
>                model_name="hkunlp/instructor-large",
>                embed_instruction="",
>                query_instruction="Represent this sentence for searching relevant passages:",
>                encode_kwargs={"normalize_embeddings": True},
>            )
> ```
> with **Sambastudio embeddings**:
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

``` python
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

``` python
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

``` python
from utils.sambanova_endpoint import SambaNovaEndpoint

load_dotenv('.env')

embedding = SambaNovaEmbeddingModel()
```

2. Use your embedding model in your langchain pipeline

See [utils/usage.ipynb](./utils/usage.ipynb) for an example. 

***

**Note:** These AI Starter Kit code samples are provided "as-is," and are not production-ready or supported code. Bugfix/support will be on a best-effort basis only. Code may use third-party open-source software. You are responsible for performing due diligence per your organization policies for use in your applications.
