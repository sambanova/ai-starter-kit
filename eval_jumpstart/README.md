# Eval Kit

A tool for evaluating the performance of LLM APIs using the RAG Evaluation methodology.

## Installation
  
  ```bash
  git clone https://github.com/sambanova/ai-starter-kit
  cd eval_jumpstart
  python -m venv eval_jumpstart_env
  source eval_jumpstart_env/bin/activate
  pip install -r requirements.txt
  ```

## Basic Usage
We implement performance tests for evaluating LLMs.

### Config File

Create a YAML configuration file to specify the evaluation settings.

Example config.yaml

```yaml
eval_dataset:
  name: general_knowledge_data
  path: data/eval_data.csv

llms:
  - name: sncloud-llama3.1-405
    model_type: "sncloud"
    model_name: "Meta-Llama-3.1-405B-Instruct"
    max_tokens: 1024
    temperature: 0.0
  - name: sambastudio-llama2-70
    model_type: "sambastudio"
    model_name: "llama-2-70b-chat-hf"
    max_tokens: 1024
    temperature: 0.0
  - name: sncloud-llama3.1-70
    model_type: "sncloud"
    model_name: "Meta-Llama-3.1-70B-Instruct"
    max_tokens: 1024
    temperature: 0.0
  - name: sncloud-llama3.2-3
    model_type: "sncloud"
    model_name: "Meta-Llama-3.2-3B-Instruct"
    max_tokens: 1024
    temperature: 0.0

eval_llm:
  model_type: "sncloud"
  model_name: "Meta-Llama-3.1-405B-Instruct"
  max_tokens: 1024
  temperature: 0.0
```

## Evaluation Use Cases

```python
from dotenv import load_dotenv
import sys
import os

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

load_dotenv(os.path.join(repo_dir, '.env'), override=True)

from utils.eval.evaluator import BaseWeaveEvaluator
import weave
import asyncio
import time

weave.init('your-project-name')

evaluator = BaseWeaveEvaluator()

start_time = time.time()

asyncio.run(evaluator.evaluate(name='dataset_name', filepath='dataset/path', use_concurrency=True))

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.2f} seconds")
```

More use cases are available in the [notebooks](./notebooks)

## Metrics

The evaluation kit uses various metrics to evaluate the performance of LLMs:

- Score
- Reason

## Logging

Results can be logged to Weights & Biases (wandb) by setting your WANDB_API_KEY in the `.env` file.

## Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory.