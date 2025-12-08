
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>

Eval Kit
======================

A tool for evaluating the performance of LLM APIs using the RAG Evaluation methodology.

## Installation
  
  ```bash
  git clone https://github.com/sambanova/ai-starter-kit
  cd eval_jumpstart
  python -m venv eval_jumpstart_env
  source eval_jumpstart_env/bin/activate
  pip intall 
  uv pip install -r requirements.txt
  ```

## Basic Usage
We implement performance tests for evaluating LLMs and RAG systems.

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

  - name: sncloud-llama3.2-1
    model_type: "sncloud"
    model_name: "Meta-Llama-3.2-1B-Instruct"
    max_tokens: 1024
    temperature: 0.0

  - name: sncloud-llama3.3-70
    model_type: "sncloud"
    model_name: "Meta-Llama-3.3-70B-Instruct"
    max_tokens: 1024
    temperature: 0.0

  - name: sncloud-llama3.2-3
    model_type: "sncloud"
    model_name: "Meta-Llama-3.2-3B-Instruct"
    max_tokens: 1024
    temperature: 0.0

rag:
  vectordb:
    db_type: "chroma"
    collection_name: "demo"
  
  embeddings:
    type: "cpu"
    batch_size: 1
    bundle: True
    select_expert: "e5-mistral-7b-instruct"

  llm:
    name: sncloud-llama3.1-405-chroma-rag
    type: "sncloud"
    model: "Meta-Llama-3.1-405B-Instruct"
    max_tokens: 1024
    temperature: 0.0

eval_llm:
  model_type: "sncloud"
  model_name: "Meta-Llama-3.1-405B-Instruct"
  max_tokens: 1024
  temperature: 0.0
```

## Evaluation Use Cases for LLMs

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

import asyncio
import time
from utils.eval.evaluator import BaseWeaveEvaluator
import weave


weave.init('your-project-name')

evaluator = BaseWeaveEvaluator()

start_time = time.time()

asyncio.run(evaluator.evaluate(name='dataset_name', filepath='dataset/path', use_concurrency=True))

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.2f} seconds")
```

More use cases are available in the [notebooks](./notebooks/eval_llm.ipynb)

# Deploy the starter kit GUI

- Before running the app, make sure to log in to [W&B](https://wandb.ai/authorize) and set your `WANDB_API_KEY` in the `.env` file to view the final results. Additionally,
when evaluating LLMs, the CSV file should follow this structure:

   ```csv
   system_message,query,expected_answer
   "Some prompt message","some quesstion?","The ground truth answer"
   ```
- **Note:** The `system_message` column is optional.

- For evaluating RAG chains, upload a PDF file, and the CSV file should contain questions related to the PDF. The structure should be as follows:

  ```csv
   query,expected_answer
   "some quesstion?","The ground truth answer"
   ```

- Now you can run the starter kit using the streamlit app with the following command:

   ```bash
   streamlit run streamlit/app.py --browser.gatherUsageStats false 
   ```

## Metrics

The evaluation kit uses various metrics to evaluate the performance of LLMs:

- Score
- Reason

## Logging

Results can be logged to Weights & Biases (wandb) by setting your WANDB_API_KEY in the `.env` file.

## Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory.