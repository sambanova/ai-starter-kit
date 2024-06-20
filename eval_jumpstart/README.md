# RAG Evaluation Kit

A tool for evaluating the performance of LLM APIs using the RAG Evaluation methodology.

## Installation
  
  ```bash
  git clone https://github.com/sambanova/ai-starter-kit
  cd eval_jumpstart
  pip install -r requirements.txt
  ```

## Basic Usage
We implement performance tests for evaluating LLMs using various metrics.

### Config File

Create a YAML configuration file to specify the evaluation settings.

Example config.yaml

```yaml
llms:
  - name: samba_llm
    sambastudio_base_url: "https://api-stage.sambanova.net"
    sambastudio_project_id: "your_project_id"
    sambastudio_endpoint_id: "your_endpoint_id"
    sambastudio_api_key: "your_api_key"

eval_llms:
  - name: eval_llm
    sambastudio_base_url: "https://api-stage.sambanova.net"
    sambastudio_project_id: "your_project_id"
    sambastudio_endpoint_id: "your_endpoint_id"
    sambastudio_api_key: "your_api_key"

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

eval_dataset:
  path: "data/eval_dataset.csv"
  question_col: "question"
  answer_col: "answer"
  ground_truth_col: "ground_truth"
  context_col: "context"

evaluation:
  num_samples: 100
  log_wandb: true
  project_name: "rag-eval"
  eval_name: "rag_eval_test"
```

## Evaluation Use Cases

```python
## Use Case 1: Evaluation without Generation

from rag_eval import RAGEvaluator, RAGEvalConfig
from langchain_community.llms.sambanova import SambaStudio
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import pandas as pd

config_path = "path/to/config.yaml"
config = RAGEvalConfig(config_path)
eval_llms = [
    (llm_name, SambaStudio(**llm_config))
    for conf in config.eval_llm_configs 
    for llm_name, llm_config in [config.get_llm_config(conf)]
]
eval_embeddings = HuggingFaceInstructEmbeddings(model_name=config.embedding_model_name)
evaluator = RAGEvaluator(eval_llms=eval_llms, eval_embeddings=eval_embeddings, config_yaml_path=config_path)
eval_df = pd.read_csv("path/to/eval.csv")
results_no_gen = evaluator.evaluate(eval_df)
print(results_no_gen)
```

## Use Case 3: Evaluation with Generation

```python
from rag_eval import RAGEvaluator, RAGEvalConfig, load_pipeline
from langchain_community.llms.sambanova import SambaStudio

config_path = "path/to/config.yaml"
config = RAGEvalConfig(config_path)
eval_llms = [
    (llm_name, SambaStudio(**llm_config))
    for conf in config.eval_llm_configs 
    for llm_name, llm_config in [config.get_llm_config(conf)]
]
eval_embeddings = HuggingFaceInstructEmbeddings(model_name=config.embedding_model_name)
evaluator = RAGEvaluator(eval_llms=eval_llms, eval_embeddings=eval_embeddings, config_yaml_path=config_path)
eval_df = pd.read_csv("path/to/eval.csv")
pipelines = [
    load_pipeline((llm_name, SambaStudio(**llm_config)), config)
    for llm_name, llm_config in [
        config.get_llm_config(conf) for conf in config.llm_configs
    ]
]
results_with_gen = evaluator.evaluate(eval_df, pipelines)
print(results_with_gen)
```

More use cases are available in the [notebooks](./notebooks)

## Running the Evaluation Using CLI

Command without Generation

```bash
python evaluate.py --config path/to/config.yaml --eval_csv path/to/eval.csv
```

Command with Generation

```bash
python evaluate.py --config path/to/config.yaml --eval_csv path/to/eval.csv --generation

```

## Metrics

The evaluation kit uses various metrics to evaluate the performance of LLMs:
-
- Answer Relevancy
- Answer Correctness
- Answer Similarity
- Context Precision
- Context Recall
- Faithfulness
- Context Relevancy

These are from RAGAS please see the RAGAS website for details on the underlying methodlogy 
https://docs.ragas.io/en/stable/concepts/metrics/index.html

## Logging

Results can be logged to Weights & Biases (wandb) by setting the log_wandb parameter in the configuration file.