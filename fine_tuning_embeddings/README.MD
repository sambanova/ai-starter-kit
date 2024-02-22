# Embedding Fine-Tuning Starter Kit

This comprehensive starter kit guides users through fine-tuning embeddings from unstructured data, leveraging Large Language Models (LLMs) and open-source embedding models to enhance NLP task performance. It supports a flexible workflow catering to different stages of the fine-tuning process, from data preparation to evaluation.

## Key Features

- **Automated Query Generation**: Automatically generate synthetic query-answer pairs from unstructured text.
- **Embedding Model Fine-Tuning**: Fine-tune open-source embedding models with a synthetic dataset via the Sentence Transformers library.
- **Performance Evaluation**: Benchmark the fine-tuned embeddings with metrics to quantify improvements.

## Getting Started

### Prerequisites

- Python 3.11+
- Required libraries: Sentence Transformers, Hugging Face Transformers

### Installation

1. Clone the repository.
2. Install dependencies:
   - With poetry: `poetry install --no-root`
   - With pip: `pip install -r requirements.txt`

### Workflows

#### Standard Workflow

1. **Data Preparation**: Place your data in a specified directory. The script defaults to PDFs but supports other file types.
2. **Script Execution**: Run the script with necessary parameters. For example:

   ```bash
   python script_name.py --input_data_directory ./your_data_directory --output_data_directory ./processed_data
   ```

#### Using Pre-generated Data for Fine-Tuning

If you've previously generated synthetic data and wish to proceed directly to fine-tuning:

1. **Specify Dataset Paths**: Use the `--train_dataset_path` and `--val_dataset_path` arguments to provide paths to your pre-generated datasets.
2. **Execute Fine-Tuning**: Run the script with the model and output directory parameters. Example:

   ```bash
   python script_name.py --train_dataset_path ./processed_data/train_dataset.json --val_dataset_path ./processed_data/val_dataset.json --model_id "your_model_id" --model_output_path ./finetuned_model
   ```

#### Evaluating a Pre-Finetuned Model

To evaluate an existing finetuned model without re-running the entire process:

1. **Specify the Model Path**: Use the `--model_output_path` argument to point to your finetuned model directory.
2. **Run Evaluation Only**: Ensure the dataset paths are specified if they are not in the default location. Example:

   ```bash
   python script_name.py --val_dataset_path ./processed_data/val_dataset.json --model_output_path ./finetuned_model --evaluate_only
   ```

### Using SNS Embeddings

The script also supports integrating SNS (SambaNova Systems) embeddings for enhancing the retrieval and understanding capabilities within LangChain workflows. Follow these steps to utilize SNS embeddings:

1. **Configuration**: Ensure you have the `export.env` file set up with your SNS credentials, including `EMBED_BASE_URL`, `EMBED_PROJECT_ID`, `EMBED_ENDPOINT_ID`, and `EMBED_API_KEY`.
2. **Integration Example**: The script demonstrates how to use SNS embeddings for document and query encoding, followed by computing cosine similarity for a given query against a set of documents. Additionally, it integrates these embeddings into a LangChain retrieval workflow, showcasing an end-to-end example of query handling and document retrieval.

To run the SNS embeddings integration:

   ```bash
   python sns_embedding_script.py
   ```

This will execute the embedding process for documents and queries, compute similarities, and demonstrate a LangChain retrieval workflow using these embeddings.

### Additional Arguments

The script supports various arguments to customize the process, including:

- `--file_extension` for specifying file types.
- `--split_ratio` to adjust the train-validation dataset split.
- `--force_retrain` to force retraining even if a finetuned model exists.

Refer to the script's help (`python script_name.py --help`) for a full list of arguments and their descriptions.

## Contributing

We welcome contributions! Feel free to improve the process or add features by submitting issues or pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Sentence Transformers and Hugging Face for their resources and pre-trained models.
- Inspired by practices from [original embedding fine-tuning repository](https://github.com/run-llama/finetune-embedding/tree/main).
