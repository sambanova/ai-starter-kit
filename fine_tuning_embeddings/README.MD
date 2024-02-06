# Embedding Fine-Tuning Starter Kit
# Embedding Fine-Tuning Starter Kit

This starter kit is designed to guide users through the process of fine-tuning embeddings from unstructured data. Leveraging the power of Large Language Models (LLMs) and open-source embedding models, we aim to enhance the performance of embeddings for various NLP tasks. This kit is structured into three main parts:

## Part 1: Generating Synthetic Query-Answer Pairs

The first part focuses on generating synthetic query-answer pairs from unstructured data using an LLM. This process involves:

- Loading and processing a corpus of documents.
- Utilizing a pre-trained LLM to generate relevant queries based on the context of each document.
- Saving the generated query-answer pairs for use in fine-tuning.

### Key Features:

- Automated query generation from unstructured text.
- Customizable query generation based on document context.

## Part 2: Fine-Tuning an Open-Source Embedding Model

In the second part, we fine-tune an open-source embedding model using the synthetic query-answer pairs generated in Part 1. This involves:

- Loading the Sentence Transformers library for embedding fine-tuning.
- Preparing the synthetic dataset for training.
- Fine-tuning the embedding model on the synthetic dataset.
- Saving the fine-tuned model for downstream tasks.

### Key Features:

- Easy-to-use fine-tuning workflow with Sentence Transformers.
- Enhanced embedding quality for NLP tasks.

## Part 3: Evaluating Embedding Performance

The final part involves evaluating the performance of the fine-tuned embeddings. We compare:

- Closed-source embeddings.
- Open-source embeddings without fine-tuning.
- Open-source embeddings with fine-tuning (as performed in Part 2).

### Evaluation Metrics:

- Similarity scores.
- Task-specific performance metrics (e.g., F1 score for question answering).

### Key Features:

- Comprehensive evaluation framework.
- Insights into the benefits of embedding fine-tuning.

## Getting Started

To get started with this starter kit, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Follow the Jupyter notebooks provided for each part of the process:
   - `Part1_Generate_Synthetic_Data.ipynb`
   - `Part2_Fine_Tuning_Embeddings.ipynb`
   - `Part3_Evaluation.ipynb`

## Requirements

- Python 3.11+
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- Additional requirements are listed in `requirements.txt`.
- Simpest Way to to get started is:
    - Create a venv with Python 3.11
    - install using poetry (recommended) via "poetry install --no-root" from this root dir
    - Pip can also be used to install req's with the supplied requirements.txt file  

## Contributing

Contributions to this starter kit are welcome! Please feel free to submit issues or pull requests with improvements or new features.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Sentence Transformers Library for providing an easy-to-use interface for embedding models.
- Hugging Face for their extensive repository of pre-trained models and LLMs.
- [Original GitRepo - on embedding fine-tuning](https://github.com/run-llama/finetune-embedding/tree/main)

