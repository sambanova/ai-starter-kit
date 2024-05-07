<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Fine-Tuning Embedding Starter Kit
====================


<!-- TOC -->

- [Key features](#key-features)
- [Before you begin](#before-you-begin)
   - [Clone the repo and install dependencies](#clone-the-repo-and-install-dependencies)
   - [Ensure your environment meets prerequisites](#ensure-your-environment-meets-prerequisites)
   - [Set up the account and config file](#set-up-the-account-and-config-file)
      - [Setup for SambaStudio](#setup-for-sambastudio)
      - [Setup for Sambaverse](#setup-for-sambaverse)
- [Using the model](#using-the-model)
   - [Prepare data and run the script](#prepare-data-and-run-the-script)
      - [Additional arguments](#additional-arguments)
   - [Perform fine tuning using pregenerated data](#perform-finetuning-using-pre-generated-data)
   - [Evaluating a finetuned model](#evaluating-a-finetuned-model)
   - [Using SambaNova systems (SNS) embeddings](#using-sambanova-systems-sns-embeddings)
- [Using pre-generated data for finetuning](#using-pre-generated-data-for-finetuning)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgments)

<!-- /TOC -->

This comprehensive starter kit guides users through fine-tuning embeddings from unstructured data, leveraging Large Language Models (LLMs) and open-source embedding models to enhance NLP task performance. It supports a flexible workflow catering to different stages of the fine-tuning process, from data preparation to evaluation.


# Key features

- **Automated Query Generation**: Automatically generate synthetic query-answer pairs from unstructured text.
- **Embedding Model Fine-Tuning**: Fine-tune open-source embedding models with a synthetic dataset via the Sentence Transformers library.
- **Performance Evaluation**: Benchmark the fine-tuned embeddings with metrics to quantify improvements.


 
Some things to note:

 * This kit is fine-tuning models from Hugging Face using the name of the model and the model_id referred to later.
 * We hope to have embedding model fine-tuning capability in SambaStudio very soon which will allow you to fine-tune embeddings models on SambaNova RDUs.
 * In this kit the SambaNova LLM is used in the dataset creation process, namely in creating questions and answers from the chunks in order to finetune the model.
 * This kit allows the use of either SambaStudio or SambaVerse: 

# Before you begin

You can use this model with Sambaverse or SambaStudio, but you have to do some setup first. 

## Clone the repo and install dependencies

1. Clone the ai-starter-kit repo.
```
  git clone https://github.com/sambanova/ai-starter-kit.git
```
2. Install the dependencies:
   - With poetry: `poetry install --no-root`
   - With pip: `pip install -r requirements.txt`
Clone the start kit repo.

## Ensure your environment meets prerequisites

- Python 3.11+
- Required libraries: Sentence Transformers, Hugging Face Transformers

## Set up the account and config file 

You can use the model with SambaStudio or Sambaverse. 

### Setup for SambaStudio

To perform SambaStudio setup, you must be a SambaNova customer with a SambaStudio account. 

1. Log in to SambaStudio and get your API authorization key. The steps for getting this key are described [here](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html#_acquire_the_api_key).
2. Select the LLM you want to use (e.g. Llama 2 70B chat) and deploy an endpoint for inference. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).
3. Update the `sn-ai-starter-kit/.env` config file in the root repo directory. Here's an example: 

```yaml
BASE_URL="https://api-stage.sambanova.net"
PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
VECTOR_DB_URL=http://localhost:6333
```
4. In the [config file](./config.yaml), set the variable `api` to `"sambastudio"`.

### Setup for Sambaverse

1. Create a Sambaverse account at [Sambaverse](sambaverse.sambanova.net) and select your model. 
2. Get your [Sambaverse API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) (from the user button).
3. In the repo root directory find the config file `sn-ai-starter-kit/.env` and specify the Sambaverse API key, as in the following example: 

```yaml
    SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
    SAMBAVERSE_URL="https://yoururl"
```

4. In the [config file](./config.yaml), set the `api` variable to `"sambaverse"`.


The script supports both SambaVerse and SambaStudio APIs. Depending on which API you want to use, provide the corresponding API keys and URLs in the .env file. If you choose to use the SNSDK instead of requests, make sure you have it installed and configured with your credentials.







# Using the model

After you've completed installation, you can use and evaluate it. 

## Prepare data and run the script

The standard workflow consists of data preparation and script execution. Follow these steps:

1. Place your data in a directory that you later specify when you run the script. The script defaults to PDFs but supports other file types.
2. Run the script with necessary parameters. For example:

   ```bash
   python scriptfine_tune_embed_model_name.py --input_data_directory ./your_data_directory --output_data_directory ./processed_data
   ```

### Additional arguments

The script supports the following arguments to customize the process:

- `--file_extension` for specifying file types.
- `--split_ratio` to adjust the train-validation dataset split.
- `--force_retrain` to force retraining even if a finetuned model exists.

Run `python fine_tune_embed_model.py --help` for a full list of arguments and their descriptions.

## Perform finetuning using pre-generated data

If you've previously generated synthetic data and wish to proceed directly to fine-tuning:

1. Run the script with the following arguments: 
* `--train_dataset_path` and `--val_dataset_path are the paths to your pre-generated datasets.
* Specify the model and output directory. 


For example:

   ```bash
   python fine_tune_embed_model.py --train_dataset_path ./processed_data/train_dataset.json --val_dataset_path ./processed_data/val_dataset.json --model_id "your_model_id" --model_output_path ./finetuned_model
   ```

## Evaluating a finetuned model

To evaluate an existing finetuned model without re-running the entire process, specify the model path and run evaluation only (`--evaluate_only` argument). Follow these steps: 

1. Use the `--model_output_path` argument to point to your finetuned model directory.
2. Ensure the dataset paths are specified if they are not in the default location. 
3. Specify `--evaluate_only`. 

For example:

   ```bash
   python fine_tune_embed_model.py --val_dataset_path ./processed_data/val_dataset.json --model_output_path ./finetuned_model --evaluate_only
   ```

## Using SambaNova systems (SNS) embeddings

The script supports integrating SNS (SambaNova systems) embeddings for enhancing the retrieval and understanding capabilities within LangChain workflows. Consider these points when using SNS embeddings: 

* **Configuration**: Ensure that you have the `export.env` file set up with your SNS credentials, including `EMBED_BASE_URL`, `EMBED_PROJECT_ID`, `EMBED_ENDPOINT_ID`, and `EMBED_API_KEY`.
* **Integration Example**: The script demonstrates how to use SNS embeddings for document and query encoding, followed by computing cosine similarity for a given query against a set of documents. Additionally, it integrates these embeddings into a LangChain retrieval workflow, showcasing an end-to-end example of query handling and document retrieval.

To run the SNS embeddings integration:

   ```bash
   python sns_embedding_script.py
   ```

The script executes the embedding process for documents and queries, computes similarities, and demonstrates a LangChain retrieval workflow using the predefined embeddings.



# Contributing

We welcome contributions! Feel free to improve the process or add features by submitting issues or pull requests.

# License

This project is licensed under the MIT License.

# Acknowledgments

- Sentence Transformers and Hugging Face for their resources and pre-trained models.
- Inspired by practices from [original embedding fine-tuning repository](https://github.com/run-llama/finetune-embedding/tree/main).
