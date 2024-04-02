Low-resource languages
======================

# Overview

## About this template

This AI Starter Kit is an example of how to train an LLM for a low-resource language, when we have a shortage of target language training data.
It includes:
 -   Scripts to prepare the tokenizer and model for training on SambaStudio.
 -   Instructions for uploading the model and datasets to SambaStudio and details on training.
 -   Illustration of how the finetuned model behaves.


## Workflow
There are two main parts to this process: (1) preparing the model,tokenizer, and dataset for training and (2) Launching the training and inference on Sambastudio

### Preparing the model and tokenizer

Step 1: Train sentencepiece tokenizer (train_sentencepiece_tokenizer.py)

    python train_sentencepiece_tokenizer.py --input --vocab_size --num_thread

Step 2: Add tokens (add_tokens.py)

    python add_tokens.py --base_tokenizer --sp_model_file --output_dir

Step 3: Initialize new token Embeddings (checkpoint_extension.py)

    python checkpoint_extension.py --model_path --output_model_path --tokenizer_path --target_config 

### Launching the training on SambaStudio
The next step is to upload the model and the datasets to SambaStudio. [ADD screenshots]


### Inference

To run a trans


