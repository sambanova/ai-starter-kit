# YoDa
YoDa is an acronym for "Your Data, Your Model". This project aims to train a Language Model (LLM) using customer's private data. The goal is to compete with general solutions on tasks that are related to the customer's data.

<p align="center">
  <img src="YoDa.png" alt="YoDa" width="300">
</p>

## Getting Started

These instructions will guide you on how to generate training data, preprocess it, train the model, evaluate it and finally launch the online service.

1. Update API information for the SambaNova LLM in your `.env` file.

In the example below 
```
BASE_URL="https://sjc3-demo2.sambanova.net"
PROJECT_ID="60774d44-3cc3-47eb-aa91-87fae2e8655e"
ENDPOINT_ID="b0e414eb-4863-4a8c-9839-3c2dfa718ae5"
API_KEY=""

FINETUNED_BASE_URL="https://sjc1-demo1.sambanova.net"
FINETUNED_PROJECT_ID=""
FINETUNED_ENDPOINT_ID=""
FINETUNED_API_KEY=""
DEMO1_API_KEY=""
``` 


2. Activate Python virtual environment.
```
conda create -n yoda python=3.10
conda activate yoda
pip install -r requirements.txt
```

3. Download the dataset from [here](https://drive.google.com/drive/folders/10chGQIgJJgBNvIdj8RL2sVwh8txnNkpO) and update
the `src_folder` variable in your config with this path.


#### For Domain adaptive pre-training and Instruction Finetune

Note: You will need a SambaStudio endpoint to the LLAMA 70B Chat model and add the configurations to your env file, which is used for synthetic data generation.
Please replace /path/to/config with your actual paths. An example config is shown in `configs/sn_expert_conf.yaml`
and this iss set as the default parameter for the data generation scripts below.

#### To Generate pretraining data
```
python -m src.gen_data
    --config /path/to/config
    --purpose pretrain 
```

#### To generate finetuning data
```
python -m src.gen_data
    --config /path/to/config
    --purpose finetune 
```

#### Or to do both in one go
```
python -m src.gen_data
    --config /path/to/config
    --purpose both 
```

### Preprocessing
In order to pretrain and finetune on SambaStudio,
we fist need the data to be in the format of hdf5 files that we can upload
To preprocess the data, open `scripts/preprocess.sh` and replace
the variables `ROOT_GEN_DATA_PREP_DIR` with the path to your [generative data preparation](https://github.com/sambanova/generative_data_prep)
directory, your output json from pretraining/finetuning with`INPUT_FILE`; and 
an `OUTPUT_DIR` where you want your hdf5 files to be dumped before you upload them to 
SambaStudio Datasets.:

```
sh scripts/preprocess.sh
```

### Launching pretraining/finetuning and hosting endpoints on SambaStudio

In our tutorial, we are creating and hosting checkpoints which needs to be done on SambaStudio. 
This can be done on the **SambaStudio GUI** as well as with **snapapi** and **snapsdk**. For those
interested in how this looks like with **snapsdk**, please have a look at the WIP notebook `SambaStudio_job_spinup.ipynb`

### Evaluation

For our evaluation, we pose the finetuned model questions from the held-out synthetic question-answer pairs we procured
when we were generating the finetuning data. We benchmark the approach against responses we get from also using RAG (not to dissimilar to the approach in chipnemo) as well as from 
a golden context.\
\
To assess the trained model, execute the following script:
```
python -m src.evaluate 
    --config /path/to/config.yaml 
```
Please replace  `/path/to/config.yaml`  with your actual paths.
