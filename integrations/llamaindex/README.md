![sambanova_logo](../../images/SambaNova-dark-logo-1.png)

# LlamaCloud integrations with SambaNova

This repository contains a collection of cookbooks to show you how to build LLM applications using SambaNova Cloud to help manage your data pipelines, and LlamaIndex as the core orchestration framework.

## Getting Started

1. Follow the instructions in the section below for setting up the Environment. You may need to refresh your kernel list once in the notebook or reopen your IDE.
1. Open one of the Jupyter notebooks in this repo (e.g. `examples/contract_review.ipynb`), follow the steps to create a SambaNova API Key and paste it into an env file in the root of this repository.

That should get you started! You should now be able to create an e2e pipeline with a LlamaCloud pipeline as the backend.

## Setting up the Environment
Here's some commands for installing the Python dependencies & running Jupyter.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=venv
```

Notebooks are in `examples`.

Note: if you encounter package issues when running notebook examples, please `rm -rf .venv` and repeat the above steps again.
