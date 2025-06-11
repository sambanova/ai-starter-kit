"""
This module implements Draft Model Training using `trl.SFTTrainer`.
It should be used in conjunction with the `03_config_training.yaml` configuration.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb
import yaml
from datasets import load_dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer  # type: ignore


class DraftModelTrainer:
    """
    Trainer class for fine-tuning language models using `trl.SFTTrainer`.

    Attributes:
        train_data (str): Path to the training data JSON file.
        output_dir (str): Directory where checkpoints and model artifacts will be saved.
        run_name (str): Name for the run.
        learning_rate (float): Learning rate for training.
        epochs (int): Number of training epochs.
        lr_scheduler_type (Optional[str]): Type of learning rate scheduler.
        response_template (str): Template string for responses.
        base_model (str): Pretrained base model identifier.
        save_strategy (str): Strategy for saving checkpoints.
        save_steps (int): Frequency of saving steps.
        gradient_accumulation_steps (int): Number of gradient accumulation steps.
        batch_size (int): Batch size for training.
        devices (str): CUDA devices to use, e.g. "0" or "1".
        dataset_text_field (str): Key name for the text field in the dataset.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the DraftModelTrainer using a YAML configuration file.

        The configuration file is expected to contain keys corresponding to the trainer's attributes.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

        # Load YAML configuration.
        with open(config_file, 'r', encoding='utf-8') as file:
            config: Dict[str, Any] = yaml.safe_load(file)

        # Read configuration settings.
        self.train_data: str = config.get('train_data', '')
        self.output_dir: str = config.get('output_dir', '')
        self.run_name: str = config.get('run_name', '')
        self.learning_rate: float = config.get('learning_rate', 2e-5)
        self.epochs: int = config.get('epochs', 2)
        self.lr_scheduler_type: Optional[str] = config.get('lr_scheduler_type', 'constant')
        self.response_template: str = config.get(
            'response_template', '<|start_header_id|>assistant<|end_header_id|>\n\n'
        ).replace('\\n', '\n')
        self.dataset_text_field: str = config.get('dataset_text_field', 'text')
        self.base_model: str = config.get('base_model', '')
        self.save_strategy: str = config.get('save_strategy', 'epoch')
        self.save_steps: int = config.get('save_steps', 1)
        self.gradient_accumulation_steps: int = config.get('gradient_accumulation_steps', 1)
        self.batch_size: int = config.get('batch_size', 8)
        self.devices: str = config.get('devices', '0')

        # Create the output directory if it doesn't exist yet
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self) -> None:
        """Set up the training pipeline and start model training."""

        # Setup configuration for wandb
        config = {
            'learning_rate': self.learning_rate,
            'train_data': self.train_data,
            'epochs': self.epochs,
            'base_model': self.base_model,
            'devices': self.devices,
            'run_name': self.run_name,
            'output_dir': self.output_dir,
            'lr_scheduler_type': self.lr_scheduler_type,
        }

        # Initialize wandb run
        wandb.init(project='Draft_Model_Training', config=config, name=self.run_name)

        # Load dataset
        dataset = load_dataset(
            'json',
            data_files={
                'train': self.train_data,
            },
        )

        # Set CUDA devices
        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices

        # Prepare training arguments using SFTConfig
        training_arguments = SFTConfig(
            output_dir=self.output_dir,  # output directory
            num_train_epochs=self.epochs,  # number of epochs
            per_device_train_batch_size=self.batch_size,  # batch size per device
            gradient_accumulation_steps=self.gradient_accumulation_steps,  # accumulation steps
            optim='adamw_torch',
            logging_steps=100,
            learning_rate=self.learning_rate,
            fp16=False,
            bf16=False,
            lr_scheduler_type=self.lr_scheduler_type,
            report_to='wandb',
            eval_strategy='no',
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            run_name=self.run_name,
            warmup_ratio=0.03,
            dataset_text_field=self.dataset_text_field,
        )

        # Load model using the base model identifier
        model = AutoModelForCausalLM.from_pretrained(self.base_model, device_map='auto', torch_dtype=torch.bfloat16)

        # Load tokenizer and set pad token to eos token
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        # Create data collator for the model
        collator = DataCollatorForCompletionOnlyLM(response_template=self.response_template, tokenizer=tokenizer)

        # Create the trainer using SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=dataset['train'].shuffle(42),
            data_collator=collator,
            processing_class=tokenizer,
        )

        # Start training
        trainer.train()

        # Save the final model and tokenizer
        trainer.model.save_pretrained(self.output_dir)
        trainer.tokenizer.save_pretrained(self.output_dir)

        # Finish the wandb run
        wandb.finish()


def main() -> None:
    """
    Main function to train with DraftModelTrainer.

    The configuration is loaded via the path passed to the DraftModelTrainer constructor.
    """
    # Configuration file path
    config_path: str = '03_config_training.yaml'

    # Instantiate the DraftModelTrainer with the configuration file path.
    trainer = DraftModelTrainer(config_path)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
