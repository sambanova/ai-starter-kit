"""Module for Draft Model Training using SFTTrainer."""

import os

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

from typing import Optional


class DraftModelTrainer:
    """
    Trainer class for fine-tuning language models using the SFTTrainer.

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
        devices (str): CUDA devices to use (default "0").
    """

    def __init__(
        self,
        train_data: str,
        output_dir: str,
        run_name: str,
        learning_rate: float = 2e-5,
        epochs: int = 2,
        lr_scheduler_type: Optional[str] = 'constant',
        response_template: str = '<|start_header_id|>assistant<|end_header_id|>\n\n',
        base_model: str = '',
        save_strategy: str = 'epoch',
        save_steps: int = 1,
        gradient_accumulation_steps: int = 1,
        batch_size: int = 8,
        devices: str = '0',
    ) -> None:
        """
        Initialize the DraftModelTrainer with configuration values.

        Args:
            train_data (str): Path to the training data file.
            output_dir (str): Directory to save checkpoints.
            run_name (str): Name of the wandb run.
            learning_rate (float): Learning rate to use.
            epochs (int): Number of epochs for training.
            lr_scheduler_type (Optional[str]): Scheduler type to use.
            response_template (str): Template for formatting responses.
            base_model (str): Identifier or path to the pretrained base model.
            save_strategy (str): Strategy for saving checkpoints.
            save_steps (int): Number of steps between saves.
            gradient_accumulation_steps (int): Steps for gradient accumulation.
            batch_size (int): Per device training batch size.
            devices (str): GPU devices to use.
        """
        self.train_data = train_data
        self.output_dir = output_dir
        self.run_name = run_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lr_scheduler_type = lr_scheduler_type
        self.response_template = response_template.replace('\\n', '\n')
        self.base_model = base_model
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = batch_size
        self.devices = devices

    def train(self) -> None:
        """
        Set up the training pipeline and start model training.
        """
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
            dataset_text_field='training_text',
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
    Example main function to demonstrate training with DraftModelTrainer.

    Replace the placeholder paths and model identifier with actual values when running.
    """
    # Instantiate the DraftModelTrainer with example parameters
    trainer = DraftModelTrainer(
        train_data='e2e_draft_model_training/data/datasets/bio_train_completions/bio_train.json',
        output_dir='e2e_draft_model_training/transformers/results',
        run_name='example_run',
        learning_rate=2e-5,
        epochs=5,
        lr_scheduler_type='cosine',
        response_template='<|start_header_id|>assistant<|end_header_id|>\n\n',
        base_model='meta-llama/Llama-3.2-1B-Instruct',
        save_strategy='epoch',
        save_steps=500,
        gradient_accumulation_steps=1,
        batch_size=8,
        devices='0',
    )
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
