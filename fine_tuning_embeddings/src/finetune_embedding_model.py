import argparse
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.corpus = None
        self.queries = None
        self.relevant_docs = None
        self.load_dataset()

    def load_dataset(self):
        with open(self.dataset_path, 'r+') as f:
            dataset = json.load(f)
        self.corpus = dataset['corpus']
        self.queries = dataset['queries']
        self.relevant_docs = dataset['relevant_docs']

    def get_examples(self):
        examples = []
        for query_id, query in self.queries.items():
            node_id = self.relevant_docs[query_id][0]
            text = self.corpus[node_id]
            examples.append(InputExample(texts=[query, text]))
        return examples

class FineTuneModel:
    def __init__(self, model_id, train_dataset_path, val_dataset_path, batch_size, epochs, output_path, checkpoint_path=None, save_best_model=False):
        self.model = SentenceTransformer(model_id, device='cuda')
        self.train_loader = DataLoader(DatasetLoader(train_dataset_path).get_examples(), batch_size=batch_size)
        self.val_loader = DatasetLoader(val_dataset_path)
        self.loss = losses.MultipleNegativesRankingLoss(self.model)
        self.epochs = epochs
        self.output_path = output_path
        self.evaluator = InformationRetrievalEvaluator(
            self.val_loader.queries,
            self.val_loader.corpus,
            self.val_loader.relevant_docs
        )

    def train(self):
        warmup_steps = int(len(self.train_loader) * self.epochs * 0.1)
        self.model.fit(
            train_objectives=[(self.train_loader, self.loss)],
            epochs=self.epochs,
            warmup_steps=warmup_steps,
            output_path=self.output_path,
            show_progress_bar=True,
            evaluator=self.evaluator,
            evaluation_steps=50,
            checkpoint_path=self.output_path,
            save_best_model=True
        )

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a SentenceTransformer model on a given dataset.")
    parser.add_argument("--model_id", type=str, default="intfloat/e5-large-v2", help="Model identifier from SentenceTransformers")
    parser.add_argument("--train_dataset_path", type=str, default="./data/train_dataset.json", help="Path to the training dataset")
    parser.add_argument("--val_dataset_path", type=str, default="./data/val_dataset.json", help="Path to the validation dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("--output_path", type=str, default="exp_finetune", help="Output path for the fine-tuned model")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Path to the checkpoint to resume training")
    
    args = parser.parse_args()

    finetune_model = FineTuneModel(
        model_id=args.model_id,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
        save_best_model=True
    )

    finetune_model.train()

if __name__ == "__main__":
    main()
