import argparse
import sentencepiece as spm
import os

def train_tokenizer(input_file, model_prefix, vocab_size, user_defined_symbols, character_coverage, num_threads):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        user_defined_symbols=user_defined_symbols,
        character_coverage=character_coverage,
        model_type='bpe',
        num_threads=num_threads
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SentencePiece tokenizer')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--model_prefix', type=str, default='lrl', help='Prefix for the model files')
    parser.add_argument('--vocab_size', type=int, help='Vocabulary size')
    parser.add_argument('--character_coverage', type=float, default=0.9995, help='What percent of characters to cover')
    parser.add_argument('--user_defined_symbols', nargs='+', help='List of user-defined symbols')
    parser.add_argument('--num_threads', type=int, help='How many parallel CPU cores to run on')
    args = parser.parse_args()

    input_file = args.input
    model_prefix = args.model_prefix
    vocab_size = args.vocab_size
    user_defined_symbols = args.user_defined_symbols
    character_coverage = args.character_coverage
    num_threads = args.num_threads

    train_tokenizer(input_file, model_prefix, vocab_size, user_defined_symbols, character_coverage, num_threads)
