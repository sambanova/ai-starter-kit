"""
In this script, we train our target language tokenizer.

Output is  config.json  special_tokens_map.json  tokenizer_config.json  tokenizer.model
"""

import argparse

import sentencepiece as spm
import yaml


def train_tokenizer(input_file, vocab_size, character_coverage, num_threads):
    spm.SentencePieceTrainer.train(
        input=input_file,
        vocab_size=vocab_size,
        model_prefix='lrl',
        character_coverage=character_coverage,
        model_type='bpe',
        num_threads=num_threads
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SentencePiece tokenizer')
    parser.add_argument('--config_file', default='config.yaml', type=str, help='Path to configuration file',
                        required=True)
    #parser.add_argument('--input_file', type=str, help='Input file path for one-sentence-per-line raw corpus file', required=True)
    #parser.add_argument('--vocab_size', type=int, help='Vocabulary size', required=True)
    #parser.add_argument('--character_coverage', type=float, default=0.9995, help='What percent of characters to cover')
    #parser.add_argument('--num_threads', type=int, default=4, help='How many parallel CPU cores to run on')
    args = parser.parse_args()
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    input_file = config['input_file']
    vocab_size = config['vocab_size']
    character_coverage = config['character_coverage']
    num_threads = config['num_threads']

    train_tokenizer(input_file, vocab_size, character_coverage, num_threads)
