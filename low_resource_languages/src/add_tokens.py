import os

import yaml

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config.yaml', type=str, help='Path to configuration file',
                        required=True)
    #parser.add_argument('--base_tokenizer', default='llama_2/Llama-2-7b-hf', type=str,
    #                    help='Path to base Hugging Face tokenizer to add tokens to')
    #parser.add_argument('--sp_model_file', default=None, type=str,
    #                    help='Path to sentence piece tokenizer model file to add to the base_tokenizer')
    #parser.add_argument('--output_dir', default=None, type=str, help='Path to save the finalized output tokenizer to ')
    args = parser.parse_args()
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    merging_sp_model_file = config['sp_model_file']
    merging_sp_model = spm.SentencePieceProcessor()
    merging_sp_model.Load(merging_sp_model_file)

    llama_tokenizer = LlamaTokenizer.from_pretrained(config['sp_model_file'])  # I've added this myself

    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    merging_spm = sp_pb2_model.ModelProto()
    merging_spm.ParseFromString(merging_sp_model.serialized_model_proto())

    print(f'Number of original tokens: {len(llama_tokenizer)}',
          f'Number of original tokens we are adding: {len(merging_sp_model)}')
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)

    pieces_to_add = []
    for p in merging_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            pieces_to_add.append(new_p)

    for new_p in pieces_to_add:
        llama_spm.pieces.append(new_p)

    print(f'New vocab size: {len(llama_spm.pieces)}')
    ## Save
    os.makedirs(config['output_dir'], exist_ok=True)
    output_hf_dir = os.path.join(config['output_dir'], 'hf_tokenizer')
    os.makedirs(output_hf_dir, exist_ok=True)
    output_sp_model_path = config['output_dir'] + '/sentence_piece_tokenizer.model'
    with open(output_sp_model_path, 'wb') as f:
        f.write(llama_spm.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=output_sp_model_path)

    tokenizer.save_pretrained(output_hf_dir)
    print(f"Merged tokenizer has been saved to {output_hf_dir}")
