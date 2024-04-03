import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config.yaml', type=str, help='Path to configuration file',
                        required=True)
    #parser.add_argument('--model_path', required=True, type=str)
    #parser.add_argument('--output_model_path', required=True, type=str)
    #parser.add_argument('--tokenizer_path', required=True, type=str)
    #parser.add_argument('--target_config', required=True, type=str)
    #parser.add_argument('--init_method', choices=['std_xavier', 'std_hf', 'avg_all', 'avg_tokens', 'zero', 'zero_all', 'hf_default', 'avg_all_w_head', 'avg_tokens_w_head'], type=str, default='std_xavier')
    args = parser.parse_args()
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    orig_tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    new_tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(config['model_path'])
    embed_tokens_key = 'embed_tokens'
    lm_head_key = 'lm_head'
    embedding = getattr(model.model, embed_tokens_key)
    avg_embedding = embedding.weight.data.detach().mean(dim=0)
    old_vocab_size = model.vocab_size
    target_config = AutoConfig.from_pretrained(config['target_config'])
    # resize_token_embeddings will do a random initialization of the new embeddings
    # call to init: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1881
    # random normal init: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L863-L866
    print(f'Resizing token embeddings to {target_config.vocab_size}')
    model.resize_token_embeddings(target_config.vocab_size)

    # first: set the unused vocab terms to a very negative number
    # this essentially sets the prob for them to be 0
    # don't want to use dtype min to avoid overflow if we reduce precision
    num_unused_vocab = target_config.vocab_size - new_tokenizer.vocab_size
    if config['init_method'] != 'hf_default':
        print(f'Setting last {num_unused_vocab} unused embeddings to -10000')
        embedding = getattr(model.model, embed_tokens_key)
        embedding_dim = embedding.weight.data.shape[1]
        embedding_dtype = embedding.weight.data.dtype
        assert embedding_dim == target_config.hidden_size
        embedding.weight.data[-num_unused_vocab:, :] = torch.ones((num_unused_vocab, embedding_dim), dtype=embedding.weight.data.dtype) * -10000

    lm_head = getattr(model, lm_head_key)

    print(f"Initializing with {config['init_method']}")
    if config['init_method'] == 'std_xavier':
        reinitialized_weights = torch.nn.init.xavier_uniform_(embedding.weight.data[old_vocab_size:new_tokenizer.vocab_size, :])
        embedding.weight.data[old_vocab_size:new_tokenizer.vocab_size, :] = reinitialized_weights
    elif 'avg_all' in config['init_method']:
        embedding.weight.data[old_vocab_size:new_tokenizer.vocab_size, :] = avg_embedding.unsqueeze(0)
        if 'w_head' in config['init_method']:
            avg_lmhead_embedding = lm_head.weight.data[:old_vocab_size, :].detach().mean(0)
            lm_head.weight.data[old_vocab_size:new_tokenizer.vocab_size, :] = avg_lmhead_embedding.unsqueeze(0)
    elif 'avg_tokens' in config['init_method']:
        for new_token_id in tqdm(range(old_vocab_size, new_tokenizer.vocab_size), desc='Initializing new tokens', dynamic_ncols=True):
            new_token = new_tokenizer.decode(new_token_id)
            old_token_ids = orig_tokenizer.encode(new_token, add_special_tokens=False)
            embeddings_to_avg = embedding.weight.data[old_token_ids, :].detach()
            embedding.weight.data[new_token_id] = embeddings_to_avg.mean(dim=0)
            if 'w_head' in config['init_method']:
                lmhead_embeddings_to_avg = lm_head.weight.data[old_token_ids, :].detach()
                lm_head.weight.data[new_token_id] = lmhead_embeddings_to_avg.mean(dim=0)
    elif config['init_method'] == 'zero':
        embedding.weight.data[old_vocab_size:new_tokenizer.vocab_size, :] = 0.0
    elif config['init_method'] == 'zero_all':
        embedding.weight.data[old_vocab_size:, :] = 0.0
    else:
        assert 'hf' in config['init_method']

    # sanity checks
    assert model.vocab_size == target_config.vocab_size
    assert getattr(model.model, embed_tokens_key).weight.data.shape[0] == target_config.vocab_size
    if num_unused_vocab > 0 and not (config['init_method'] in ['zero', 'zero_all', 'hf_default']):
        assert torch.max(getattr(model.model, embed_tokens_key).weight.data[-num_unused_vocab:, :]).item() == -10000
        assert torch.min(getattr(model.model, embed_tokens_key).weight.data[-num_unused_vocab:, :]).item() == -10000
    print(f"Saving model to {config['output_model_path']}")
    os.makedirs(config['output_model_path'], exist_ok=True)
    # set safe_serialization=False so that models are compatible with RDU
    model.save_pretrained(config['output_model_path'], safe_serialization=False)

    # also save the arguments!
    config['orig_vocab_size'] = old_vocab_size
    config['new_vocab_size'] = new_tokenizer.vocab_size
    config['total_vocab_size'] = target_config.vocab_size
    for tokenizer_file in ['special_tokens_map.json', 'tokenizer_config.json', 'tokenizer.model']:
        shutil.copy(Path(config['tokenizer_path']) / tokenizer_file, config['output_model_path'])
    with open(Path(config['output_model_path']) / 'embedding_extension_args.json', 'w') as f:
        json.dump(vars(args), f)
    print('Done!')