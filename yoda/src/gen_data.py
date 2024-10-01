import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import argparse
import logging
import random

import jsonlines
import tqdm
import yaml
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv(os.path.join(repo_dir, '.env'))

from typing import Any, Dict, List, Optional

from yoda.prompts.prompts import QA_GEN_TEMPLATE
from yoda.tools import data_reader, qa_processing

# set the log format to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='sn_expert_conf.yaml', help='path to the config file')
    parser.add_argument(
        '--purpose',
        type=str,
        default='pretrain',
        choices=['pretrain', 'finetune', 'both'],
        help='purpose of the training data, pretrain or finetune or both',
    )

    return parser.parse_args()


# create training data for pretraining


def pretrain(
    articles: List[Dict[str, Any]],
    data_dir: str,
    processed_data_folder: str = 'processed_data',
    output_file_name: str = 'article_data.jsonl',
) -> List[Any]:
    article_data: List[Any] = []
    for d in articles:
        article = d['article']
        # clean the article
        article = article.replace('.\n', '.[NEWLINE]]')
        article = article.replace('\n', '')
        article = article.replace('[NEWLINE]]', '\n')
        article_data.append({'prompt': '', 'completion': article})
        # print sample data for debugging
        # print(article)

    output_path = os.path.join(data_dir, processed_data_folder, output_file_name)
    with jsonlines.open(output_path, 'w') as wf:
        wf.write_all(article_data)

    return article_data


# create training data for finetuning
def finetune(
    articles: List[Dict[str, Any]],
    data_dir: str,
    tokenizer: Any,
    pretrain_data: Optional[Any] = None,
    n_eval_samples: int = 50,
) -> None:
    # Get LLM response for each article
    for doc in tqdm.tqdm(articles):
        filename = doc['filename']
        filepath = doc['filepath']
        response_text, prompt_length = qa_processing.generate_qa_pairs(
            sample=doc,
            template=QA_GEN_TEMPLATE,
            base_url=os.getenv('BASE_URL', ''),
            project_id=os.getenv('PROJECT_ID', ''),
            endpoint_id=os.getenv('ENDPOINT_ID', ''),
            api_key=os.getenv('API_KEY', ''),
            tokenizer=tokenizer,
        )

        # Write the result into a jsonl file
        with jsonlines.open(
            os.path.join(data_dir, 'response_data', 'llama2_70b_chat_qa_gen_responses.jsonl'), mode='a'
        ) as writer:
            writer.write(
                {
                    'filename': filename,
                    'filepath': filepath,
                    'prompt_length': prompt_length,
                    'response_text': response_text,
                }
            )

    # define input and output file paths
    response_file = os.path.join(data_dir, 'response_data', 'llama2_70b_chat_qa_gen_responses.jsonl')
    qa_train_file = os.path.join(data_dir, 'processed_data', 'synthetic_qa_train.jsonl')
    qa_eval_file = os.path.join(data_dir, 'processed_data', 'synthetic_qa_eval.jsonl')
    regeneration_list_file = os.path.join(data_dir, 'response_data', 'response_needed_regeneration.jsonl')

    # process the responses data into QA data
    response_data = data_reader.read_jsonl_data(response_file)
    qa_data, need_to_regenerate = qa_processing.process_response_data(response_data)
    templated_data = qa_processing.format_qa_data(qa_data)
    random.shuffle(templated_data)
    eval_data = templated_data[:n_eval_samples]
    training_data = templated_data[n_eval_samples:]

    with jsonlines.open(qa_train_file, 'w') as wf:
        wf.write_all(training_data)

    with jsonlines.open(qa_eval_file, 'w') as wf:
        wf.write_all(eval_data)

    with jsonlines.open(regeneration_list_file, 'w') as wf:
        wf.write_all(need_to_regenerate)

    # save a mixture of synthetic qa data and article data
    qa_article_mix_file = os.path.join(data_dir, 'processed_data', 'qa_article_mix.jsonl')
    if pretrain_data:
        qa_article_mix_data = training_data + pretrain_data
    else:
        qa_article_mix_data = training_data
    random.shuffle(qa_article_mix_data)

    with jsonlines.open(qa_article_mix_file, 'w') as wf:
        wf.write_all(qa_article_mix_data)


# main entry point
if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    # all the generated training and evaluation data will be store under DATA_DIR
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Destination folder for storing the training data
    data_dir = config['dest_folder']
    os.makedirs(data_dir, exist_ok=True)
    # create sub-folders within DATA_DIR
    os.makedirs(os.path.join(data_dir, 'processed_data'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'response_data'), exist_ok=True)

    # Source folder for the raw data
    base_dir = config['src_folder']
    subfolders = config['src_subfolders']
    tokenizer_path = config['tokenizer']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    articles = data_reader.collect_articles([os.path.join(base_dir, folder) for folder in subfolders])
    logging.info('number of articles: {}'.format(len(articles)))

    n_eval_samples = config['n_eval_samples']

    if args.purpose == 'pretrain':
        pretrain(articles, data_dir)
    elif args.purpose == 'finetune':
        finetune(articles, tokenizer=tokenizer, data_dir=data_dir, n_eval_samples=n_eval_samples)
    else:
        assert args.purpose == 'both'
        pretrain_data = pretrain(articles, data_dir)
        finetune(
            articles, tokenizer=tokenizer, data_dir=data_dir, pretrain_data=pretrain_data, n_eval_samples=n_eval_samples
        )
