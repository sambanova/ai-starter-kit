# qa_processing.py
import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(os.path.join(repo_dir, 'utils'))

from typing import Any, Dict, List, Tuple

from utils.model_wrappers.api_gateway import APIGateway

from .data_reader import format_text, read_txt_data


def process_response_data(response_data: List[Dict[str, Any]]) -> Tuple[List[Any], List[Any]]:
    valid_count = 0
    all_qa_pairs = []
    need_to_regenerate = []
    question_set = set()
    for d in response_data:
        qa_pairs = []
        response_content = d['response_text'].strip('#')
        response_content = response_content.replace('</human>', '<human>').replace('</bot>', '<bot>')
        if '\n\n\n\n\n\n\n\n\n\n' in response_content:
            need_to_regenerate.append(d)
            continue

        article = format_text(read_txt_data(d['filepath']))

        if '###' in response_content:
            qa_pairs = response_content.strip().split('###')
            qa_pairs = [
                [q_a.split('<human>:')[1].split('<bot>:')[0].strip('\n '), q_a.split('<bot>:')[1].strip('\n ')]
                for q_a in qa_pairs
                if '<human>:' in q_a and '<bot>:' in q_a
            ]
            if len(qa_pairs) == 10:
                valid_count += 1
            if len(qa_pairs) < 5:
                qa_pairs = []
                for qa_pairs_temp in response_content.strip().split('###'):
                    qa_pairs_temp = qa_pairs_temp.split('\n\n')
                    qa_pairs += [
                        [q_a.split('<human>:')[1].split('<bot>:')[0].strip('\n '), q_a.split('<bot>:')[1].strip('\n ')]
                        for q_a in qa_pairs_temp
                        if '<human>:' in q_a and '<bot>:' in q_a
                    ]

        elif '<human>:' in response_content:
            qa_pairs = response_content.strip().split('\n\n')
            qa_pairs = [
                [q_a.split('<human>:')[1].split('<bot>:')[0].strip('\n '), q_a.split('<bot>:')[1].strip('\n ')]
                for q_a in qa_pairs
                if '<human>:' in q_a and '<bot>:' in q_a
            ]

        valid_qa_pairs = []
        for qa in qa_pairs:
            question = qa[0]
            if question not in question_set:
                question_set.add(question)
                valid_qa_pairs.append(qa)
            else:
                continue
        if len(valid_qa_pairs) <= 3 and d['prompt_length'] < 3500:
            need_to_regenerate.append(d)

        for qa in valid_qa_pairs:
            new_d = {}
            new_d['filename'] = d['filename']
            new_d['filepath'] = d['filepath']
            new_d['article'] = article
            new_d['question'] = qa[0]
            new_d['answer'] = qa[1]

            all_qa_pairs += [new_d]
    return all_qa_pairs, need_to_regenerate


def format_qa_data(qa_data: List[Dict[str, Any]]) -> list[Dict[str, Any]]:
    processed_count = 0
    training_data = []
    for d in qa_data:
        question = d['question'].strip(' \n')
        assert '###' not in question
        assert '\n\n<human>' not in d['answer']
        answer = d['answer'].replace('?', "'")

        question_with_context = (
            f'Here is some relevant context that might assist in answering the SambaNova-related'
            f' question.\n\n{d["article"]}\n\nAnswer the following'
            f' SambaNova-related question: {question}\nAnswer:'
        )

        completion = answer
        training_data.append(
            {
                'question': question,
                'answer': answer,
                'prompt': question,
                'completion': answer,
                'question_with_context': question_with_context,
                'context_filepath': d['filepath'],
            }
        )
        processed_count += 1

    logging.info('processed_answers: {}'.format(processed_count))
    logging.info('total training data: {}'.format(len(training_data)))
    return training_data


def generate_qa_pairs(
    sample: Dict[str, Any],
    template: str,
    base_url: str,
    project_id: str,
    endpoint_id: str,
    api_key: str,
    tokenizer: Any,
) -> Tuple[Any, int]:
    article = sample['article']
    prompt = article + template

    # hacky way to control output length so that the endpoint can survive

    prompt_length = len(tokenizer.encode(prompt))
    max_output_token = min(4096 - prompt_length, 1000)
    print(max_output_token)
    if max_output_token <= 20:
        # too few tokens left for generation, skip this article
        return '', prompt_length

    llm = APIGateway.load_llm(
        type='sambastudio',
        streaming=True,
        coe=True,
        do_sample=False,
        max_tokens_to_generate=500,
        temperature=0.0,
        select_expert='Meta-Llama-3-8B-Instruct',
        process_prompt=False,
    )
    response_text = llm(prompt=prompt)
    return response_text, prompt_length
