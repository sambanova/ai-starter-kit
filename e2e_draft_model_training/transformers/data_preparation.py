import os
import json
import datasets
from transformers import AutoTokenizer


def tokenize_dialog(item, tokenizer):
    if 'conversation' not in item:
        dialog = [{'role': 'user', 'content': item['prompt']}, {'role': 'assistant', 'content': item['completion']}]
    else:
        dialog = item['conversation']
    tokenized_input = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=False)
    # return {"conversation": dialog, "training_text": tokenized_input}
    return {'training_text': tokenized_input}


def get_dataset(file_paths, tokenizer, output_path=None):
    print('file paths: ', file_paths)
    if len(file_paths) > 1:
        dataset = datasets.load_dataset('json', data_files=file_paths, split='train', streaming=False)
        # column_types = datasets.Features({
        #     "prompt": datasets.Value("string"),
        #     "completion": datasets.Value("string")
        # })
        # dataset = dataset.cast(column_types)

        dataset = dataset.map(lambda x: tokenize_dialog(x, tokenizer))
        # column_types = datasets.Features({
        #     "training_text": datasets.Value("string")
        # })
        # dataset = dataset.cast(column_types)

        assert output_path != None
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    else:
        file_path = file_paths[0]
        file_extension = os.path.splitext(file_path)[-1].lower()
        # Load dataset based on file type
        if file_extension == '.jsonl':
            dataset = datasets.load_dataset('json', data_files=file_path, split='train', streaming=True)
        else:
            dataset = datasets.load_dataset('json', data_files=file_path)['train']

        dataset = dataset.map(lambda x: tokenize_dialog(x, tokenizer))
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'training_text'])
        data_list = dataset.to_list()

        if not output_path:
            output_path = file_path.rsplit('.json', 1)[0].rsplit('.jsonl', 1)[0] + '_templated.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)

    print(f'Processed file saved at: {output_path}')


if __name__ == '__main__':
    # Extract the configuration parameters
    with open('01_config_data_preparation', 'r', encoding='utf-8') as file:
        config: Dict[str, Any] = yaml.safe_load(file)

    file_path: str = config['file_path']
    model: str = config['model']
    output_path = config.get(['output_path'], None)
    if output_path is None:
        output_path = file_path.rsplit('.json', 1)[0].rsplit('.jsonl', 1)[0] + '_templated.json'

    tokenizer = AutoTokenizer.from_pretrained(model)

    if os.path.isdir(file_path):
        file_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.jsonl')]
    else:
        file_paths = file_path.split(',')
    get_dataset(file_paths, tokenizer, output_path)
