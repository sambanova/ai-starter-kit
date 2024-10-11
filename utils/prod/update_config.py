import argparse
import os
import re
from typing import Any, Dict, List, Optional

import yaml

# List of starter kits for production updates
PROD_KITS: List[str] = [
    'benchmarking',
    'enterprise_knowledge_retriever',
    'financial_assistant',
    'function_calling',
    'search_assistant',
    'multimodal_knowledge_retriever',
]

# List of starter kits for test updates
TEST_KITS: List[str] = [
    'benchmarking',
    'enterprise_knowledge_retriever',
    'financial_assistant',
    'function_calling',
    'search_assistant',
    'image_search',
    'multimodal_knowledge_retriever',
    'post_call_analysis',
    'prompt_engineering',
    'web_crawled_data_retriever',
]


def get_nested_dict(data: Dict[str, Any], keys: List[str]) -> Optional[Dict[str, Any]]:
    """Retrieve a nested dictionary given a list of keys."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data if isinstance(data, dict) else None


def update_nested_dict(data: Dict[str, Any], keys: List[str], value: Any) -> None:
    """Update a nested dictionary given a list of keys and a value."""
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value


def update_embedding_model(config: Dict[str, Any]) -> None:
    """Update the embedding model configuration."""
    embedding_model = get_nested_dict(config, ['rag', 'embedding_model']) or get_nested_dict(
        config, ['embedding_model']
    )
    if embedding_model:
        embedding_model['type'] = 'sambastudio'
        embedding_model['batch_size'] = 32
        embedding_model['coe'] = False


def update_config_prod(config: Dict[str, Any]) -> None:
    """Update configuration for production mode."""
    update_nested_dict(config, ['prod_mode'], True)
    update_embedding_model(config)

    audio_model = get_nested_dict(config, ['audio_model'])
    if audio_model:
        audio_model['model'] = 'whisper-1'

    # Update pages_to_show for prod mode
    update_nested_dict(config, ['pages_to_show'], ['synthetic_eval'])


def update_makefile(root_dir: str, port: int) -> None:
    """Update the STREAMLIT_PORT in the Makefile."""
    makefile_path = os.path.join(root_dir, 'Makefile')
    if os.path.exists(makefile_path):
        with open(makefile_path, 'r', encoding='utf-8') as file:
            makefile_content = file.read()

        new_makefile_content = re.sub(
            r'^(STREAMLIT_PORT\s*:=\s*)\d+',
            f'\\g<1>{port}',
            makefile_content,
            flags=re.MULTILINE,
        )

        with open(makefile_path, 'w', encoding='utf-8') as file:
            file.write(new_makefile_content)
        print(f'Updated STREAMLIT_PORT to {port} in {makefile_path}')
    else:
        print(f'Makefile not found at {makefile_path}. Skipping STREAMLIT_PORT update.')


def update_configs(mode: str = 'prod', port: int = 8501) -> None:
    """Update configurations for all kits based on the specified mode."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    kits = PROD_KITS if mode == 'prod' else TEST_KITS

    for kit_name in kits:
        kit_path = os.path.join(root_dir, kit_name)
        config_path = os.path.join(kit_path, 'config.yaml')

        if not os.path.exists(config_path):
            print(f'Config file not found for {kit_name}. Skipping.')
            continue

        with open(config_path, 'r', encoding='utf-8') as file:
            config: Dict[str, Any] = yaml.safe_load(file)

        if mode == 'prod':
            update_config_prod(config)
        else:
            update_embedding_model(config)

        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f'Updated config for {kit_name} in {mode} mode.')

    if mode == 'prod':
        update_makefile(root_dir, port)


def main() -> None:
    """Main function to parse arguments and run the update process."""
    parser = argparse.ArgumentParser(description='Update configurations for AI Starter Kits.')
    parser.add_argument(
        'mode',
        nargs='?',
        default='prod',
        choices=['prod', 'test'],
        help='The mode to run in (prod or test). Default is prod.',
    )
    parser.add_argument('--port', type=int, default=8501, help='The port to use for Streamlit. Default is 8501.')

    args = parser.parse_args()

    update_configs(args.mode, args.port)


if __name__ == '__main__':
    main()
