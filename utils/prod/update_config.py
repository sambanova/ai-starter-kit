import os
import re
from typing import Any, Dict, List

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

def update_embedding_model(config: Dict[str, Any]) -> None:
    """Update the embedding model configuration."""
    if 'embedding_model' in config:
        embedding_model = config['embedding_model']
        if isinstance(embedding_model, dict):
            embedding_model['type'] = 'sambastudio'
            embedding_model['batch_size'] = 32
            embedding_model['coe'] = False

def update_config_prod(config: Dict[str, Any]) -> None:
    """Update configuration for production mode."""
    if 'prod_mode' in config:
        config['prod_mode'] = True

    update_embedding_model(config)

    if 'audio_model' in config:
        audio_model = config['audio_model']
        if isinstance(audio_model, dict):
            audio_model['model'] = 'whisper-1'

def update_makefile(root_dir: str) -> None:
    """Update the STREAMLIT_PORT in the Makefile."""
    makefile_path = os.path.join(root_dir, 'Makefile')
    if os.path.exists(makefile_path):
        with open(makefile_path, 'r') as file:
            makefile_content = file.read()

        new_makefile_content = re.sub(
            r'^(STREAMLIT_PORT\s*:=\s*)\d+', r'\g<1>443', makefile_content, flags=re.MULTILINE
        )

        with open(makefile_path, 'w') as file:
            file.write(new_makefile_content)
        print(f'Updated STREAMLIT_PORT to 443 in {makefile_path}')
    else:
        print(f'Makefile not found at {makefile_path}. Skipping STREAMLIT_PORT update.')

def update_configs(mode: str = 'prod') -> None:
    """Update configurations for all kits based on the specified mode."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    kits = PROD_KITS if mode == 'prod' else TEST_KITS

    for kit_name in kits:
        kit_path = os.path.join(root_dir, kit_name)
        config_path = os.path.join(kit_path, 'config.yaml')

        if not os.path.exists(config_path):
            print(f"Config file not found for {kit_name}. Skipping.")
            continue

        with open(config_path, 'r') as file:
            config: Dict[str, Any] = yaml.safe_load(file)

        if mode == 'prod':
            update_config_prod(config)
        else:
            update_embedding_model(config)

        with open(config_path, 'w') as file:
            yaml.dump(config, file)

        print(f"Updated config for {kit_name} in {mode} mode.")

    if mode == 'prod':
        update_makefile(root_dir)

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 2:
        print('Usage: python update_config.py [mode]')
        sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) == 2 else 'prod'
    if mode not in ['prod', 'test']:
        print(f"Invalid mode: {mode}. Choose 'prod' or 'test'.")
        sys.exit(1)

    update_configs(mode)