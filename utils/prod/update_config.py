import argparse
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


def update_all_embedding_models(config: Any) -> None:
    """Recursively update all 'embedding_model' configurations in the config dictionary.

    This function traverses the entire configuration dictionary recursively,
    finds all occurrences of the 'embedding_model' key regardless of their nesting,
    and updates their settings to standardize the embedding model to use 'sambastudio'.

    Args:
        config: The configuration dictionary or list to update.
    """
    if isinstance(config, dict):
        for key, value in config.items():
            if key == 'embedding_model' and isinstance(value, dict):
                # Overwrite the embedding model settings to use 'sambastudio'
                value['type'] = 'sambastudio'
                value['batch_size'] = 32
                value['coe'] = False
                # We standardize the embedding model settings to use 'sambastudio',
                # which is suitable for hosted kits that can utilize our embeddings.
            else:
                # Recursively call the function for nested dictionaries or lists
                update_all_embedding_models(value)
    elif isinstance(config, list):
        for item in config:
            update_all_embedding_models(item)


def update_config_prod(config: Dict[str, Any]) -> None:
    """Update configuration for production mode.

    Args:
        config: The configuration dictionary to update.
    """
    # Enforcing 'prod_mode' ensures that production-specific optimizations
    # and configurations are active.
    config['prod_mode'] = True
    update_all_embedding_models(config)

    # This overwrite is to use a currently supported audio model in prod_mode
    audio_model = config.get('audio_model')
    if isinstance(audio_model, dict):
        audio_model['model'] = 'whisper-1'

    # Update pages_to_show for prod mode
    config['pages_to_show'] = ['synthetic_eval']


def update_makefile(root_dir: str, port: int) -> None:
    """Update the STREAMLIT_PORT in the Makefile.

    Args:
        root_dir: The root directory where the Makefile is located.
        port: The port number to set for STREAMLIT_PORT.
    """
    makefile_path = os.path.join(root_dir, 'Makefile')
    if os.path.exists(makefile_path):
        with open(makefile_path, 'r', encoding='utf-8') as file:
            makefile_content = file.read()

        # Update the STREAMLIT_PORT variable in Makefile using regex
        # This provides flexibility when running multiple containers for production
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
    """Update configurations for all kits based on the specified mode.

    Args:
        mode: The mode to run in ('prod' or 'test'). Default is 'prod'.
        port: The port number to set for STREAMLIT_PORT. Default is 8501.
    """
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
            # Only runs embedding changes; used for running global tests
            update_all_embedding_models(config)

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
        choices=['prod', 'test'],  # 'test' is for running the global suite of tests for kits
        help='The mode to run in (prod or test). Default is prod.',
    )
    parser.add_argument('--port', type=int, default=8501, help='The port to use for Streamlit. Default is 8501.')

    args = parser.parse_args()

    update_configs(args.mode, args.port)


if __name__ == '__main__':
    main()
