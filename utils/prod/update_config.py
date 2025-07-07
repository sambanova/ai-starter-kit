import argparse
import os
import re
from typing import Any, Dict, List

import ruamel.yaml
from ruamel.yaml.scalarstring import DoubleQuotedScalarString, FoldedScalarString

# List of starter kits for production updates
PROD_KITS: List[str] = [
    'benchmarking',
    'enterprise_knowledge_retriever',
    'financial_assistant',
    'function_calling',
    'search_assistant',
    'multimodal_knowledge_retriever',
    'sambanova_scribe',
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

def update_tools(config: Dict[str, Any]) -> None:
    """Update 'st_tools' configurations in the config dictionary for prod mode.

    Args:
        config: The configuration dictionary to update.
    """
    st_tools = config.get('st_tools')
    if isinstance(st_tools, dict):
        python_repl = st_tools.get('python_repl')
        if isinstance(python_repl, dict):
            python_repl['enabled'] = False
            python_repl['default'] = False
        calculator = st_tools.get('calculator')
        if isinstance(calculator, dict):
            calculator['default'] = True


def update_config_prod(config: Dict[str, Any]) -> None:
    """Update configuration for production mode.

    Args:
        config: The configuration dictionary to update.
    """
    # Enforcing 'prod_mode' ensures that production-specific optimizations
    # and configurations are active.
    config['prod_mode'] = True
    update_tools(config)

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


def update_preset_queries(kit_path: str) -> None:
    """Update the prompts/streamlit_preset_queries.yaml file for function_calling kit.

    Args:
        kit_path: The path to the kit directory.
    """
    prompts_file_path = os.path.join(kit_path, 'prompts', 'streamlit_preset_queries.yaml')
    if os.path.exists(prompts_file_path):
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = True
        with open(prompts_file_path, 'r', encoding='utf-8') as file:
            prompts = yaml.load(file)

        # Modify the prompts
        old_key = 'Create insightful plots of the summary table'
        new_key = 'Check the price of a track and do a currency conversion'
        if old_key in prompts:
            # Remove the old key-value pair
            prompts.pop(old_key)

            # Create the new key as a double-quoted scalar
            new_key_scalar = DoubleQuotedScalarString(new_key)

            # Prepare the escaped value
            escaped_string = (
                'What is the price in colombian pesos of the track \\"Snowballed\\" in the db '
                'if one usd is equal to 3800 cop?'
            )

            # Create the new value as a folded scalar string
            new_value = FoldedScalarString(escaped_string)

            # Add the new key-value pair
            prompts[new_key_scalar] = new_value

            with open(prompts_file_path, 'w', encoding='utf-8') as file:
                yaml.dump(prompts, file)
            print(f'Updated prompts in {prompts_file_path}')
        else:
            print(f'Key "{old_key}" not found in prompts for {kit_path}')
    else:
        print(f'Prompts file not found at {prompts_file_path}')


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

        yaml = ruamel.yaml.YAML()
        with open(config_path, 'r', encoding='utf-8') as file:
            config: Dict[str, Any] = yaml.load(file)

        if mode == 'prod':
            update_config_prod(config)

        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file)

        print(f'Updated config for {kit_name} in {mode} mode.')

        # Custom change for function_calling kit
        if mode == 'prod' and kit_name == 'function_calling':
            update_preset_queries(kit_path)

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
