"""
Configuration module for the educational content generation system.

This module defines all configuration settings, including available models,
provider configurations, and default settings for the content generation system.
It centralizes configuration management for the entire application.
"""

import os
from typing import Any, Dict, List, Optional

# Available SambaNova models configuration
SAMBANOVA_MODELS: List[str] = [
    'Meta-Llama-3.1-70B-Instruct',
    'Meta-Llama-3.3-70B-Instruct',
    'DeepSeek-R1',
    'DeepSeek-R1-Distill-Llama-70B',
    'Llama-3.1-Tulu-3-405B',
    'Meta-Llama-3.1-8B-Instruct',
    'Meta-Llama-3.1-405B-Instruct',
    'Meta-Llama-3.2-1B-Instruct',
    'Meta-Llama-3.2-3B-Instruct',
    'Qwen2.5-Coder-32B-Insruct',
    'Qwen2.5-72B-Instruct',
    'LLama-3.2-11B-Vision-Instruct',
    'LLama-3.2-90B-Vision-Instruct',
]

# Provider-specific model configurations
AVAILABLE_MODELS: Dict[str, List[str]] = {
    'sambanova': SAMBANOVA_MODELS,
}

# Detailed provider configurations
PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    'sambanova': {
        'model_prefix': 'sambanova/',
        'api_key_env': 'SAMBANOVA_API_KEY',
        'display_name': 'SambaNova',
        'base_url': 'https://api.sambanova.ai/v1',
        'models': SAMBANOVA_MODELS,
    },
}

# Default configuration settings
DEFAULT_PROVIDER: str = 'sambanova'
DEFAULT_MODEL: str = PROVIDER_CONFIGS[DEFAULT_PROVIDER]['models'][6]

# LLM configuration settings
LLM_CONFIGS: Dict[str, Dict[str, Optional[str]]] = {
    provider: {
        'model': f"{config['model_prefix']}{config['models'][6]}",
        'api_key': os.getenv(config['api_key_env']),
        'base_url': config['base_url'],
    }
    for provider, config in PROVIDER_CONFIGS.items()
}

# Default LLM configuration
LLM_CONFIG: Dict[str, Optional[str]] = LLM_CONFIGS[DEFAULT_PROVIDER]

# Educational flow input variables
EDU_FLOW_INPUT_VARIABLES: Dict[str, str] = {
    'audience_level': 'intermediate',
    'topic': 'Speculative Decoding',
}
