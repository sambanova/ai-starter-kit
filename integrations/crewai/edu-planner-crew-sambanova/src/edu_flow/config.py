import os

# Existing SambaNova models
SAMBANOVA_MODELS = [
    'Meta-Llama-3.1-70B-Instruct',
    'Meta-Llama-3.1-8B-Instruct',
    'Meta-Llama-3.1-405B-Instruct',
    'Meta-Llama-3.2-1B-Instruct',
    'Meta-Llama-3.2-3B-Instruct',
    'Qwen2.5-Coder-32B-Insruct',
    'Qwen2.5-72B-Instruct',
    'LLama-3.2-11B-Vision-Instruct',
    'LLama-3.2-90B-Vision-Instruct',
]

# Combined available models with provider prefixes
AVAILABLE_MODELS = {
    'sambanova': SAMBANOVA_MODELS,
}

# Provider configurations
PROVIDER_CONFIGS = {
    'sambanova': {
        'model_prefix': 'sambanova/',
        'api_key_env': 'SAMBANOVA_API_KEY',
        'display_name': 'SambaNova',
        'base_url': 'https://api.sambanova.ai/v1',
        'models': SAMBANOVA_MODELS,
    },
}

# Default configurations
DEFAULT_PROVIDER = 'sambanova'
DEFAULT_MODEL = PROVIDER_CONFIGS[DEFAULT_PROVIDER]['models'][0]

# LLM configurations - generated from PROVIDER_CONFIGS
LLM_CONFIGS = {
    provider: {
        'model': f"{config['model_prefix']}{config['models'][6]}",
        'api_key': os.getenv(config['api_key_env']),
        'base_url': config['base_url'],
    }
    for provider, config in PROVIDER_CONFIGS.items()
}

# Default LLM config
LLM_CONFIG = LLM_CONFIGS[DEFAULT_PROVIDER]

EDU_FLOW_INPUT_VARIABLES = {
    'audience_level': 'intermediate',
    'topic': 'Speculative Decoding',
}
