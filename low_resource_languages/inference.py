import os
from dotenv import load_dotenv
load_dotenv('.env')

import argparse
import yaml

from utils.sambanova_endpoint import SambaNovaEndpoint

llm = SambaNovaEndpoint(
    base_url=os.getenv('BASE_URL'),
    project_id=os.getenv('PROJECT_ID'),
    endpoint_id=os.getenv('ENDPOINT_ID'),
    api_key=os.getenv('API_KEY'),
    model_kwargs={
        "do_sample": False,
        "temperature": 0.0,
        "max_tokens_to_generate": 250
    },
)

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='config.yaml', type=str, help='Path to configuration file',
                    required=True)
args = parser.parse_args()
with open(args.config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    CUSTOM_PROMPT_TEMPLATE = config['CUSTOM_PROMPT_TEMPLATE']

def translate(sentence):
    return llm(CUSTOM_PROMPT_TEMPLATE + sentence + '=')

if __name__ == '__main__':
    print(translate('The cat sat on the mat'))