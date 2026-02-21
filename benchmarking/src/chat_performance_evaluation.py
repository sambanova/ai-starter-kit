import base64
import os
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

import benchmarking.benchmarking_utils as benchmarking_utils
from benchmarking.src.llmperf.models import RequestConfig
from benchmarking.src.llmperf.sambanova_client import llm_request
from benchmarking.streamlit.streamlit_utils import set_api_variables


class ChatPerformanceEvaluator:
    """Handler that wraps SamabaNova LLM client to parse output"""

    def __init__(self, model_name: str, llm_api: str, image_path: str, params: Optional[Dict[str, Any]]) -> None:
        self.model = model_name
        self.llm_api = llm_api
        self.image_path = image_path
        self.params = params

    def generate(self, prompt: str) -> Tuple[Dict[str, Any], str, RequestConfig]:
        """Generates LLM output in a tuple. It wraps SambaNova LLM client.

        Args:
            prompt (str): user's prompt

        Returns:
            tuple: contains the api response, generated text and input parameters
        """
        family_model_type = benchmarking_utils.find_family_model_type(self.model)

        if family_model_type == 'llama3':
            prompt_template = f"""<|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>"""
        else:
            prompt_template = f'[INST]{prompt}[/INST]'

        # Build prompt dict for Request Config
        prompt_dict = {'name': 'chat_prompt', 'template': prompt_template}

        tokenizer = benchmarking_utils.get_tokenizer(self.model)

        api_variables = set_api_variables()

        # Image to be sent in LLM request if exists
        image = None
        if self.image_path:
            with open(self.image_path, 'rb') as image_file:
                image = base64.b64encode(image_file.read()).decode('utf-8')

        request_config = RequestConfig(
            request_idx=1,
            prompt_tuple=(prompt_dict, 0),
            image=image,
            model=self.model,
            llm_api=self.llm_api,
            api_variables=api_variables,
            sampling_params=self.params,
            is_stream_mode=True,
            num_concurrent_workers=1,  # type: ignore
        )
        output = llm_request(request_config, tokenizer)

        if output[0]['error_code']:
            nl = '\n'
            raise Exception(
                f"""Unexpected error happened when executing requests:\
                {nl}{nl}- {output[0]['error_code']}\
                {nl}{nl}Additional messages:{nl}- {output[0]['error_msg']}"""
            )
        return output


if __name__ == '__main__':
    # load env variables
    load_dotenv('../.env', override=True)
    env_vars = dict(os.environ)

    model_name = 'Meta-Llama-3.3-70B-Instruct'
    llm_api = 'sncloud'

    params = {
        # "do_sample": False,
        'max_tokens_to_generate': 1024,
        # "temperature": 1,
        # "repetition_penalty":1.0,
        # "top_k":50,
        # "top_p":0.95,
    }

    handler = ChatPerformanceEvaluator(model_name, llm_api, image_path='', params=params)
    response = handler.generate(prompt='Tell me about SambaNova in one sentence')
    print(response)
