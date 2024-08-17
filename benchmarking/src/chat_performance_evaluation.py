import os

from llmperf.sambanova_client import llm_request
from llmperf.models import RequestConfig
import llmperf.utils as utils

from dotenv import load_dotenv

class ChatPerformanceEvaluator:
    """Samba Studio COE handler that wraps SamabaNova LLM client to parse output"""

    def __init__(self, model_name, llm_api, params):
        self.model = model_name
        self.llm_api = llm_api
        self.params = params

    def generate(self, prompt: str) -> tuple:
        """Generates LLM output in a tuple. It wraps SambaNova LLM client.

        Args:
            prompt (str): user's prompt

        Returns:
            tuple: contains the api response, generated text and input parameters
        """
        if utils.MODEL_TYPE_IDENTIFIER["llama3"] in self.model.lower().replace("-",""):
            prompt_template = (
                f"<|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
        else:
            prompt_template = f"[INST]{prompt}[/INST]"

        tokenizer = utils.get_tokenizer(self.model)

        request_config = RequestConfig(
            prompt_tuple=(prompt_template, 10),
            model=self.model,
            llm_api=self.llm_api,
            sampling_params=self.params,
            is_stream_mode=True,
            num_concurrent_requests=1,
        )
        output = llm_request(request_config, tokenizer)

        if output[0]["error_code"]:
            raise Exception(
                f"Unexpected error happened when executing requests: {output[0]['error_code']}. Additional message: {output[0]['error_msg']}"
            )
        return output


if __name__ == "__main__":

    # load env variables
    load_dotenv("../.env", override=True)
    env_vars = dict(os.environ)

    model_name = "COE/Meta-Llama-3-8B-Instruct"
    llm_api = "sambastudio"

    params = {
        # "do_sample": False,
        "max_tokens_to_generate": 1024,
        # "temperature": 1,
        # "repetition_penalty":1.0,
        # "top_k":50,
        # "top_p":0.95,
    }

    handler = ChatPerformanceEvaluator(model_name, llm_api, params)
    response = handler.generate(prompt="Tell me about SambaNova in one sentence")
    print(response)
