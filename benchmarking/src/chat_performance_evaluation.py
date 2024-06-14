import os
import ray
from llmperf.ray_clients.sambanova_client import llm_request
from llmperf.models import RequestConfig
import llmperf.utils as utils
from dotenv import load_dotenv

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don'''t know the answer to a question, please don'''t share false information. Don'''t show any HTML tag in your answer."


class SambaStudioCOEHandler:
    """Samba Studio COE handler that wraps SamabaNova LLM client to parse output"""

    def __init__(self, model_name, params):
        self.model = model_name
        self.params = params

    def generate(self, prompt: str) -> tuple:
        """Generates LLM output in a tuple. It wraps SambaNova LLM client.

        Args:
            prompt (str): user's prompt

        Returns:
            tuple: contains the api response, generated text and input parameters
        """
        if utils.MODEL_TYPE_IDENTIFIER["llama3"] in self.model.lower():
            prompt_template = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{DEFAULT_SYSTEM_PROMPT}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
        else:
            prompt_template = f"[INST]{prompt}[/INST]"

        tokenizer = utils.get_tokenizer(self.model)

        request_config = RequestConfig(
            prompt=(prompt_template, 10),
            model=self.model,
            sampling_params=self.params,
            mode="stream",
            num_concurrent_requests=1,
        )
        output = ray.get(llm_request.remote(request_config, tokenizer))

        if output[0]["error_code"]:
            raise Exception(
                f"Unexpected error happened when executing requests: {output[0]['error_code']}. Additional message: {output[0]['error_msg']}"
            )
        return output


if __name__ == "__main__":

    # load env variables
    load_dotenv("../../.env", override=True)
    env_vars = dict(os.environ)

    # set log_to_driver = True if you'd like to have ray's logs in terminal
    ray.init(local_mode=True, runtime_env={"env_vars": env_vars}, log_to_driver=False)

    model_name = "COE/Meta-Llama-3-8B-Instruct"

    params = {
        # "do_sample": False,
        "max_tokens_to_generate": 1024,
        # "temperature": 1,
        # "repetition_penalty":1.0,
        # "top_k":50,
        # "top_p":0.95,
    }

    handler = SambaStudioCOEHandler(model_name, params)
    response = handler.generate(prompt="Tell me about SambaNova in one sentence")
    print(response)
