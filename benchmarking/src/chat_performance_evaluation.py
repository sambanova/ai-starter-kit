import os
import ray
from llmperf.ray_clients.sambanova_client import SambaNovaLLMClient
from llmperf.models import RequestConfig
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

        prompt_template = (
            f"<INST> <SYSTEM> {DEFAULT_SYSTEM_PROMPT} </SYSTEM> {prompt} </INST>"
        )

        client = SambaNovaLLMClient.remote()
        request_config = RequestConfig(
            prompt=(prompt_template, 10),
            model=self.model,
            sampling_params=self.params,
            mode="stream",
            num_concurrent_requests=1,
        )
        output = ray.get(client.llm_request.remote(request_config))
        return output


if __name__ == "__main__":

    # load env variables
    load_dotenv("../../.env", override=True)
    env_vars = dict(os.environ)

    # init ray
    ray.init(local_mode=True, runtime_env={"env_vars": env_vars}, log_to_driver=True)

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
