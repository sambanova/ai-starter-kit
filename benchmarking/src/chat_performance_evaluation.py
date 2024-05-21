import ray
from llmperf.ray_clients.sambanova_client import SambaNovaLLMClient
from llmperf.models import RequestConfig

from dotenv import load_dotenv
load_dotenv('../../.env', override=True)

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
        client = SambaNovaLLMClient.remote()
        request_config = RequestConfig(
            prompt=(prompt, 10),
            model=self.model,
            sampling_params=self.params,
            mode="stream",
            num_concurrent_requests=1,
        )
        output = ray.get(client.llm_request.remote(request_config))
        return output
        
        
if __name__ == '__main__':
    
    ray.init(local_mode=True)
    
    model_name = "COE/Mistral-7B-Instruct-v0.2"

    params = {
        "do_sample": False,
        "max_tokens_to_generate": 1024,
        "temperature": 1,
        "repetition_penalty":1.0,
        "top_k":50,
        "top_p":0.95,
    }
    
    handler = SambaStudioCOEHandler(model_name, params)
    response = handler.generate(prompt='Tell me about SambaNova in one sentence')
    print(response)