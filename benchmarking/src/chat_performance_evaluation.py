import os
import json 
import requests 

from dotenv import load_dotenv
load_dotenv('../.env', override=True)

URL = "https://sambaverse.sambanova.ai/api/predict"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\\\\n\\\\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don'\''t know the answer to a question, please don'\''t share false information."

class SambaVerseHandlerTmp:
    
    def __init__(self, model_name, params):
        self.url = URL
        self.key = os.getenv('SAMBAVERSE_API_KEY')
        self.headers = {
            'Content-Type': 'application/json',
            'key': self.key,
            'modelName': model_name
        }
        self.params = params

    def _get_data_structure(self, prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
        data = {
            "instance": prompt,
            "params": self.params
        }
        
        return data
        
    def generate(self, prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
        data = self._get_data_structure(prompt, system_prompt)
        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        if response.status_code == 200:
            successful_response = ''
            for resp in response.text.split('\n'):
                response_dict = json.loads(resp)
                complete = response_dict['result']['status']['complete']
                if complete == True:
                    successful_response = response_dict
                    break
            return successful_response
        else:
            print(response.text)
            raise Exception("Error in generate")
        
if __name__ == '__main__':
    
    model_name = 'Mistral/Mistral-7B-Instruct-v0.2'

    params = {
        "do_sample": {"type": "bool", "value": "false"},
        "max_tokens_to_generate": {"type": "int", "value":"1024"},
        "temperature": {"type": "float", "value": "1"},
        "repetition_penalty":{"type":"float","value":"1.0"},
        "top_k":{"type":"int","value":"50"},
        "top_p":{"type":"float","value":"0.95"},
        
        "process_prompt":{"type":"bool", "value":"false"},
        "select_expert":{"type":"str", "value":f"{model_name.split('/')[-1]}"},
    }
    
    sambaverse = SambaVerseHandlerTmp(model_name, params)
    response = sambaverse.generate(prompt='Tell me about SambaNova in one sentence')
    print(response)