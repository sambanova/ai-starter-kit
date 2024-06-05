import os
import yaml
from dotenv import load_dotenv
from langchain_community.llms import SambaStudio, Sambaverse
from langchain_core.prompts import load_prompt
from typing import Dict, List, Union, Optional

guardrails_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(guardrails_dir, ".." ))
repo_dir = os.path.abspath(os.path.join(utils_dir, ".." ))

load_dotenv(os.path.join(repo_dir, ".env") )

class Guard():
            
    def __init__(
        self,
        api: str = "sambaverse",
        prompt_path: Optional[str] = None,
        guardrails_path: Optional[str] = None, 
        sambaverse_base_url: Optional[str] = None,
        sambaverse_api_key: Optional[str] = None,
        sambastudio_base_url: Optional[str] = None,
        sambastudio_project_id: Optional[str] = None,
        sambastudio_endpoint_id: Optional[str] = None,
        sambastudio_api_key: Optional[str] = None,
        ):
        
        if prompt_path is None:
            prompt_path = os.path.join(guardrails_dir, "prompt.yaml")
        self.prompt = load_prompt(prompt_path)
        if guardrails_path is None:
            guardrails_path = os.path.join(guardrails_dir, "guardrails.yaml")        
        self.guardrails ,self.parsed_guardrails = self.load_guardrails(guardrails_path)
        params = {}
        if api == "sambastudio":
            if sambastudio_base_url:
                params["sambastudio_base_url"] = sambastudio_base_url
            if sambastudio_project_id:
                params["sambastudio_project_id"] = sambastudio_project_id
            if sambastudio_endpoint_id:
                params["sambastudio_endpoint_id"] = sambastudio_endpoint_id
            if sambastudio_api_key:
                params["sambastudio_api_key"] = sambastudio_api_key
            self.llm = self.set_llm("sambastudio", params)
        elif api == "sambaverse":
            params = {}
            if sambaverse_base_url:
                params["sambaverse_base_url"] = sambaverse_base_url
            if sambaverse_api_key:
                params["sambaverse_api_key"] = sambaverse_api_key
            self.llm = self.set_llm("sambaverse", params)
    
    def load_guardrails(self, path: str):
        with open(path, 'r') as yaml_file:
            guardrails = yaml.safe_load(yaml_file)
        enabled_guardrails = {k: v for k, v in guardrails.items() if v.get('enabled')}
        guardrails_list = [f"{k}: {v.get('name')}\n{v.get('description')}" for k, v in enabled_guardrails.items()]
        guardrails_str = "\n".join(guardrails_list)
        return guardrails, guardrails_str
    
    def set_llm(self, api: str, params: Optional[Dict]):
        if api == "sambastudio":
            llm = SambaStudio(
                **params,
                model_kwargs={
                "select_expert": "Meta-Llama-Guard-2-8B",
                "process_prompt": False,
                "do_sample": False,
                "max_tokens_to_generate": 1024,
                "temperature": 0.1,
                },
            )
        elif api == "sambaverse":
            llm = Sambaverse(
                **params,
                sambaverse_model_name="Meta/Meta-Llama-Guard-2-8B",
                model_kwargs={
                    "select_expert": "Meta-Llama-Guard-2-8B",
                    "process_prompt": True,
                    "do_sample": False,
                    "max_tokens_to_generate": 1024,
                    "temperature": 0.1,
                },
            )
        return llm

    def evaluate(
        self,
        input: Union[List[Dict], str],
        role: str,
        error_message: str = None,
        return_guardrail_type: bool = True,
        raise_exception: bool =False
        ):
        
        if isinstance(input, str):
            if role.lower() == "user":
                conversation = f"User: {input}"
            elif role.lower() == "assistant":
                conversation = f"Assistant: {input}"
            else:
                raise ValueError(f"Invalid role: {role}, only User and Assistant")
        elif isinstance(input, List):
            # example conversation input
            # [
            # {"message_id":0,"role":"user", "content":"this is an user message"},
            # {"message_id":1,"role":"assistant","content":"this is an assistant response"},
            #]
            conversation = ""
            for message in input:
                if message["role"].lower() == "user":
                    conversation += f"User: {message['content']}\n"
                elif message["role"].lower() == "assistant":
                    conversation += f"Assistant: {message['content']}\n"
        values = {"conversation": input, "guardrails": self.parsed_guardrails, "role": conversation}
        formatted_input =  self.prompt.format(**values)
        result = self.llm.invoke(formatted_input)
        if "unsafe" in result:
            violated_categories = result.split("\n")[-1].split(",")
            violated_categories = [f'{k}: {v.get("name")}' for k, v in self.guardrails.items() if k in violated_categories]
            if error_message is None:
                error_message = f"The message violate guardrails"
            response_msg = f'{error_message}\nViolated categories: {", ".join(violated_categories)}'
            if raise_exception:
                raise ValueError(response_msg)
            else:
                if return_guardrail_type:
                    return response_msg
                else:
                    return error_message 
        else:
            return input 
            