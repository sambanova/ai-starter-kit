import json
import random
import re
from transformers import AutoTokenizer

class Prompt():
    def __init__(self, json_file):
        self.json_file = json_file

        with open(json_file, 'r') as file_reader:
            json_list = json.load(file_reader)

        self.prompt_list = json_list
        self.num_prompts = len(json_list)

        return

# // escape shell code
# export function shellEscape(a: string[]): string { <-- input list of strings
#   const ret: string[] = []; 

#   a.forEach((s) => {
#     if (!/^[A-Za-z0-9_/-]+$/.test(s)) { <-- Ignore if no special characters
#       s = "'" + s.replace(/'/g, "'\\''") + "'";
#       s = s.replace(/^(?:'')+/g, '')
#         .replace(/\\'''/g, "\\'" );
#     }
#     ret.push(s);
#   });

#   return ret.join(' '); <-- output string
# }

    def scrub_prompt(self, prompt: str):
        return prompt.replace('\\', '\\\\').replace('\n','\\n').replace("'","\'").replace('"', '\\"').replace('\t', '\\t')

    def scrub_prompt_new_stashing(self, prompt: str):
        prompt_list = prompt.split()
        scrubbed_prompt = []
        for prompt_string in prompt_list:
            new_string = prompt_string
            if(not re.match(r'^[A-Za-z0-9_/-]+$', prompt_string)):
                replace_string = f"'\\\\''"
                replace_string_2 = r"'"
                new_string = f"'{re.sub(replace_string_2, replace_string, prompt_string)}'"
                new_string = re.sub(r"\\\\'''", r"\\\\'",re.sub(r'^(?:'')+', "", new_string))

            scrubbed_prompt.append(new_string)

        return str.join(' ',scrubbed_prompt)
        
    def get_random_user_prompt(self, randomIndex=None):
        if not randomIndex:
            index = random.Random().randint(0,self.num_prompts-1)
        else:
            index = randomIndex
        return self.scrub_prompt(self.prompt_list[index])

    def create_llama2_prompt(self, user_prompt):
        return f"<s>[INST]<<SYS>>\\nYou are a helpful ai assistant. Please answer the following question.\\n<</SYS>>\\n\\n{user_prompt}[/INST]"

    def random_llama2_prompt(self, user_prompt):
        # user_prompt = self.get_random_user_prompt()
        return self.create_llama2_prompt(user_prompt)
    
    def create_llama3_prompt(self, user_prompt):
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful ai assistant. Please answer the following question.<|eot_id|><|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    def random_llama3_prompt(self, user_prompt):
        # user_prompt = self.get_random_user_prompt()
        return self.create_llama3_prompt(user_prompt)
    
    def create_mistral_prompt(self, user_prompt):
        return f"[INST]{user_prompt}[/INST]"

    def random_mistral_prompt(self, user_prompt):
        # user_prompt = self.get_random_user_prompt()
        return self.create_mistral_prompt(user_prompt)
    
    def create_deepseek_prompt(self, user_prompt):
        return f"### Instruction:{user_prompt}### Response:"

    def random_deepseek_prompt(self, user_prompt):
        # user_prompt = self.get_random_user_prompt()
        return self.create_deepseek_prompt(user_prompt)
    
    def create_solar_prompt(self, user_prompt):
        return f"### User:\\n{user_prompt}\\n\\n### Assistant:"

    def random_solar_prompt(self, user_prompt):
        # user_prompt = self.get_random_user_prompt()
        return self.create_solar_prompt(user_prompt)
    
    def create_eeve_prompt(self, user_prompt):
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\\nHuman: {user_prompt}\\nAssistant:"

    def random_eeve_prompt(self, user_prompt):
        # user_prompt = self.get_random_user_prompt()
        return self.create_eeve_prompt(user_prompt)
    
    def create_prompt(self, user_prompt, prompt_template):
        return prompt_template.replace('{user_prompt}', user_prompt)

    def get_tokenizer(self, model_name: str, prompt_template, template, input_tokens_num) -> str:
        """Gets generic tokenizer according to model type
        Args:
            model_name (str): model name
        Returns:
            str: Adjusted text with the prompt_template applied
        """    
        # Selecting the appropriate tokenizer based on the model name
        if "mistral" in model_name.lower().replace("-", "").replace(" ", ""):
            tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
        elif "llama3" in model_name.lower().replace("-", "").replace(" ", ""):
            tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
        elif "deepseek" in model_name.lower().replace("-", "").replace(" ", ""):
            if "coder" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/deepseek-coder-1.3b-base"
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/deepseek-llm-7b-base"
                )
        elif "solar" in model_name.lower().replace("-", "").replace(" ", ""):
            tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0")
        elif "eeve" in model_name.lower().replace("-", "").replace(" ", ""):
            tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-10.8B-v1.0")
        else:
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

        # Get the number of tokens in the prompt and prompt_template
        # prompt_tokens = tokenizer.tokenize(prompt)
        prompt_template_tokens = tokenizer.tokenize(prompt_template)

        # Calculate the maximum number of tokens available for the template
        max_template_tokens = input_tokens_num - len(prompt_template_tokens) + 5

        # Tokenize and truncate the template to fit within the available token limit
        template_tokens = tokenizer.tokenize(template, truncation=True, max_length=max_template_tokens)

        # Convert the truncated template tokens back to a string
        truncated_template = tokenizer.convert_tokens_to_string(template_tokens)

        # Replace '{user_prompt}' in the prompt_template with the truncated template
        adjusted_text = prompt_template.replace('{user_prompt}', truncated_template)

        return adjusted_text
    
    def get_tokenizer_old(self, model_name: str,  prompt, input_tokens_num) -> AutoTokenizer:
        """Gets generic tokenizer according to model type
        Args:
            model_name (str): model name
        Returns:
            AutoTokenizer: generic HuggingFace tokenizer
        """    
        # Using NousrResearch for calling out model tokenizers without requesting access. 
        # Ref: https://huggingface.co/NousResearch
        # Ref: https://huggingface.co/TheBloke
        # Ref: https://huggingface.co/unsloth
        # Ref: https://huggingface.co/deepseek-ai
        # Ref: https://huggingface.co/upstage
        # Ref: https://huggingface.co/yanolja
        
        if "mistral" in model_name.lower().replace("-","").replace(" ", ""):
            tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
        elif "llama3" in model_name.lower().replace("-","").replace(" ", ""):
            tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
        elif "deepseek" in model_name.lower().replace("-","").replace(" ", ""):
            if "coder" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/deepseek-coder-1.3b-base"
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/deepseek-llm-7b-base"
                )
        elif "solar" in model_name.lower().replace("-","").replace(" ", ""):
            tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0")
        elif "eeve" in model_name.lower().replace("-","").replace(" ", ""):
            tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-10.8B-v1.0")
        else:
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

        tokenizer.padding_side = "right"
        if(not tokenizer.pad_token):
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokens = tokenizer.tokenize(prompt, padding="max_length", max_length=input_tokens_num-1, truncation=True)

        # Convert tokens back to text
        adjusted_text = tokenizer.convert_tokens_to_string(tokens)

        adjusted_tokens = tokenizer.tokenize(adjusted_text, padding="max_length", max_length=input_tokens_num-1, truncation=True)

        assert len(adjusted_tokens) == (input_tokens_num - 1), "Token count mismatch!"
        
        return adjusted_text