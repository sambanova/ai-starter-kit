# import dependencies
import os                                                   # for using env variables
import streamlit as st                                      # for gui elements, secrets management
import json                                                 # for loading prompt example config file
import requests                                             # for calling web APIs
import base64                                               # for showing the SVG Sambanova icon
from typing import Tuple                                    # for type hint
from langchain.prompts import PromptTemplate, load_prompt   # for creating and loading prompting yaml files
from dotenv import load_dotenv                              # for loading env variables

# load env variables in this AISK
load_dotenv('../prompt_engineering.env')

# populate variables from secrets file
LLAMA2_7B_LLM_ENDPOINT = os.getenv("LLAMA2_7B_LLM_ENDPOINT") 
LLAMA2_7B_LLM_API_KEY = os.getenv("LLAMA2_7B_LLM_API_KEY") 

# define config path
CONFIG_PATH = '../config.json'


@st.cache_data
def call_sambanova_llama2_7b_api(prompt: str) -> str:
    """Calls a LLama2-7B Sambanova endpoint. Uses an input prompt and returns a completion of it.

    Args:
        prompt (str): prompt text

    Returns:
        str: completion of the input prompt
    """
    
    # Headers including Content-Type and Authorization with your API key
    headers = {
        'Content-Type': 'application/json',
        'key': LLAMA2_7B_LLM_API_KEY
    }

    # Data payload
    payload_data = {
        "inputs": [f"{prompt}"],
        "params": {                                             
#            "do_sample": {"type": "bool", "value": "false"},
            "max_tokens_to_generate": {"type": "int", "value": "200"},
#            "repetition_penalty": {"type": "float", "value": "1"},
#            "temperature": {"type": "float", "value": "1"},
#            "top_k": {"type": "int", "value": "50"},
#            "top_logprobs": {"type": "int", "value": "0"},
#            "top_p": {"type": "float", "value": "1"}
        }
    }

    # Convert the data payload to JSON
    json_data = json.dumps(payload_data)

    # Make the POST request
    response = requests.post(LLAMA2_7B_LLM_ENDPOINT, headers=headers, data=json_data)

    # Get text from response
    response_text = response.text

    # Split the response text into lines
    lines = response_text.split('\n')

    # Initialize a variable to hold the completion text
    completion_text = ""

    # Iterate through each line
    for line in lines:
        # Check if this line signifies the end event
        if 'event: end_event' in line:
            # The next line should contain the data with completion
            data_line_index = lines.index(line) + 1  # Get the index of the next line
            if data_line_index < len(lines):
                data_line = lines[data_line_index]
                # Extract the JSON part after 'data: '
                data_json_str = data_line.split('data: ', 1)[1] if 'data: ' in data_line else '{}'
                # Parse the JSON string
                data_json = json.loads(data_json_str)
                # Extract the 'completion' field from the JSON
                completion_text = data_json.get('completion', '')
                break  # Exit the loop as we've found the completion text

    return completion_text


def get_config_info() -> Tuple[dict, list]:
    """Loads json config file
    """
    
    # Read config file
    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        config = json.load(file)
    model_info = config["models"]
    prompt_use_cases = config["use_cases"]
    
    return model_info, prompt_use_cases

def get_prompt_template(model: str, prompt_use_case: str) -> str:
    """Reads a prompt template from an specified model and use case

    Args:
        model (str): model name 
        prompt_use_case (str): use case name

    Returns:
        str: prompt template associated to the model and use case selected
    """
    
    # Load prompt from the corresponding yaml file
    prompt_file_name = f"{model.lower().replace(' ','_')}-prompt_engineering-{prompt_use_case.lower().replace(' ','_')}_usecase.yaml"
    prompt = load_prompt(f'../prompts/{prompt_file_name}')
    
    return prompt.template
    

def render_svg(svg_path: str) -> None:
    """Renders the given svg string.

    Args:
        svg_path (str): SVG file path
    """
    
    # Render SVG file
    with open(svg_path, 'r') as file:
        svg = file.read()
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" width="60"/>' % b64
    st.write(html, unsafe_allow_html=True)


def create_prompt_yamls() -> None:
    """Shows a way how prompt yamls can be created. We're going to save our prompts in yaml files.
    """
    
    # Given a set of prompts based on the use case and model used
    prompts_templates = {
        "General Assistant": {
            "Llama2 7B": "[INST] <<SYS>> You are a helpful, respectful, positive, and honest assistant. Your answers should not include any unsafe, unethical, or illegal content. If you don't understand the question or don't know the answer, please don't share false information. <</SYS>>\n\nHow can I write better prompts for large language models? [/INST]"        
        },
        "Document Search": {
            "Llama2 7B": "[INST] <<SYS>> Use the following pieces of context to answer the question at the end. If the answer is not in context for answering, say that you don't know, don't try to make up an answer or provide an answer not extracted from provided context. <</SYS>>\nContext: Early Account Closure Fee $30 (if account is closed within 6 months of opening) \nReplacement of ATM Card $5 per card \nReplacement Of Lost Passbook $15 per passbook \nQuestion: I lost my ATM card. How much will it cost to replace it? [/INST]"
        },
        "Product Selection": {
            "Llama2 7B": "[INST] <<SYS>> You are an electrical engineer ai assistant.  You will be given context that will help answer a question.  You will interpret the context for useful information that will answer the question.  Provide concise, helpful, technical information (focusing on numerical information).  <</SYS>> [/INST] \n    Here is the question and context: \n\n    question: 'what is the switching frequency of max20710?'\n    contexts: VREF  Range | | See Table 8 for accuracy vs. �REF VREF (Note 6 ) | 0.6016 | | 1.0 | V | | FEEDBACK LOOP | | | | | | | | Integrator Recovery-Time Constant | tREC tREC  | | | 20 | | �s μs | | Gain (see the Control Loop Stability section for details) | RGAIN RGAIN  | Selected by R_SELB or PMBus ( ( Note 5,6,7,8) 5,6,7,8) | 0.72 | 0.9 | 1.1 | mV/A mV/A | | | | | 1.4 | 1.8 | 2.2 | | | | | | 2.9 | 3.6 | 4.3 | | | SWITCHING FREQUENCY | | | | | | | | Switching Frequency | ��� fSW  | 400kHz/600kHz/800kHz 400kHz/600kHz/800kHz selected by C_SELB; other values are set through PMBus (Note 6 ) | | 400 | | kHz kHz | | | | | | 500 | | | | | | | | 600 | | | | | | | | 700 | | | | | | | | 800 | | | | | | | | 900 | | | | Switching Frequency Accuracy | | ( ( Note 5,7,8) 5,7,8) | -20 | | +20 | % % | | INPUT PROTECTION | | | | | | | | Rising VDDH VDDH  UVLO Threshold | VDDH UVLO | (Note 5) | | 4.25 | 4.47 | V | | Falling VDDH VDDH  UVLO Threshold | | | 3.7 | 3.9 | | | | Hysteresis | | | | 350 | | mV mV |\n\n    [INST] Given the context you received, answer the question.  Please do not infer anything if the question cannot be deduced from the context or make anything up; simply state you cannot find the information. [/INST]\n\n    Sure, here is the answer to the question:"
        },
        "Code Generation": {
            "Llama2 7B": "[INST] <<SYS>> You are a helpful code assistant. Generate a valid JSON object based on the inpput. \nExample: name: John lastname: Smith address: #1 Samuel St. would be converted to:\n{\n\"\"address\"\": \"\"#1 Samuel St.\"\",\n\"\"lastname\"\": \"\"Smith\"\",\n\"\"name\"\": \"\"John\"\"\n} <</SYS>>\n[INST] Input: name: Ted lastname: Pot address: #1 Bisson St. [/INST]"
        }
    }
    
    # Iterate and save prompts in yaml files
    for usecase_key, usecase_value in prompts_templates.items():
        for model_key, prommpt_template in usecase_value.items():
            prompt_file_name = f"{model_key.lower().replace(' ','_')}-prompt_engineering-{usecase_key.lower().replace(' ','_')}_usecase"
            prompt = PromptTemplate.from_template(prommpt_template)
            prompt.save(f"../prompts/{prompt_file_name}.yaml")
            

def main():
    
    # Set up title
    st.set_page_config(page_title='Prompt Engineering - SambaNova Starter Kits',  layout="centered", initial_sidebar_state="auto", menu_items={'Get help': 'https://github.com/sambanova/ai-starter-kit/issues/new'})  #:mechanical-arm:, :toolbox:, :test-tube:, :play-button:, 
    col1, mid, col2 = st.columns([1,1,20])
    with col1:
        render_svg('../docs/sambanova-ai.svg')
    with col2:
        st.title('Prompt Engineering Starter Kit')

    # Get model information and prompt use cases from config file
    model_info, prompt_use_cases = get_config_info()
    model_names = [key for key, _ in model_info.items()]
    
    st.session_state["model_info"] = model_info
    
    # Set up model selection drop box
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_model = st.selectbox(
            "Model Selection",
            model_names,
            index=0,
            help='''
            \nNote: Working only with Llam2 7B for now.
            ''',
        )
        st.write(f":red[**Architecture:**] {st.session_state.model_info[selected_model]['Model Architecture']}  \n:red[**Prompting Tips:**] {st.session_state.model_info[selected_model]['Architecture Prompting Implications']}")

    # Set up use case drop box
    with col2:
        selected_prompt_use_case = st.radio(
            "Use Case for Sample Prompt",
            prompt_use_cases,
            help='''
            \n:red[**General Assistant:**] Provides comprehensive assistance on a wide range of topics, including answering questions, offering explanations, and giving advice. It's ideal for general knowledge, trivia, educational support, and everyday inquiries.
            \n:red[**Document Search:**] Specializes in locating and summarizing relevant information from large documents or databases. Useful for research, data analysis, and extracting key points from extensive text sources.
            \n:red[**Product Selection:**] Assists in choosing products by comparing features, prices, and reviews. Ideal for shopping decisions, product comparisons, and understanding the pros and cons of different items.
            \n:red[**Code Generation:**] Helps in writing, debugging, and explaining code. Useful for software development, learning programming languages, and automating simple tasks through scripting.
            ''',
        )
        st.write(f":red[**Meta Tag Format:**]  \n {st.session_state.model_info[selected_model]['Meta Tag Format']}")

    # Set up prompting area. Show prompt depending on the model selected and use case
    prompt_template = get_prompt_template(selected_model, selected_prompt_use_case)
    prompt = st.text_area(
        "Prompt",
        prompt_template,
        height=210,
    )
    
    # Process prompt and show the completion
    if st.button('Send'):
        response_content = ""
        # Call Llama2 endpoint and show the response content
        if selected_model == "Llama2 7B":
            response_content = call_sambanova_llama2_7b_api(prompt)
        st.write(response_content)

if __name__ == "__main__":
    # run following method if you want to know how prompt yaml files were created.
    # create_prompt_yamls()
    
    main()
    
    
    
    
    
    
    