# import dependencies
import streamlit as st      # for gui elements
import json                 # for loading prompt example config file
import os #for json debug

# populate variables from secrets file
LLAMA27B_LLM_ENDPOINT = st.secrets["LLAMA27B_LLM_ENDPOINT"]
LLAMA27B_LLM_API_KEY = st.secrets["LLAMA27B_LLM_API_KEY"]



# temp debug json on cloud
st.write("Current directory:", os.getcwd())
st.write("Directory contents:", os.listdir('.'))
IS_STREAMLIT_CLOUD = os.getenv('USER') == 'appuser'
if IS_STREAMLIT_CLOUD:
    os.chdir(os.getcwd() = 'prompt_engineering')
    st.write("Current directory:", os.getcwd())
    st.write("Directory contents:", os.listdir('.'))


# load the json prompt example config file
json_file_path = 'model_prompt_data.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract options data from JSON file 
model_names = list(data["Model Architecture"].keys())   # model names
prompt_use_cases = [key for key in data.keys() if key != "Model Architecture" and key != "References" and key != "Architecture Prompting Implications" and key != "Heuristic Guidance" and key !=  "Meta Tag Format"] # Assuming they start from "Task-Agnostic Prompt" and are at the same level in the JSON structure

  
# Retrieve the task-agnostic prompt for the given model
#task_agnostic_prompt = data[prompt_type][model_name]


 

# functions will go here


## gui 
st.title('Prompt Engineering Starter Kit')

# get model type from user
selected_model = st.selectbox(
    "Model",
    model_names,
)

# get use case from user
selected_prompt_use_case = st.selectbox(
    "Use Case",
    prompt_use_cases,
)

prompt = st.text_area(
    "Prompt",
    data[selected_prompt_use_case][selected_model],

    height=280,
    )

if st.button('Submit Query'):
    response = st.write(
        "If you have a llama in your garden...",
        )


# evolutionary task prompt design automation
