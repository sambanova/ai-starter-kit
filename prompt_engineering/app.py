# import dependencies
import streamlit as st      # for gui elements, secrets management
import json                 # for loading prompt example config file
import os                   # for validating run/working directory to allow local and cloud execution
import requests             # for calling web APIs

# populate variables from secrets file
LLAMA27B_LLM_ENDPOINT = st.secrets["LLAMA27B_LLM_ENDPOINT"]
LLAMA27B_LLM_API_KEY = st.secrets["LLAMA27B_LLM_API_KEY"]
OPENAI_API_ENDPOINT = 'https://api.openai.com/v1/chat/completions'
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# if streamlit cloud, change down to prompt_engineering working directory (streamlit cloud runs from repo root always)
if os.getcwd() == "/mount/src/ai-starter-kit":
    path = os.getcwd()
    path += '/prompt_engineering'
    os.chdir(path)
# note: if streamlit cloud, consider mixpanel event
# note: if streamlit cloud, confirm externally facing endpoints


# load the json prompt example config file
json_file_path = 'model_prompt_data.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    model_prompt_data = json.load(file)

# Extract options data from JSON file 
model_names = list(model_prompt_data["Model Architecture"].keys())   # model names
prompt_use_cases = [key for key in model_prompt_data.keys() if key != "Model Architecture" and key != "References" and key != "Architecture Prompting Implications" and key != "Heuristic Guidance" and key !=  "Meta Tag Format"] # Assuming they start from "Task-Agnostic Prompt" and are at the same level in the JSON structure



@st.cache_data
def query_model(prompt):
    # Headers including Content-Type and Authorization with your API key
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }

    # Data payload
    payload_data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # Convert the data payload to JSON
    json_data = json.dumps(payload_data)

    # Make the POST request
    response = requests.post(OPENAI_API_ENDPOINT, headers=headers, data=json_data)

    # Load as json
    response_json = response.json() 

    # Extract the LLM reply
    response_content = response_json["choices"][0]["message"]["content"]
    return response_content










## gui 
st.title('Prompt Engineering Starter Kit')

# get model type user selection
selected_model = st.selectbox(
    "Select Model",
    model_names,
)

st.write(f"**Model Architecture --** {model_prompt_data['Model Architecture'][selected_model]}")
st.write(f"**How to Prompt --** {model_prompt_data['Architecture Prompting Implications'][selected_model]}")


# get use case user selection
selected_prompt_use_case = st.selectbox(
    "Use Case",
    prompt_use_cases,
)


prompt = st.text_area(
    "Prompt",
    model_prompt_data[selected_prompt_use_case][selected_model],
    height=280,
    )

if st.button('Send'):

    response_content = query_model(prompt)

    # Print the response
    st.write(response_content)




# evolutionary task prompt design automation
