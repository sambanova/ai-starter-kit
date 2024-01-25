# import dependencies

import streamlit as st          # for gui elements, secrets management
import json                     # for loading prompt example config file
import os                       # for validating run/working directory to allow local and cloud execution
import requests                 # for calling web APIs
from mixpanel import Mixpanel   # to allow gathering of feedback
from streamlit.web.server.websocket_headers import _get_websocket_headers # to allow reporting session id to mixpanel


# populate variables from secrets file
LLAMA27B_LLM_ENDPOINT = st.secrets["LLAMA27B_LLM_ENDPOINT"]
LLAMA27B_LLM_API_KEY = st.secrets["LLAMA27B_LLM_API_KEY"]
OPENAI_API_ENDPOINT = 'https://api.openai.com/v1/chat/completions'
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MIXPANEL_TOKEN = st.secrets["MIXPANEL_TOKEN"]

# initialize mixpanel object for feedback gathering
mixpanel = Mixpanel(MIXPANEL_TOKEN)

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
def query_model_oa(prompt):
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



# use langchain or generalize model, key, header format, payload format, 
@st.cache_data
def query_model_sn(prompt):
    # Headers including Content-Type and Authorization with your API key
    headers = {
        'Content-Type': 'application/json',
        'key': LLAMA27B_LLM_API_KEY
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
    response = requests.post(LLAMA27B_LLM_ENDPOINT, headers=headers, data=json_data)

    # Get text from response
    response_text = response.text

    # Split the response text into lines
    lines = response_text.split('\n')

    # Initialize a variable to hold the completion text
    completion_text = ""

    # Iterate over each line
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

 #   st.write(response.text)


 #   return response



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
    "Use Case Example",
    prompt_use_cases,
)


prompt = st.text_area(
    "Prompt",
    model_prompt_data[selected_prompt_use_case][selected_model],
    height=280,
    )

if st.button('Send'):

    response_content = query_model_sn(prompt)

    # Print the response
    st.write(response_content)



with st.expander("Provide Feedback"):
    goals_feedback = st.radio(
    "What are you trying to accomplish by using this AI Starter Kit?",
    ["Build a PoC", "General learning", "Something else", "No Answer"],
    index=3,
    horizontal=True
    )

    helpfulness_feedback = st.radio(
    "Is this AI Starter Kit helping you accomplish your goal?", 
    ["No", "Somewhat", "Yes", "No Answer"],
    index=3,
    horizontal=True
    )

    freetext_feedback = st.text_input("What should we start, stop, or continue doing?")
    user_email = st.text_input("What's your email address?",f"{st.experimental_user.email}")

    #nps_feedback = st.radio(
    #"On a scale of 0 to 10, how likely would you be to recommend someone use AI Starter Kits?",
    #["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    #horizontal=True
    #)


    if st.button('Submit Feedback'):

        headers = _get_websocket_headers()
        session_id = "Unknown"
        session_id = headers.get("Sec-Websocket-Key")

        mixpanel.track(f"AISK: {user_email}", 'TEST_AISK:FEEDBACK_SUBMITTED',  {
        'Feedback: User Goal': f'{goals_feedback}',
        'Feedback: Are We Helping': f'{helpfulness_feedback}',
        'Feedback: Freetext': f'{freetext_feedback}',
        'Feedback: Email': f'{user_email}',
        'Feedback: Session ID': f'{session_id}',
        })
        
        # Print the response
        st.write("Thank you, your feedback is a big deal to us!")


# evolutionary task prompt design automation

# add feedmack to more starter kits - AISK mixpanel
    
# beginner's mind, declutter with config file? how to make what they need to lear fit on a page

# use cases as radio buttons? as it grow
        

