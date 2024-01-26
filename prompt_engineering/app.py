# import dependencies

import streamlit as st          # for gui elements, secrets management
import json                     # for loading prompt example config file
import os                       # for validating run/working directory to allow local and cloud execution
import requests                 # for calling web APIs
from mixpanel import Mixpanel   # to allow gathering of feedback
from streamlit.web.server.websocket_headers import _get_websocket_headers # to allow reporting session id to mixpanel
import replicate                # to allow calling non-sn endpoints


# populate variables from secrets file
LLAMA27B_LLM_ENDPOINT = st.secrets["LLAMA27B_LLM_ENDPOINT"]
LLAMA27B_LLM_API_KEY = st.secrets["LLAMA27B_LLM_API_KEY"]
# for deployment mixpanel will require adjustment, inclusion in secrets sample
MIXPANEL_TOKEN = st.secrets["MIXPANEL_TOKEN"]
# for deployment these debugging endpoints will be removed, as will the secrets references, as will some includes and requirements
OPENAI_API_ENDPOINT = 'https://api.openai.com/v1/chat/completions'
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]


# initialize mixpanel object for feedback gathering
mixpanel = Mixpanel(MIXPANEL_TOKEN)

# if streamlit cloud, change down to prompt_engineering working directory (streamlit cloud runs from repo root always)
path = os.getcwd()
if path == "/mount/src/ai-starter-kit":
    path += '/prompt_engineering'
    os.chdir(path)

# load the json prompt example config file
json_file_path = 'model_prompt_data.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    model_prompt_data = json.load(file)

# Extract options data from JSON file 
model_names = list(model_prompt_data["Model Architecture"].keys())   # model names
prompt_use_cases = [key for key in model_prompt_data.keys() if key != "Model Architecture" and key != "References" and key != "Architecture Prompting Implications" and key != "Heuristic Guidance" and key !=  "Meta Tag Format"] # Assuming they start from "Task-Agnostic Prompt" and are at the same level in the JSON structure



@st.cache_data
def query_model_openai(prompt,model):
    # Headers including Content-Type and Authorization with your API key
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }

    # Data payload
    payload_data = {
        "model": model,
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



@st.cache_data
def query_model_replicate(prompt,model):

    output = replicate.run(
        model,
        input={
            "prompt": f"{prompt}"
        }
    )


    # Initialize an empty string to accumulate the response
    response_content = ""

    # Iterate over the output generator
    for item in output:
        # Check if item is a string and append it to response_content
        if isinstance(item, str):
            response_content += item


    # Return the accumulated response content
    return response_content








## gui 
st.title('Prompt Engineering Starter Kit')

col1, col2 = st.columns([1, 1])

# get model type user selection
with col1:
    selected_model = st.selectbox(
        "Model Selection",
        model_names,
        index=1,
    )
    st.write(f":red[**Architecture:**] {model_prompt_data['Model Architecture'][selected_model]}  \n:red[**Prompting Tips:**] {model_prompt_data['Architecture Prompting Implications'][selected_model]}")

# get use case user selection
with col2:
    selected_prompt_use_case = st.radio(
        "Use Case for Sample Prompt",
        prompt_use_cases,
    )
    st.write(f":red[**Meta Tag Format:**]  \n {model_prompt_data['Meta Tag Format'][selected_model]}")


 

 
#st.write(f"**Model Architecture --** {model_prompt_data['Model Architecture'][selected_model]}")
#st.write(f"**How to Prompt --** {model_prompt_data['Architecture Prompting Implications'][selected_model]}")




prompt = st.text_area(
    "Prompt",
    model_prompt_data[selected_prompt_use_case][selected_model],
    height=210,
    )

if st.button('Send'):
    if selected_model == "Llama 2 7B":
        response_content = query_model_sn(prompt)
    elif selected_model == "Mistral 7B":
        response_content = query_model_replicate(prompt, "mistralai/mistral-7b-instruct-v0.1:5fe0a3d7ac2852264a25279d1dfb798acbc4d49711d126646594e212cb821749")
    elif selected_model == "DeepSeek Coder":
        response_content = query_model_replicate(prompt, "kcaverly/deepseek-coder-33b-instruct-gguf:ea964345066a8868e43aca432f314822660b72e29cab6b4b904b779014fe58fd")
    elif selected_model == "Falcon 40B":
        response_content = query_model_replicate(prompt, "joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173")
#    elif selected_model == "Bloom":
#        response_content = query_model_replicate(prompt)
    elif selected_model == "GPT 3.5T":
        response_content = query_model_openai(prompt, "gpt-3.5-turbo")

    # Print the response
    st.write(response_content)

 
 


with st.expander("Provide Feedback"):
    #goals_feedback = st.radio(
    #"What are you trying to accomplish by using this AI Starter Kit?",
    #["Build a PoC", "General learning", "Something else", "No Answer"],
    #index=3,
    #horizontal=True
    #)

    helpfulness_feedback = st.radio(
    "Is this AI Starter Kit helping you accomplish your goal?", 
    ["No", "Somewhat", "Yes", "No Answer"],
    index=3,
    horizontal=True
    )

    freetext_feedback = st.text_input("What are your goals with the kit, and what should we start, stop, or continue doing to help?")
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

        user_id = "streamlitcloud"
        try:
            user_id = os.getlogin()
        except:
            user_id = "streamlitcloud"

        mixpanel.track(f"$aisk:{user_id}:{user_email}", 'TEST_AISK:FEEDBACK_SUBMITTED',  {
        'Feedback: User Goal': f'{goals_feedback}',
        'Feedback: Are We Helping': f'{helpfulness_feedback}',
        'Feedback: Freetext': f'{freetext_feedback}',
        'Feedback: Email Entry': f'{user_email}',
        'Feedback: Streamlit Email': f'{st.experimental_user.email}',
        'Feedback: Session ID': f'{session_id}',
        'Feedback: Path': f'{path}',
        'Feedback: Login': f'{user_id}',
        })
        # Print the response
        st.write("Thank you, your feedback is a big deal to us!")




# evolutionary task prompt design automation
    
# beginner's mind, declutter with config file? how to make what they need to learn fit on a page. 
    # defenitely remove the feedback form to another file for centralization and reuse

# use cases as radio buttons? as it grow
        

# replicate for all models
# more specific model names?
# model selection - not really, will need 
# prompt quality