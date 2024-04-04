
<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Prompt Engineering Starter Kit
======================
<!-- TOC -->

- [Prompt Engineering Starter Kit](#prompt-engineering-starter-kit)
- [Overview](#overview)
    - [About this template](#about-this-template)
- [Getting started](#getting-started)
    - [Deploy your model in SambaStudio](#deploy-your-model-in-sambastudio)
    - [Integrate your model](#integrate-your-model)
    - [Deploy the starter kit](#deploy-the-starter-kit)
    - [Starterkit usage](#starterkit-usage)
- [Customizing the template](#customizing-the-template)
    - [Include model](#include-model)
    - [Edit a template](#edit-a-template)
    - [Add more templates](#add-more-templates)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->
# Getting started

## Deploy your model in SambaStudio

Begin creating an account and using the available models included in [Sambaverse](sambaverse.sambanova.net), and [get your API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) from the user button

Alternatively begin by deploying your LLM (e.g. Llama 2 70B chat, etc) to an endpoint for inference in SambaStudio either through the GUI or CLI, as described in the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).

## Integrate your model

Integrate your LLM deployed on SambaStudio or connect your Sambaverse API key with this AI starter kit in two simple steps:
1. Clone repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

2. **Sambaverse Endpoint:**  Update API information for your Sambaverse account.  These are represented as configurable variables in the environment variables file in the root repo directory **```sn-ai-starter-kit/.env```**. For example, an api key
"456789ab-cdef-0123-4567-89abcdef0123"
would be entered in the env file (with no spaces) as:
```
SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
```

Set in the [config file](./config.json), the variable *api* as: "sambaverse"

2. **SambaStudio Endpoint:**  Update API information for the SambaNova LLM.  These are represented as configurable variables in the environment variables file in the root repo directory **```sn-ai-starter-kit/.env```**. For example, an endpoint with the URL
"https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
would be entered in the env file (with no spaces) as:
```
BASE_URL="https://api-stage.sambanova.net"
PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
```

Set in the [config file](./config.json), the variable *api* as: "sambastudio"


3. Install requirements: It is recommended to use virtualenv or conda environment for installation, and to update pip.
```
cd ai-starter-kit/prompt-engineering
python3 -m venv prompt_engineering_env
source prompt_engineering_env/bin/activate
pip install -r requirements.txt
```

## Deploy the starter kit
To run the demo, run the following commands:
```
streamlit run streamlit/app.py --browser.gatherUsageStats false 
```

After deploying the starter kit you should see the following application user interface

![capture of prompt_engineering_demo](./docs/prompt_enginnering_app.png)

## Docker-usage

To run this with docker, run the command:

    docker-compose up --build

You will be prompted to go to the link (http://localhost:8501/) in your browser where you will be greeted with the streamlit page as above.

## Starterkit usage 

1- Choose the LLM to use from the options available under ***Model Selection*** (Currently, only Llama2 70B is available) Upon selection, you'll see a description of the architecture, along with prompting tips and the Meta tag format required to optimize the model's performance.

2- Select the template in ***Use Case for Sample Prompt***, you will find a list of available templates:

-  **General Assistant**: Provides comprehensive assistance on a wide range of topics, including answering questions, offering explanations, and giving advice. It's ideal for general knowledge, trivia, educational support, and everyday inquiries.

- **Document Search**: Specializes in locating and briefing relevant information from large documents or databases. Useful for research, data analysis, and extracting key points from extensive text sources.

- **Product Selection**: Assists in choosing products by comparing features, prices, and reviews. Ideal for shopping decisions, product comparisons, and understanding the pros and cons of different items.

- **Code Generation**: Helps in writing, debugging, and explaining code. Useful for software development, learning programming languages, and automating simple tasks through scripting.

- **Summarization**: Outputs a summary based on a given context. Essential for condensing large volumes of text 

3-  Review and edit the input to the model in the ***Prompt*** text input field

4- Click the ***Send*** button, button to submit the prompt. The model will retrieve and display the response.

# Customizing the template

## Include model

You can include more models to the kit 

If using Sambastudio:

- First [set your model] in Sambastudio (#integrate-your-model)
- Include your model description in the model section in the ```config.json``` file
- Populate key variables from your env file in ```streamlit/app.py```
- Define the method for calling the model, you can see the example *call_sambanova_llama2_70b_api* in ```streamlit/app.py```
- Include the method usage in ```st.buton(send)``` copletition section in the ```streamlit/app.py```

If using Sambaverse endpoint:

- Search in the available models in playground and select the three dots the click in show code, you should search the values of these two tags `modelName` and `select_expert` 
- Define the method for calling the model, you can see the example *call_sambaverse_llama2_70b_api* in ```streamlit/app.py``` setting the values of  `sambaverse_model_name` and the keyword argument `select_expert` 
- Include the method usage in ```st.buton(send)``` copletition section in the ```streamlit/app.py```


## Edit a template

You can edit one of the existing templates using the *create_prompt_yamls()* method in ```streamlit/app.py```, then execute the method, this will modify the prompt yaml file in ```prompts``` folder

## Add more templates

You can include more templates to the app following the instructions in [Edit a template](#edit-a-template) and then including the template use case in the *use_cases* list of ```config.yaml``` file

For further examples, we encourage you to visit any of the following resources:
- [Awesome chatgpt prompts](https://github.com/f/awesome-chatgpt-prompts)
- [Smith - Langchain hub](https://smith.langchain.com/hub)

# Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- streamlit (version 1.25.0)
- langchain (version 1.1.4)
- python-dotenv (version 1.0.0)
- Requests (version 2.31.0)
- sseclient (version 0.0.27)
- streamlit-extras (version 0.3.6)
- pydantic (version 1.10.14)
- pydantic_core (version 2.10.1)
