<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Prompt Engineering Starter Kit
======================
<!-- TOC -->


- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the account and config file](#set-up-the-account-and-config-file)
        - [Setup for Sambaverse users](#setup-for-sambaverse-users)
        - [Setup for SambaStudio users](#setup-for-sambastudio-users)
- [Deploy the starter kit GUI](#deploy-the-starter-kit-gui)
    - [Option 1: Use a virtual environment](#option-1-use-a-virtual-environment)
    - [Option 2: Deploy the starter kit in a Docker container](#option-2-deploy-the-starter-kit-in-a-docker-container)
- [Use the starter kit GUI](#use-the-starterkit-gui)
- [Customize the starter kit](#customize-the-starter-kit)
    - [Include additional models](#include-additional-models)
        - [Include models using SambaStudio](#include-models-using-sambastudio)
        - [Include models using Sambaverse](#include-models-using-sambaverse)
    - [Edit a prompt template](#edit-a-prompt-template)
    - [Add prompt templates](#add-prompt-templates)
- [Examples, third-party tools, and data sources](#examples-third-party-tools-and-data-sources)

<!-- /TOC -->
# Before you begin

You have to set up your environment before you can run the starter kit. 

## Clone this repository

Clone the starter kit repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the account and config file

The next step sets you up to use one of the models available from SambaNova. It depends on whether you're a SambaNova customer who uses SambaStudio or want to use the publicly available Sambaverse. 

### Setup for Sambaverse users 

1. Create a Sambaverse account at [Sambaverse](sambaverse.sambanova.net) and select your model. 
2. Get your [Sambaverse API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) (from the user button).
3. In the repo root directory find the config file in `sn-ai-starter-kit/.env` and specify the Sambaverse API key, as in the following example: 

```yaml
    SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
```

4. In the [config file](./config.yaml), set the `api` variable to `"sambaverse"`.


### Setup for SambaStudio users

To perform this setup, you must be a SambaNova customer with a SambaStudio account. 

1. Log in to SambaStudio and get your API authorization key. The steps for getting this key are described [here](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html#_acquire_the_api_key).
2. Select the LLM you want to use (e.g. Llama 2 70B chat) and deploy an endpoint for inference. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).
3. Update the `sn-ai-starter-kit/.env` config file in the root repo directory. Here's an example: 

```
BASE_URL="https://api-stage.sambanova.net"
PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
```

4. Open the [config file](./config.yaml) and set  the variable `api` to `"sambastudio"`.

# Deploy the starter kit GUI

We recommend that you run  the the starter kit in a virtual environment or use a container. 

## Option 1: Use a virtual environment

If you want to use virtualenv or conda environment 

1. Install and update pip.

    ```
    cd ai-starter-kit/prompt-engineering
    python3 -m venv prompt_engineering_env
    source prompt_engineering_env/bin/activate
    pip install -r requirements.txt
    ```
2. Run the following command:
    ```
    streamlit run streamlit/app.py --browser.gatherUsageStats false 
    ```

You should see the following user interface:

![capture of prompt_engineering_demo](./docs/prompt_enginnering_app.png)


## Option 2: Deploy the starter kit in a Docker container 

If you want to use Docker:

1. Update the `SAMBASTUDIO_KEY`, `SNAPI`, `SNSDK` args in [docker-compose.yaml file](docker-compose.yaml)

2. Run the command:

    docker-compose up --build

You will be prompted to go to the link (http://localhost:8501/) in your browser where you will be greeted with the streamlit page as above.


# Use the starterkit GUI 

To use the starter kit, follow these steps:

1. Choose the LLM to use from the options available under **Model Selection** (Currently, only Llama2 70B is available). You'll see a description of the architecture, prompting tips, and the metatag format required to optimize the model's performance.

2. In **Use Case for Sample Prompt**, select a template. You have the following choices:

    - **General Assistant**: Provides comprehensive assistance on a wide range of topics, including answering questions, offering explanations, and giving advice. It's ideal for general knowledge, trivia, educational support, and everyday inquiries.

    - **Document Search**: Specializes in locating and briefing relevant information from large documents or databases. Useful for research, data analysis, and extracting key points from extensive text sources.

    - **Product Selection**: Assists in choosing products by comparing features, prices, and reviews. Ideal for shopping decisions, product comparisons, and understanding the pros and cons of different items.

    - **Code Generation**: Helps in writing, debugging, and explaining code. Useful for software development, learning programming languages, and automating simple tasks through scripting.

    - **Summarization**: Outputs a summary based on a given context. Essential for condensing large volumes of text 

3. In the **Prompt** field, review and edit the input to the model 

4. Click the **Send** button to submit the prompt. The model will retrieve and display the response.

# Customize the starter kit

You have several options for customizing this starter kit. 

## Include additional models

You can include more models with the kit. They will then show up in the **Model Selection** pulldown in the GUI. 

### Include models using SambaStudio

If you're using a SambaStudio endpoint, follow these steps:

1. Create a SambaStudio endpoint for inference. 
2. In the `config.json` file, include the model description in the model section 
3. Populate key variables from your env file in `streamlit/app.py`
4. Define the method for calling the model. See `call_sambanova_llama2_70b_api` in `streamlit/app.py` for an example.
5. Include the new method in the `st.button(send)` section in the `streamlit/app.py`.

### Include models using Sambaverse

If you're using a Sambaverse endpoint, follow these steps:

1. In the playground, find the model you're interested in. 
2. Select the three dots and then **Show code** and note down the values of `modelName` and `select_expert`. 
3. Define the method for calling the model. In `streamlit/app.py`, set the values of  `sambaverse_model_name` and `select_expert`. See `call_sambaverse_llama2_70b_api` for an example. 
4. Include the new method in the `st.button(send)` section in the `streamlit/app.py`.`


## Edit a prompt template

To change a template: 

1. Edit the `create_prompt_yamls()` method in `streamlit/app.py`.
2. Execute the method to modify the prompt yaml file in the `prompts` folder.

## Add prompt templates

To add a prompt template:

1. Follow the instructions in [Edit a template](#edit-a-prompt-template).
2. Include the template use case in the `use_cases` list of `config.yaml` file.


# Examples, third-party tools, and data sources

For further examples, we encourage you to visit any of the following resources:
- [Awesome chatgpt prompts](https://github.com/f/awesome-chatgpt-prompts)
- [Smith - Langchain hub](https://smith.langchain.com/hub)

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- streamlit (version 1.25.0)
- langchain (version 1.1.4)
- python-dotenv (version 1.0.0)
- Requests (version 2.31.0)
- sseclient (version 0.0.27)
- streamlit-extras (version 0.3.6)
- pydantic (version 1.10.14)
- pydantic_core (version 2.10.1)
