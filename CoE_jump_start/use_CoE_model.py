import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from utils.sambanova_endpoint import (
    SambaNovaEndpoint,
    SambaverseEndpoint,
    SambaNovaEmbeddingModel,
)
import yaml
from snsdk import SnSdk
import json


# Use embeddings As Part of Langchain
from langchain.vectorstores import Chroma

CONFIG_PATH = os.path.join(current_dir, "config.yaml")

with open(CONFIG_PATH, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)
api_info = config["api"]
llm_info = config["llm"]


# load dot_env
load_dotenv(os.path.join(current_dir, ".env"))

try:
    from snsdk import SnSdk
except ImportError:
    snsdk_installed = False


def get_expert_val(res):
    if not res or not res.get("data") or not res["data"]:
        return "Generalist"

    supported_experts_map = {
        "finance": "Finance expert",
        "economics": "Finance expert",
        "maths": "Math expert",
        "mathematics": "Math expert",
        "code generation": "Code expert",
        "computer science": "Code expert",
        "legal": "Legal expert",
        "medical": "Medical expert",
    }

    # String check - medicals or health

    data = (res["data"][0].get("completion", "") or "").strip().lower()
    supported_experts = list(supported_experts_map.keys())
    expert = next((x for x in supported_experts if x in data), "Generalist")
    return supported_experts_map.get(expert, "Generalist")


def get_expert(
    input_text,
    do_sample=False,
    max_tokens_to_generate=500,
    repetition_penalty=1.0,
    temperature=0.7,
    top_k=50,
    top_p=1.0,
    select_expert="llama-2-7b-chat-hf",
):
    """
    Classifies the given input text into one of the following categories:
    'finance', 'economics', 'maths', 'code generation', 'legal', 'medical' or 'None of the above'.

    Args:
        input_text (str): The input text to classify.
        do_sample (bool): Whether to sample from the model's output distribution. Default is False.
        max_tokens_to_generate (int): The maximum number of tokens to generate. Default is 30.
        repetition_penalty (float): The penalty for repeating tokens. Default is 1.0.
        temperature (float): The temperature for sampling. Default is 0.7.
        top_k (int): The number of top most likely tokens to consider. Default is 50.
        top_p (float): The cumulative probability threshold for top-p sampling. Default is 1.0.
        select_expert (str): The name of the expert model to use. Default is "Mistral-7B-Instruct-v0.2".

    Returns:
        str: The response from the model.
    """
    sdk = SnSdk("https://sjc3-svqa.sambanova.net", "endpoint_secret")

    prompt = """<s>[INST] 

A message can be classified as only one of the following categories: 'finance',  'economics',  'maths',  'code generation', 'legal', 'medical', 'history' or 'None of the above'.  

Examples for few of these categories are given below:
- 'code generation': Write a python program
- 'code generation': Debug the following code
- 'None of the above': Who are you?
- 'None of the above': What are you?
- 'None of the above': Where are you?

Based on the above categories, classify this message: 

{input}

Always remember the following instructions while classifying the given statement:
- Think carefully and if you are not highly certain then classify the  given statement as 'None of the above'
- Always begin your response by putting the classified category of the given statement after  '<<detected category>>:'
- Explain you answer

[/INST]"""

    inputs = (
        r'{"conversation_id":"sambaverse-conversation-id","messages":[{"message_id":0,"role":"user","content":"'
        + input_text
        + '"}], "prompt": "'
        + prompt
        + '"}'
    )

    tuning_params = {
        "do_sample": {"type": "bool", "value": str(do_sample).lower()},
        "max_tokens_to_generate": {"type": "int", "value": str(max_tokens_to_generate)},
        "repetition_penalty": {"type": "float", "value": str(repetition_penalty)},
        "temperature": {"type": "float", "value": str(temperature)},
        "top_k": {"type": "int", "value": str(top_k)},
        "top_p": {"type": "float", "value": str(top_p)},
        "select_expert": {"type": "str", "value": select_expert},
        "process_prompt": {"type": "bool", "value": "false"},
    }

    tuning_params_str = json.dumps(tuning_params)

    response = sdk.nlp_predict(
        os.getenv("PROJECT_ID"),
        os.getenv("ENDPOINT_ID"),
        os.getenv("API_KEY"),
        inputs,
        tuning_params_str,
    )

    return response


def main():
    # snsdk_model retuns a langchain Embedding Object which can be used within langchain
    snsdk_model = SambaNovaEmbeddingModel()

    # A Small Example to see how SNS embeddings can be integrated within Langchain Workflow
    # Let's set embeddings to equal or snsdk_model to keep with the langchain convention
    embeddings = snsdk_model
    loader = WebBaseLoader("https://docs.smith.langchain.com")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = Chroma.from_documents(documents, embeddings)

    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}"""
    )

    # Example 1 - Using SambaVerse to call CoE Model. They provide the expert name, and their Sambaverse api_key
    if api_info == "sambaverse":
        llm = SambaverseEndpoint(
            sambaverse_model_name=llm_info["sambaverse_model_name"],
            # sambaverse_url=os.getenv("SAMBAVERSE_URL"),
            sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
            model_kwargs={
                "do_sample": False,
                "max_tokens_to_generate": llm_info["max_tokens_to_generate"],
                "temperature": llm_info["temperature"],
                "process_prompt": True,
                "select_expert": llm_info["samabaverse_select_expert"]
                # "stop_sequences": { "type":"str", "value":""},
                # "repetition_penalty": {"type": "float", "value": "1"},
                # "top_k": {"type": "int", "value": "50"},
                # "top_p": {"type": "float", "value": "1"}
            },
        )

    # Example 2a - Using SambaStudio to call CoE with Named Expert
    elif api_info == "sambastudio":
        llm = SambaNovaEndpoint(
            model_kwargs={
                "do_sample": True,
                "temperature": llm_info["temperature"],
                "max_tokens_to_generate": llm_info["max_tokens_to_generate"],
                # If using a CoE endpoint, you must select the expert using the select_expert parameter below and set process prompt = False.
                "select_expert": llm_info["samabaverse_select_expert"],
                "process_prompt": False,
                # "stop_sequences": { "type":"str", "value":""},
                # "repetition_penalty": {"type": "float", "value": "1"},
                # "top_k": {"type": "int", "value": "50"},
                # "top_p": {"type": "float", "value": "1"}
            }
        )

    user_query = "Give me the code for creating a vector db in langchain"

    if llm_info["coe_routing"]:
        # Example 2B - Use SambaStudio to Call CoE with Routing v1

        # We Get the Expert By Calling SambaStudio with a Custom Prompt Workflow
        expert_response = get_expert(user_query)
        print(expert_response)

        # After this we extract the expert name from the response
        expert = get_expert_val(expert_response)
        print(expert)

        # Once we have this we simply instantiate the LLM by calling the CoE using a named model, by mapping the expert name to the relevant model.
        # Lookup Model Name
        coe_name_map = {
            "Finance expert": "finance-chat",
            "Math expert": "deepseek-llm-67b-chat",
            "Code expert": "deepseek-llm-67b-chat",
            "Medical expert": "medicine-chat",
            "Legal expert": "law-chat",
            "Generalist": "Mistral-7B-Instruct-v0.2",
        }
        named_expert = coe_name_map[expert]
        print(named_expert)

        llm = SambaNovaEndpoint(
            model_kwargs={
                "do_sample": True,
                "temperature": llm_info["temperature"],
                "max_tokens_to_generate": llm_info["max_tokens_to_generate"],
                # If using a CoE endpoint, you must select the expert using the select_expert parameter below and set process prompt = False.
                "select_expert": named_expert,
                "process_prompt": False,
                # "stop_sequences": { "type":"str", "value":""},
                # "repetition_penalty": {"type": "float", "value": "1"},
                # "top_k": {"type": "int", "value": "50"},
                # "top_p": {"type": "float", "value": "1"}
            }
        )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": user_query})

    print(response["answer"])


if __name__ == "__main__":
    main()
