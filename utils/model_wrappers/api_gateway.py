import os
import sys
import logging
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_community.embeddings import SambaStudioEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.llms.sambanova import SambaStudio, Sambaverse

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(utils_dir, ".."))
sys.path.append(utils_dir)
sys.path.append(repo_dir)

from utils.model_wrappers.langchain_llms import SambaNovaFastAPI


EMBEDDING_MODEL = "intfloat/e5-large-v2"
NORMALIZE_EMBEDDINGS = True

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s", 
    handlers=[
        logging.StreamHandler(), 
    ],
)
logger = logging.getLogger(__name__)

class APIGateway():
    @staticmethod
    def load_embedding_model(
        type: str= "cpu",
        batch_size: Optional[int] = None,
        coe: bool = False,
        select_expert: Optional[str]  = None,
        sambastudio_embeddings_base_url: Optional[str]  = None,
        sambastudio_embeddings_base_uri: Optional[str]  = None,
        sambastudio_embeddings_project_id: Optional[str]  = None,
        sambastudio_embeddings_endpoint_id: Optional[str]  = None,
        sambastudio_embeddings_api_key: Optional[str]  = None,
        ) -> Embeddings:
            """Loads a langchain embedding model given a type and parameters
            Args:
                type (str): wether to use sambastudio embedding model or in local cpu model
                batch_size (int, optional): batch size for sambastudio model. Defaults to None.
                coe (bool, optional): whether to use coe model. Defaults to False. only for sambastudio models
                select_expert (str, optional): expert model to be used when coe selected. Defaults to None.
                    only for sambastudio models.
                sambastudio_embeddings_base_url (str, optional): base url for sambastudio model. Defaults to None.
                sambastudio_embeddings_base_uri (str, optional): endpoint base uri for sambastudio model. Defaults to None.
                sambastudio_embeddings_project_id (str, optional): project id for sambastudio model. Defaults to None.
                sambastudio_embeddings_endpoint_id (str, optional): endpoint id for sambastudio model. Defaults to None.
                sambastudio_embeddings_api_key (str, optional): api key for sambastudio model. Defaults to None.
            Returns:
                langchain embedding model
            """
    
            if type == "sambastudio":
                envs = {
                    "sambastudio_embeddings_base_url": sambastudio_embeddings_base_url,
                    "sambastudio_embeddings_base_uri": sambastudio_embeddings_base_uri,
                    "sambastudio_embeddings_project_id": sambastudio_embeddings_project_id,
                    "sambastudio_embeddings_endpoint_id": sambastudio_embeddings_endpoint_id,
                    "sambastudio_embeddings_api_key": sambastudio_embeddings_api_key,
                }
                envs = {k: v for k, v in envs.items() if v is not None}
                    
                if coe:
                    if batch_size is None:
                        batch_size = 1
                    embeddings = SambaStudioEmbeddings(
                        **envs,
                        batch_size=batch_size,
                        model_kwargs = {
                            "select_expert":select_expert
                            }
                        )
                else:
                    if batch_size is None:
                        batch_size = 32
                    embeddings = SambaStudioEmbeddings(
                        **envs,
                        batch_size=batch_size
                    )
            elif type == "cpu":
                encode_kwargs = {"normalize_embeddings": NORMALIZE_EMBEDDINGS}
                embedding_model = EMBEDDING_MODEL
                embeddings = HuggingFaceInstructEmbeddings(
                    model_name=embedding_model,
                    embed_instruction="",  # no instruction is needed for candidate passages
                    query_instruction="Represent this sentence for searching relevant passages: ",
                    encode_kwargs=encode_kwargs,
                )
            else:
                raise ValueError(f"{type} is not a valid embedding model type")

            return embeddings
        
    @staticmethod
    def load_llm(
        type: str,
        streaming: bool = False,
        coe: bool = False,
        
        do_sample: Optional[bool] = None,
        max_tokens_to_generate: Optional[int] = None,
        temperature: Optional[float] = None,
        select_expert: Optional[str] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop_sequences: Optional[str] = None,
        process_prompt: Optional[bool] = False,
        
        sambaverse_url: Optional[str] = None,
        sambaverse_api_key: Optional[str] = None,
        sambaverse_model_name: Optional[str] = None,
        
        sambastudio_base_url: Optional[str] = None,
        sambastudio_base_uri: Optional[str] = None,
        sambastudio_project_id: Optional[str] = None,
        sambastudio_endpoint_id: Optional[str] = None,
        sambastudio_api_key: Optional[str] = None,
        
        fastapi_url: Optional[str] = None,
        fastapi_api_key: Optional[str] = None,        
        ) -> LLM:
            """Loads a langchain Sambanova llm model given a type and parameters
            Args:
                type (str): wether to use sambastudio, sambaverse, or fastapi model
                streaming (bool): wether to use streaming method. Defaults to False.
                coe (bool): whether to use coe model. Defaults to False.
                
                do_sample (bool) : Optional wether to do sample.
                max_tokens_to_generate (int) : Optional max number of tokens to generate.
                temperature (float) : Optional model temperature.
                select_expert (str) : Optional expert to use when using CoE models.
                top_p (float) : Optional model top_p.
                top_k (int) : Optional model top_k.
                repetition_penalty (float) : Optional model repetition penalty.
                stop_sequences (str) : Optional model stop sequences.
                process_prompt (bool) : Optional default to false.
                
                sambaverse_model_name (str): Optional sambaverse model name to use.
                sambaverse_url (str): Optional Sambaverse url to use.
                sambaverse_api_key (str): Optional Sambaverse api key to use.
                
                sambastudio_base_url (str): Optional SambaStudio environment URL".
                sambastudio_base_uri (str): Optional SambaStudio-base-URI".
                sambastudio_project_id (str): Optional SambaStudio project ID.
                sambastudio_endpoint_id (str): Optional SambaStudio endpoint ID.
                sambastudio_api_token (str): Optional SambaStudio endpoint API key.
                
                fastapi_url (str): Optional fastApi CoE endpoint URL",
                fastapi_api_key (str): Optional fastApi CoE endpoint API key.,
                
            Returns:
                langchain llm model
            """

            if type == "sambaverse":
                
                envs = {
                    "sambaverse_url": sambaverse_url,
                    "sambaverse_api_key": sambaverse_api_key,
                    "sambaverse_model_name":sambaverse_model_name,
                }
                envs = {k: v for k, v in envs.items() if v is not None}
                
                model_kwargs = {
                    "do_sample": do_sample,
                    "max_tokens_to_generate": max_tokens_to_generate,
                    "temperature": temperature,
                    "select_expert": select_expert,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "stop_sequences": stop_sequences,
                    "process_prompt": process_prompt,
                }
                model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
                
                llm = Sambaverse(
                    **envs,
                    streaming=streaming,
                    model_kwargs=model_kwargs,
                )
            elif type == "sambastudio":
                envs = {
                    "sambastudio_base_url": sambastudio_base_url,
                    "sambastudio_base_uri": sambastudio_base_uri,
                    "sambastudio_project_id": sambastudio_project_id,
                    "sambastudio_endpoint_id": sambastudio_endpoint_id,
                    "sambastudio_api_key": sambastudio_api_key,
                }
                envs = {k: v for k, v in envs.items() if v is not None}
                if coe:
                    model_kwargs = {
                        "do_sample": do_sample,
                        "max_tokens_to_generate": max_tokens_to_generate,
                        "temperature": temperature,
                        "select_expert": select_expert,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "stop_sequences": stop_sequences,
                        "process_prompt": process_prompt,
                    }
                    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
                    
                    llm = SambaStudio(
                        **envs,
                        streaming=streaming,
                        model_kwargs=model_kwargs,
                    )
                else:
                    model_kwargs = {
                        "do_sample": do_sample,
                        "max_tokens_to_generate": max_tokens_to_generate,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "stop_sequences": stop_sequences,
                    }
                    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
                    llm = SambaStudio(
                        **envs,
                        streaming=streaming,
                        model_kwargs=model_kwargs,
                    )
        
            elif type == "fastapi":
                envs = {
                    "fastapi_url":fastapi_url,
                    "fastapi_api_key": fastapi_api_key,
                }
                envs = {k: v for k, v in envs.items() if v is not None}
                llm = SambaNovaFastAPI(
                    ** envs,
                    max_tokens=max_tokens_to_generate,
                    model=select_expert,
                )
            
            else:
                raise ValueError(
                    f"Invalid LLM API: {type}, only 'fastapi' 'sambastudio' and 'sambaverse' are supported."
                )
                
            return llm