import logging
import os
import sys
from typing import Any, Dict, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_huggingface import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)

from langchain_sambanova import ChatSambaNovaCloud, ChatSambaStudio, SambaNovaCloudEmbeddings, SambaStudioEmbeddings

from utils.model_wrappers.langchain_llms import SambaNovaCloud, SambaStudio

EMBEDDING_MODEL = 'intfloat/e5-large-v2'
NORMALIZE_EMBEDDINGS = True

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class APIGateway:
    @staticmethod
    def load_embedding_model(
        type: str,
        batch_size: Optional[int] = None,
        bundle: bool = False,
        model: Optional[str] = None,
        select_expert: Optional[str] = None,
        dimensions: Optional[int] = None,
        max_characters: Optional[int] = None,
        sambastudio_url: Optional[str] = None,
        sambastudio_api_key: Optional[str] = None,
        sambanova_url: Optional[str] = None,
        sambanova_api_key: Optional[str] = None,
    ) -> Embeddings:
        """Loads a langchain embedding model given a type and parameters
        Args:
            type (str): wether to use sambastudio embedding model or in local cpu model
            batch_size (int, optional): batch size for sambastudio model. Defaults to None.
            bundle (bool, optional): whether to use bundle model. Defaults to False. only for sambastudio models

            model (str) : Optional expert to use when using CoE models or cloud.
            select_expert (str) : Optional alias for model.

            dimensions (int) : shorten embeddings by trimming some values from the end of the sequence
            max_characters (int) : max characters, longer will be trimmed

            sambastudio_url (str): Optional SambaStudio environment URL".
            sambastudio_api_token (str): Optional SambaStudio endpoint API key.

            sambanova_url (str): Optional SambaNova Cloud URL",
            sambanova_api_key (str): Optional SambaNovaCloud API key.

        Returns:
            langchain embedding model
        """
        if type == 'sncloud':
            envs = {
                'sambanova_url': sambanova_url,
                'sambanova_api_key': sambanova_api_key,
            }
            envs = {k: v for k, v in envs.items() if v is not None}
            if batch_size is None:
                batch_size = 16
            if max_characters is None:
                max_characters = 16384
            embeddings = SambaNovaCloudEmbeddings(
                **envs,
                batch_size=batch_size,
                model=model or select_expert,
                max_characters=max_characters,
                dimensions=dimensions,
            )

        elif type == 'sambastudio':
            envs = {
                'sambastudio_url': sambastudio_url,
                'sambastudio_api_key': sambastudio_api_key,
            }
            envs = {k: v for k, v in envs.items() if v is not None}
            extra_args = {'max_characters': max_characters, 'dimensions': dimensions}
            extra_args = {k: v for k, v in extra_args.items() if v is not None}

            if bundle:
                if batch_size is None:
                    batch_size = 1
                embeddings = SambaStudioEmbeddings(
                    **envs,
                    **extra_args,
                    batch_size=batch_size,
                    model=model or select_expert,
                )
            else:
                if batch_size is None:
                    batch_size = 32
                embeddings = SambaStudioEmbeddings(
                    **envs,
                    **extra_args,
                    batch_size=batch_size,
                )
        elif type == 'cpu':
            encode_kwargs = {
                'normalize_embeddings': NORMALIZE_EMBEDDINGS,
                'prompt': 'Represent this sentence for searching relevant passages: ',
            }
            embedding_model = EMBEDDING_MODEL
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                encode_kwargs=encode_kwargs,
            )
        else:
            raise ValueError(f'{type} is not a valid embedding model type')

        return embeddings

    @staticmethod
    def load_llm(
        type: str,
        streaming: bool = False,
        bundle: bool = False,
        do_sample: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        max_tokens_to_generate: Optional[int] = 1024,
        temperature: Optional[float] = 0.7,
        model: Optional[str] = None,
        select_expert: Optional[str] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stop_sequences: Optional[str] = None,
        process_prompt: Optional[bool] = False,
        sambastudio_url: Optional[str] = None,
        sambastudio_api_key: Optional[str] = None,
        sambanova_url: Optional[str] = None,
        sambanova_api_key: Optional[str] = None,
    ) -> LLM:
        """Loads a langchain Sambanova llm model given a type and parameters
        Args:
            type (str): whether to use sambastudio, or SambaNova Cloud model "sncloud"
            streaming (bool): whether to use streaming method. Defaults to False.
            bundle (bool): whether to use bundle model. Defaults to False.

            do_sample (bool) : Optional whether to do sample.
            max_tokens (int) : Optional max number of tokens to generate.
            max_tokens_to_generate (int) : Optional alias for max_tokens.
            temperature (float) : Optional model temperature.
            model (str) : Optional expert to use when using CoE models or cloud.
            select_expert (str) : Optional alias for model.
            top_p (float) : Optional model top_p.
            top_k (int) : Optional model top_k.
            repetition_penalty (float) : Optional model repetition penalty.
            stop_sequences (str) : Optional model stop sequences.
            process_prompt (bool) : Optional default to false.

            sambastudio_url (str): Optional SambaStudio environment URL".
            sambastudio_api_token (str): Optional SambaStudio endpoint API key.

            sambanova_url (str): Optional SambaNova Cloud URL",
            sambanova_api_key (str): Optional SambaNovaCloud API key.

        Returns:
            langchain llm model
        """

        if type == 'sambastudio':
            envs = {
                'sambastudio_url': sambastudio_url,
                'sambastudio_api_key': sambastudio_api_key,
            }
            envs = {k: v for k, v in envs.items() if v is not None}
            if bundle:
                model_kwargs = {
                    'do_sample': do_sample,
                    'max_tokens': max_tokens or max_tokens_to_generate,
                    'temperature': temperature,
                    'model': model or select_expert,
                    'top_p': top_p,
                    'repetition_penalty': repetition_penalty,
                    'stop_sequences': stop_sequences,
                    'process_prompt': process_prompt,
                }
                model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

                llm = SambaStudio(
                    **envs,
                    streaming=streaming,
                    model_kwargs=model_kwargs,
                )
            else:
                model_kwargs = {
                    'do_sample': do_sample,
                    'max_tokens': max_tokens or max_tokens_to_generate,
                    'temperature': temperature,
                    'top_p': top_p,
                    'repetition_penalty': repetition_penalty,
                    'stop_sequences': stop_sequences,
                }
                model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
                llm = SambaStudio(
                    **envs,
                    streaming=streaming,
                    model_kwargs=model_kwargs,
                )

        elif type == 'sncloud':
            envs = {
                'sambanova_url': sambanova_url,
                'sambanova_api_key': sambanova_api_key,
            }
            envs = {k: v for k, v in envs.items() if v is not None}
            llm = SambaNovaCloud(
                **envs,
                max_tokens=max_tokens or max_tokens_to_generate,
                model=model or select_expert,
                temperature=temperature,
                top_p=top_p,
            )

        else:
            raise ValueError(f"Invalid LLM API: {type}, only 'sncloud' and 'sambastudio' are supported.")

        return llm

    @staticmethod
    def load_chat(
        type: str,
        model: str,
        streaming: bool = False,
        max_tokens: int = 1024,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        process_prompt: Optional[bool] = True,
        stream_options: Optional[Dict[str, bool]] = {'include_usage': True},
        special_tokens: Optional[Dict[str, str]] = {
            'start': '<|begin_of_text|>',
            'start_role': '<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>',
            'end_role': '<|eot_id|>',
            'end': '<|start_header_id|>assistant<|end_header_id|>\n',
        },
        model_kwargs: Optional[Dict[str, Any]] = None,
        sambanova_url: Optional[str] = None,
        sambanova_api_key: Optional[str] = None,
        sambastudio_url: Optional[str] = None,
        sambastudio_api_key: Optional[str] = None,
    ) -> BaseChatModel:
        """
        Loads a langchain Sambanova chat model given some parameters
        Args:
            type (str): whether to use sambastudio, or SambaNova Cloud chat model "sncloud"
            model (str): The name of the model to use, e.g., llama3-8b.
            streaming (bool): whether to use streaming method. Defaults to False.
            max_tokens (int): Optional max number of tokens to generate.
            temperature (float): Optional model temperature.
            top_p (float): Optional model top_p.
            top_k (int): Optional model top_k.
            do_sample (bool): whether to do sampling
            process_prompt (bool): whether use API process prompt (for CoE generic v1 and v2 endpoints)
            stream_options (dict): stream options, include usage to get generation metrics
            special_tokens (dict): start, start_role, end_role and end special tokens
            (set for CoE generic v1 and v2 endpoints when process prompt set to false
            or for StandAlone v1 and v2 endpoints)
            default to llama3 special tokens
            model_kwargs (dict): Key word arguments to pass to the model.


            sambanova_url (str): Optional SambaNova Cloud URL",
            sambanova_api_key (str): Optional SambaNovaCloud API key.
            sambastudio_url (str): Optional SambaStudio URL",
            sambastudio_api_key (str): Optional SambaStudio API key.

        Returns:
            langchain BaseChatModel
        """

        if type == 'sambastudio':
            envs = {
                'sambastudio_url': sambastudio_url,
                'sambastudio_api_key': sambastudio_api_key,
            }
            envs = {k: v for k, v in envs.items() if v is not None}
            model = ChatSambaStudio(
                **envs,
                model=model,
                streaming=streaming,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                process_prompt=process_prompt,
                stream_options=stream_options,
                special_tokens=special_tokens,
                model_kwargs=model_kwargs,
            )

        elif type == 'sncloud':
            envs = {
                'sambanova_url': sambanova_url,
                'sambanova_api_key': sambanova_api_key,
            }
            envs = {k: v for k, v in envs.items() if v is not None}
            model = ChatSambaNovaCloud(
                **envs,
                model=model,
                streaming=streaming,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream_options=stream_options,
            )

        else:
            raise ValueError(f"Invalid LLM API: {type}, only 'sncloud' and 'sambastudio' are supported.")

        return model
