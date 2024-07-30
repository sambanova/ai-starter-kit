import logging
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import SambaStudioEmbeddings

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

class ModelsGateway():
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
                if coe:
                    if batch_size is None:
                        batch_size = 1
                    embeddings = SambaStudioEmbeddings(
                        sambastudio_embeddings_base_url=sambastudio_embeddings_base_url,
                        sambastudio_embeddings_base_uri=sambastudio_embeddings_base_uri,
                        sambastudio_embeddings_project_id=sambastudio_embeddings_project_id,
                        sambastudio_embeddings_endpoint_id=sambastudio_embeddings_endpoint_id,
                        sambastudio_embeddings_api_key=sambastudio_embeddings_api_key,
                        batch_size=batch_size,
                        model_kwargs = {
                            "select_expert":select_expert
                            }
                        )
                else:
                    if batch_size is None:
                        batch_size = 32
                    embeddings = SambaStudioEmbeddings(
                        sambastudio_embeddings_base_url=sambastudio_embeddings_base_url,
                        sambastudio_embeddings_base_uri=sambastudio_embeddings_base_uri,
                        sambastudio_embeddings_project_id=sambastudio_embeddings_project_id,
                        sambastudio_embeddings_endpoint_id=sambastudio_embeddings_endpoint_id,
                        sambastudio_embeddings_api_key=sambastudio_embeddings_api_key,
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
        