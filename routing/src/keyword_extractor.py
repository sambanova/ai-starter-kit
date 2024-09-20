import sys, os, yaml
from typing import Union
from keybert import KeyBERT, KeyLLM
from keyphrase_vectorizers import KeyphraseTfidfVectorizer
from custom_models import CustomEmbedder, CustomTextGeneration
from keyLLM import CustomKeyLLM
from langchain_core.prompts import load_prompt
import torch
import pickle
from dotenv import load_dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(repo_dir)
sys.path.append(kit_dir)
sys.path.append(current_dir)

load_dotenv(os.path.join(repo_dir, '.env'))

from utils.model_wrappers.api_gateway import APIGateway

def read_files(directory: str, extension=".txt") -> list:
    """
    Read files from directory.

    Args:
        directory (str): The directory path that contains files.
        extension (str, optional):The extension of the files. Defaults to ".txt".

    Raises:
        ValueError: Check if the directory exist.

    Returns:
        list: the list of file contents.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} doesn't exist!")
    file_contents = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_contents.append(file.read())
    return file_contents

class KeywordExtractor:
    """
    Extract keywords using KeyBert https://github.com/MaartenGr/keyBERT
    """
    def __init__(self, configs: str, 
                 docs: list[str], 
                 use_bert: bool=True, 
                 use_llm: bool=False) -> None:
        """_summary_

        Args:
            configs (str): The config file path.
            docs (list[str]): The list of docs contents.
            use_bert (bool, optional): If use bert as keyword extractor. Defaults to True.
            use_llm (bool, optional): If use llm as keyword extractor. Defaults to False.
        """
        self.configs = self.load_config(configs)
        self.docs = docs
        self.use_bert = use_bert
        self.use_llm = use_llm
        self.load_models()
        self.create_kw_models()

    def load_config(self, filename: str) -> dict:
        """
        Loads a YAML configuration file and returns its contents as a dictionary.

        Args:
            filename: The path to the YAML configuration file.

        Returns:
            A dictionary containing the configuration file's contents.
        """

        try:
            with open(filename, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'The YAML configuration file {filename} was not found.')
        except yaml.YAMLError as e:
            raise RuntimeError(f'Error parsing YAML file: {e}')
      
    def load_models(self) -> None:
        """
        Load embedding model and LLM model.
        """
        sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')
        sambastudio_embeddings_base_url = os.environ.get('SAMBASTUDIO_EMBEDDINGS_BASE_URL')
        sambastudio_embeddings_base_uri = os.environ.get('SAMBASTUDIO_EMBEDDINGS_BASE_URI')
        sambastudio_embeddings_project_id = os.environ.get('SAMBASTUDIO_EMBEDDINGS_PROJECT_ID')
        sambastudio_embeddings_endpoint_id = os.environ.get('SAMBASTUDIO_EMBEDDINGS_ENDPOINT_ID')
        sambastudio_embeddings_api_key = os.environ.get('SAMBASTUDIO_EMBEDDINGS_API_KEY')
        
        # 1. load embedding model
        self.embedding_model = APIGateway.load_embedding_model(
                                type=self.configs['embedding_model']['type'],
                                batch_size=self.configs['embedding_model']['batch_size'],
                                coe=self.configs['embedding_model']['coe'],
                                select_expert=self.configs['embedding_model']['select_expert'],
                                sambastudio_embeddings_base_url=sambastudio_embeddings_base_url,
                                sambastudio_embeddings_base_uri=sambastudio_embeddings_base_uri,
                                sambastudio_embeddings_project_id=sambastudio_embeddings_project_id,
                                sambastudio_embeddings_endpoint_id=sambastudio_embeddings_endpoint_id,
                                sambastudio_embeddings_api_key=sambastudio_embeddings_api_key,
                            )

        # 2. load llm model
        if self.use_llm:
            self.llm = APIGateway.load_llm(
                    type=self.configs["api"],
                    streaming=False,
                    coe=self.configs['router']["coe"],
                    do_sample=self.configs['router']["do_sample"],
                    max_tokens_to_generate=self.configs['router']["max_tokens_to_generate"],
                    temperature=self.configs['router']["temperature"],
                    select_expert=self.configs['router']["select_expert"],
                    process_prompt=False,
                    sambanova_api_key=sambanova_api_key,
            )

    def create_kw_models(self, use_llm_prompt: bool=False) -> None:
        """
        Create keword extractor using KeyBERT.

        Args:
            use_llm_prompt (bool, optional): If use customized prompt for llm. Defaults to False.
                                             Only applied when self.use_llm=True 

        Raises:
            NotImplementedError: Not support use both bert and llm as keyword extractor.
        """
        # create kw_model
        self.custom_embedder = CustomEmbedder(embedding_model=self.embedding_model)
        # pass custom backend to keybert
        self.kw_bert_model = KeyBERT(model=self.custom_embedder)
        # load it in KeyLLM
        if self.use_bert and self.use_llm:
            raise NotImplementedError("Not support both bert and generative llm yet.")
        elif self.use_bert:
            self.kw_llm_model = CustomKeyLLM(self.kw_bert_model)
        elif self.use_llm:
            llm_prompt = None
            if use_llm_prompt:
                llm_prompt = load_prompt(repo_dir + '/' + self.configs['prompts']['kw_etr_prompt']).template
            self.text_generator = CustomTextGeneration(self.llm, llm_prompt)
            self.kw_llm_model = KeyLLM(self.text_generator)

    def docs_embedding(self) -> None:
        """
        Embedding documents.
        """
        # embed docs
        self.docs_embed = self.custom_embedder.embed(documents=self.docs)

    def extract_first_values(self, data: list, return_list: bool=False) -> Union[set, list]:
        """
        Extract only the first set of keywords in each cluster, since each file in the same cluster has the same keywords. 

        Args:
            data (list): The list of keywords in all clusters. 
            return_list (bool, optional): Format the results as list or set. Defaults to False (format as set).

        Returns:
            Union[set, list]: The set/list of keywords in each cluster.
        """
        if return_list:
            result = []
            for sublist in data:
                sublist_result = []
                for s in sublist:
                    # Extract the first element from the set
                    first_value = next(iter(s))
                    sublist_result.append(first_value)
                result.append(sublist_result)
        else:
            result = set()
            for sublist in data:
                for s in sublist:
                    # Extract the first element from the set
                    first_value = next(iter(s))
                    result.add(first_value)    
        return result
 
    def extract_keywords(self, 
                         use_clusters: bool=True, 
                         use_vectorizer: bool=True, 
                         keyphrase_ngram_range: tuple[int, int] = (1,1)) -> Union[set, list]:
        """
        Extract keywords from docs.

        Args:
            use_clusters (bool, optional): If enabled, semantically similar files are grouped into the same cluster. 
                                           Only the first file in each cluster is used for keyword extraction to minimize latency. Defaults to True.
            use_vectorizer (bool, optional): If use keyphrase-vectorizers as vectorizer. Defaults to True. 
                                             If set to True, keyphrase_ngram_range is not used.
                                             Details of keyphrase-vectorizers in https://pypi.org/project/keyphrase-vectorizers/. 
            keyphrase_ngram_range (tuple[int, int], optional): Length, in words, of the extracted keywords/keyphrases. Defaults to (1,1).
                                                               NOTE: This is not used if you passed a `vectorizer`. 

        Returns:
            Union[set, list]: The top n keywords for the documents 
        """
        # retrieve keywords
        vectorizer = None
        if use_vectorizer:
            vectorizer = KeyphraseTfidfVectorizer()
            keyphrase_ngram_range = None

        if use_clusters and self.use_bert:
            _, keywords = self.kw_llm_model.extract_keywords(docs=self.docs, embeddings=torch.as_tensor(self.docs_embed), threshold=.9,use_maxsum=True,nr_candidates=20, top_n=5, vectorizer=vectorizer, keyphrase_ngram_range=keyphrase_ngram_range)
            self.keywords = self.extract_first_values(keywords, return_list=False)
        elif use_clusters and self.use_llm:
            _, self.keywords = self.kw_llm_model.extract_keywords(docs=self.docs, embeddings=torch.as_tensor(self.docs_embed), threshold=.9)
        elif not use_clusters and self.use_bert:
            keywords = self.kw_bert_model.extract_keywords(docs[:1], keyphrase_ngram_range=keyphrase_ngram_range, use_maxsum=True,nr_candidates=20, top_n=5, vectorizer=vectorizer)
            self.keywords = self.extract_first_values(keywords, return_list=False)
        elif not use_clusters and self.use_llm:
            self.keywords = self.text_generator.extract_keywords(docs=self.docs, threshold=.9)
        return self.keywords

    def save_keywords(self, save_filepath: str) -> None:
        """
        Save keywords to local path.

        Args:
            save_filepath (str): The file path to save keywords.
        """
        # save keywords
        with open(save_filepath, "wb") as file:
            pickle.dump(self.keywords, file)


if __name__ == "__main__":
    # 1. load docs
    file_folder = kit_dir + "/data"
    docs = read_files(file_folder, extension=".txt")
    
    # 2. extract keywords
    CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
    kw_etr = KeywordExtractor(configs=CONFIG_PATH, docs=docs, use_bert=True)
    kw_etr.docs_embedding()
    keywords = kw_etr.extract_keywords()

    # 3. save keywords to local 
    save_filepath="./keywords/keywords.pkl"
    kw_etr.save_keywords(save_filepath)