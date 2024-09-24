### Router
import os, sys, pickle, yaml
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from utils.model_wrappers.api_gateway import APIGateway
from typing import Union
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(current_dir)
sys.path.append(repo_dir)
from keyword_extractor import KeywordExtractor
load_dotenv(os.path.join(kit_dir, '.env'))

def read_keywords(filepath: str) -> Union[set, list]:
    """
    Read keywords from local file path.

    Args:
        filepath (str): The path of the keyword file.

    Returns:
        set | list: the set/list of keywords.
    """
    with open(filepath, "rb") as file:
        keywords = pickle.load(file)
        return keywords

def read_files(directory: str, extension: str=".txt") -> list:
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
        raise NotADirectoryError(f"The directory {directory} doesn't exist!")
    file_contents = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_contents.append(file.read())
    return file_contents
  
class Router:
    def __init__(self, configs: str) -> None:
        """
        Initializes the router.

        Args:
            configs: The configuration file path.

        Returns:
            None
        """
        self.configs = self.load_config(configs)
        self.init_llm()
        self.keywords = None
        self.init_router()
    
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
    
    def init_llm(self) -> None:
        """
        Initializes the Large Language Model (LLM) based on the specified API.

        Args:
            self: The instance of the class.

        Returns:
            None
        """     
        # 1. load models
        sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')

        self.llm = APIGateway.load_llm(
                    type=self.configs["router"]["type"],
                    streaming=False,
                    coe=self.configs['router']["coe"],
                    do_sample=self.configs['router']["do_sample"],
                    max_tokens_to_generate=self.configs['router']["max_tokens_to_generate"],
                    temperature=self.configs['router']["temperature"],
                    select_expert=self.configs['router']["select_expert"],
                    process_prompt=False,
                    sambanova_api_key=sambanova_api_key,
                )

    def init_router(self) -> None:
        """
        Initializes the router.

        This method loads the router prompt and keywords, then combines it with the language
        model and a JSON output parser.

        Args:
            None

        Returns:
            None
        """
        # create prompt
        route_prompt = load_prompt(repo_dir + '/' + self.configs['prompts']['router_prompt'])

        # load/extract keywords for docs  
        keyword_filpath = os.path.join(repo_dir, self.configs['router']['keyword_path'])
        if os.path.isfile(keyword_filpath):
            self.keywords = read_keywords(keyword_filpath)
        else:
            document_path = os.path.join(repo_dir, self.configs['router']['document_folder'])
            self.extract_keyword(document_path, save_filepath=keyword_filpath)

        # create output parser
        response_schemas = [
            ResponseSchema(name="datasource", description="choose vectorstore or llm"),
            ResponseSchema(
                name="explanation",
                description="explain the reason to choose this datasource.",
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        # format prompt 
        format_instructions = output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template=route_prompt.template,
            input_variables=["query"],
            partial_variables={"format_instructions": format_instructions, "keywords": self.keywords},
        )

        # create LCEL
        self.router = prompt | self.llm | output_parser
        
    def routing(self, query: str) -> str:
        """
        Route the user query to either vectorstore or llm.

        Args:
            query (str): the user query

        Returns:
            str: "vectorstore" or "llm"
        """
        results = self.router.invoke({'query': query})
        return results["datasource"]

    def extract_keyword(self, file_folder: str, extension: str=".txt", save_filepath: str = None) -> None:
        """
        Extract keywords from documents.

        Args:
            file_folder (str): The folder contains the documents
            extension (str, optional): The extension of the files. Defaults to ".txt".
            save_filepath (str, optional): The file path to save the keywords. Defaults to None.
        """
        # load docs
        if os.path.isdir(file_folder):
            docs = read_files(file_folder, extension=extension)
        else:
            raise NotADirectoryError(f'{file_folder} is not a directory.')
        
        # extract keywords
        kw_etr = KeywordExtractor(configs=self.configs, 
                                  docs=docs, 
                                  use_bert=self.configs['router']['use_bert'], 
                                  use_llm=self.configs['router']['use_llm'])
        kw_etr.docs_embedding()
        self.keywords = kw_etr.extract_keywords(self.configs['router']['use_clusters'])
        
        # save keywords to local 
        if save_filepath:
            kw_etr.save_keywords(save_filepath)
    