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
        fastapi_url = os.environ.get("FASTAPI_URL")
        fastapi_api_key = os.environ.get("FASTAPI_API_KEY")

        self.llm = APIGateway.load_llm(
                    type=self.configs["router"]["type"],
                    streaming=False,
                    # coe=self.configs['router']["coe"],
                    do_sample=self.configs['router']["do_sample"],
                    max_tokens_to_generate=self.configs['router']["max_tokens_to_generate"],
                    temperature=self.configs['router']["temperature"],
                    select_expert=self.configs['router']["select_expert"],
                    process_prompt=False,
                    fastapi_url=fastapi_url,
                    fastapi_api_key=fastapi_api_key
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

        # load docs keywords 
        self.keywords = read_keywords(repo_dir + '/' + self.configs['keywords'])

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

    def init_response_router(self) -> None:
        response_router_prompt_template = load_prompt(os.path.join(repo_dir, self.configs['prompts']['ouput_router_prompt']))
        response_router_response_schemas = [
            ResponseSchema(name="reroute", description="true/false value that determines whether query needs to be rerouted")]

        response_router_output_parser = StructuredOutputParser.from_response_schemas(response_router_response_schemas)
        response_router_format_instructions = response_router_output_parser.get_format_instructions()

        response_router_prompt = PromptTemplate(input_variables=["response"], template=response_router_prompt_template.template,
                               partial_variables={"format_instructions": response_router_format_instructions})

        self.response_router = response_router_prompt | self.llm | response_router_output_parser 

    def check_response(self, query, response, datasource) -> str:
        decision = self.response_router.invoke({"datasource": datasource,
                                     "question": query,
                                     "response": response})
        reroute = str(decision["reroute"]).lower()
        return reroute




