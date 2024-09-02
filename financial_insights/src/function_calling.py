import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit
import yaml
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import StructuredTool, Tool

from financial_insights.prompts.function_calling_prompts import FUNCTION_CALLING_PROMPT_TEMPLATE
from financial_insights.src.tools import time_llm
from financial_insights.streamlit.constants import *
from utils.model_wrappers.api_gateway import APIGateway


class FunctionCalling:
    """Class for function calling."""

    def __init__(
        self,
        tools: Optional[Union[StructuredTool, Tool, List[Union[StructuredTool, Tool]]]] = None,
        default_tool: Optional[StructuredTool | Tool | type[BaseModel]] = None,
        system_prompt: Optional[str] = FUNCTION_CALLING_PROMPT_TEMPLATE,
        config_path: str = CONFIG_PATH,
    ) -> None:
        """
        Args:
            tools: The optional tools to use.
            default_tool: The optional default tool to use. Defaults to `ConversationalResponse`.
            system_prompt: The optional system prompt to use. Defaults to `FUNCTION_CALLING_SYSTEM_PROMPT`.
            config_path: The path to the config file. Defaults to `CONFIG_PATH`.

        Raises:
            TypeError: If `tools` is not a list of
                `langchain_core.tools.StructuredTool` or `langchain_core.tools.Tool` objects.
            TypeError: If `default_tool` is not a `langchain_core.tools.StructuredTool` object.
            TypeError: If `system_prompt` is not a string.
        """
        # Load the configs from the config file
        llm_info, prod_mode_info = self.get_llm_config_info(config_path)

        # Set the production flag
        self.prod_mode = prod_mode_info

        # Set the llm information
        self.llm_info = llm_info

        # Set the LLM
        self.llm = self.set_llm()

        # Set the list of tools to use
        if isinstance(tools, Tool) or isinstance(tools, StructuredTool):
            tools = [tools]
        assert (
            isinstance(tools, list)
            and all(isinstance(tool, StructuredTool) or isinstance(tool, Tool) for tool in tools)
        ) or tools is None, TypeError('tools must be a list of StructuredTool or Tool objects.')
        self.tools = tools

        # Set the tools schemas
        assert isinstance(default_tool, (StructuredTool, Tool, type(BaseModel))) or default_tool is None, TypeError(
            'Default tool must be a StructuredTool.'
        )
        tools_schemas = self.get_tools_schemas(tools)
        self.tools_schemas = '\n'.join([json.dumps(tool, indent=2) for tool in tools_schemas])

        # Set the system prompt
        assert isinstance(system_prompt, str), TypeError('System prompt must be a string.')
        self.system_prompt = system_prompt

    def get_llm_config_info(self, config_path: str) -> Tuple[Any, Any]:
        """
        Loads the json config file.

        Args:
            config_path: Path to the config json file.

        Returns:
            A tuple of dictionaries containing the llm information as a single element.

        Raises:
            TypeError: If `config_path` not found or `config_path` is not a string.
        """
        assert isinstance(config_path, str), TypeError('Config path must be a string.')

        # Read config file
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)

        # Get the llm information
        llm_info = config['llm']

        prod_mode_info = config['prod_mode']

        return (llm_info, prod_mode_info)

    def set_llm(self) -> LLM:
        """
        Set the LLM to use.

        SambaVerse, SambaStudio, FastAPI endpoints are implemented.

        Returns:
            The LLM to use.

        Raises:
            ValueError: If the LLM API is not one of `sambastudio`, `sambaverse`, or `fastapi`.
        """
        if self.llm_info['api'] in ['sambastudio', 'sambaverse', 'fastapi']:
            # Check config parameters
            assert isinstance(self.llm_info['api'], str), ValueError(
                'LLM API must be one of `sambastudio`, `sambaverse`, or `fastapi`.'
            )
            assert isinstance(self.llm_info['coe'], bool), TypeError('Sambaverse `coe` must be a boolean.')
            assert isinstance(self.llm_info['do_sample'], bool), TypeError('`do_sample` must be a boolean.')
            assert isinstance(self.llm_info['max_tokens_to_generate'], int), TypeError(
                '`max_tokens_to_generate` must be an integer.'
            )
            assert isinstance(self.llm_info['select_expert'], str), TypeError('`select_expert` must be a string.')
            assert isinstance(self.llm_info['sambaverse_model_name'], str | None), TypeError(
                'Sambaverse `model_name` must be a string.'
            )

            # Get the LLM credentials following `prod_mode`
            if self.prod_mode:
                fastapi_url = streamlit.session_state.FASTAPI_URL
                fastapi_api_key = streamlit.session_state.FASTAPI_API_KEY
            else:
                fastapi_url = os.environ.get('FASTAPI_URL') or streamlit.session_state.FASTAPI_URL
                fastapi_api_key = os.environ.get('FASTAPI_API_KEY') or streamlit.session_state.FASTAPI_API_KEY

            # Instantiate the LLM
            llm = APIGateway.load_llm(
                type=self.llm_info['api'],
                streaming=False,
                coe=self.llm_info['coe'],
                do_sample=self.llm_info['do_sample'],
                max_tokens_to_generate=self.llm_info['max_tokens_to_generate'],
                temperature=self.llm_info['temperature'],
                select_expert=self.llm_info['select_expert'],
                process_prompt=False,
                sambaverse_model_name=self.llm_info['sambaverse_model_name'],
                fastapi_url=fastapi_url,
                fastapi_api_key=fastapi_api_key,
            )
        else:
            raise ValueError(
                f"Invalid LLM API: {self.llm_info['api']}, only 'sambastudio' and 'sambaverse' are supported."
            )
        return llm

    def get_tools_schemas(
        self,
        tools: Optional[Union[StructuredTool, Tool, List[Union[StructuredTool, Tool]]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Get the tools schemas.

        Args:
            tools: The tools to use.

        Returns:
            The list of tools schemas, where each tool schema is a dictionary with the following keys:
                - `name`: The tool name.
                - `description`: The tool description.
                - `parameters`: The tool parameters.

        Raises:
            TypeError: If `tools` is not a `langchain_core.tools.Tool` or a list of `langchain_core.tools.Tools`.
        """
        if tools is None or isinstance(tools, list):
            pass
        elif isinstance(tools, Tool) or isinstance(tools, StructuredTool):
            tools = [tools]
        else:
            raise TypeError('tools must be a Tool or a list of Tools')

        # Get the tools schemas
        tools_schemas = []
        if tools is not None:
            for tool in tools:
                tool_schema = tool.get_input_schema().schema()
                schema = {
                    'name': tool.name,
                    'description': tool_schema['description'],
                    'parameters': tool_schema['properties'],
                }
                tools_schemas.append(schema)

        return tools_schemas

    def invoke_tools(self, query: str) -> Any:
        """
        Invocation method for the function calling workflow.

        Find the relevant tools and execute them with the given query.

        Args:
            query: The query to execute.

        Returns:
            The LLM response, resulting from the exeecution of the relevant tool.

        Raises:
            TypeError: If `query` is not of type str.
        """
        # Checks the inputs
        assert isinstance(query, str), TypeError(f'Query must be a string. Got {type(query)}.')

        # Find the relevant tool
        invoked_tool = self.find_relevant_tool(query)

        # Extract the tool parameters
        tool_name = invoked_tool['name']
        tool_parameters = invoked_tool['parameters']

        # Create a map of tools with their names
        if self.tools is not None:
            tools_map = {tool.name: tool for tool in self.tools}
        else:
            tools_map = {}

        # Invoke the tool with the retrieved inputs
        answer = tools_map[tool_name].invoke(tool_parameters)

        return answer

    @time_llm
    def find_relevant_tool(self, query: str) -> Any:
        """
        Find the relevant tool to be invoked based on the query.

        Args:
            query: The query to be used.

        Returns:
            The relevant tool to be invoked based on the query.
        """
        # JSON output parser
        function_calling_parser = JsonOutputParser()

        # Prompt template for function calling
        function_calling_prompt = PromptTemplate(
            template=FUNCTION_CALLING_PROMPT_TEMPLATE,
            input_variables=['tools', 'user_query'],
            partial_variables={'format_instructions': function_calling_parser.get_format_instructions()},
        )

        # Chain for function calling
        chain_function_calling = function_calling_prompt | self.llm | function_calling_parser

        # Invoke the LLM to find the relevant tools
        for i in range(MAX_RETRIES):
            try:
                invoked_tools = chain_function_calling.invoke({'tools': self.tools_schemas, 'user_query': query})
                break
            except:
                continue

        assert len(invoked_tools) == 1, f'Expected one tool, got {len(invoked_tools)}.'

        invoked_tool = invoked_tools[0]

        return invoked_tool
