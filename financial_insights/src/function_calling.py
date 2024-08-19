import json
import logging
import re
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import yaml
from langchain_community.llms.sambanova import SambaStudio, Sambaverse
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool, Tool

import pydantic

from financial_insights.streamlit.constants import *

# Prompt template for function calling
FUNCTION_CALLING_SYSTEM_PROMPT = """
You are a helpful assistant and you have access to the following tools:

{tools}

You must always select one or more of the above tools and answer 
with only a list of JSON objects matching the following schema:

```json
[{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}]
```

If one tool is called after another, the subsequent tool must follow the previous tool in the list.
Please call all the relevant tools.

Your answer should be in the same language as the initial query.
"""


# Tool schema for the conversational response tool
class ConversationalResponse(BaseModel):
    """Respond conversationally if you have a final answer. Response must be in the same language as the user query"""

    response: str = Field(
        ..., description='Conversational response to the user. Must be in the same language as the user query.'
    )


class FunctionCalling:
    """Class for function calling."""

    def __init__(
        self,
        tools: Optional[Union[StructuredTool, Tool, List[Union[StructuredTool, Tool]]]] = None,
        default_tool: Optional[StructuredTool | Tool | type[BaseModel]] = ConversationalResponse,
        system_prompt: Optional[str] = FUNCTION_CALLING_SYSTEM_PROMPT,
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
        configs = self.get_config_info(config_path)

        # Set the llm information
        self.llm_info = configs[0]

        # Set the LLM
        self.llm = self.set_llm()

        # Set the list of tools to use
        if isinstance(tools, Tool) or isinstance(tools, StructuredTool):
            tools = [tools]
        assert (isinstance(tools, list) and all(
            isinstance(tool, StructuredTool) or isinstance(tool, Tool) for tool in tools
        )) or tools is None, TypeError('tools must be a list of StructuredTool or Tool objects.')
        self.tools = tools

        # Set the tools schemas
        assert isinstance(default_tool, (StructuredTool, Tool, type(BaseModel))) or default_tool is None, TypeError('Default tool must be a StructuredTool.')
        tools_schemas = self.get_tools_schemas(tools, default=default_tool)
        self.tools_schemas = '\n'.join([json.dumps(tool, indent=2) for tool in tools_schemas])

        # Set the system prompt
        assert isinstance(system_prompt, str), TypeError('System prompt must be a string.')
        self.system_prompt = system_prompt

    def get_config_info(self, config_path: str) -> Tuple[Dict[str, str]]:
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

        return (llm_info,)

    def set_llm(self) -> Union[SambaStudio, Sambaverse]:
        """
        Set the LLM to use

        SambaVerse, SambaStudio and CoE endpoints implemented.

        Returns:
            The LLM to use.

        Raises:
            ValueError: If the LLM API is not one of `sambastudio` or `sambaverse`.
        """
        # SambaStudio LLM
        if self.llm_info['api'] == 'sambastudio':
            if self.llm_info['coe']:
                llm = SambaStudio(
                    streaming=True,
                    model_kwargs={
                        'max_tokens_to_generate': self.llm_info['max_tokens_to_generate'],
                        'select_expert': self.llm_info['select_expert'],
                        'temperature': self.llm_info['temperature'],
                    },
                )
            else:
                llm = SambaStudio(
                    model_kwargs={
                        'max_tokens_to_generate': self.llm_info['max_tokens_to_generate'],
                        'temperature': self.llm_info['temperature'],
                    },
                )

        # SambaVerse LLM
        elif self.llm_info['api'] == 'sambaverse':
            llm = Sambaverse(  # type:ignore
                sambaverse_model_name=self.llm_info['sambaverse_model_name'],
                model_kwargs={
                    'max_tokens_to_generate': self.llm_info['max_tokens_to_generate'],
                    'select_expert': self.llm_info['select_expert'],
                    'temperature': self.llm_info['temperature'],
                },
            )
        else:
            raise ValueError(
                f"Invalid LLM API: {self.llm_info['api']}, only 'sambastudio' and 'sambaverse' are supported."
            )
        return llm

    def get_tools_schemas(
        self,
        tools: Optional[Union[StructuredTool, Tool, List[Union[StructuredTool, Tool]]]] = None,
        default: Optional[Union[StructuredTool, Tool, Type[BaseModel]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Get the tools schemas.

        Args:
            tools: The tools to use.
            default: The default tool to use.

        Returns:
            The list of tools schemas, where each tool schema is a dictionary with the following keys:
                - `name`: The tool name.
                - `description`: The tool description.
                - `properties`: The tool properties.
                - `required`: The tool required properties.

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
                    'properties': tool_schema['properties'],
                }
                if 'required' in schema:
                    schema['required'] = tool_schema['required']
                tools_schemas.append(schema)

        # Get the default tools schemas
        if default is not None:
            if isinstance(default, Tool) or isinstance(default, StructuredTool):
                tool_schema = default.get_input_schema().schema()
            elif issubclass(default, BaseModel):
                tool_schema = default.schema()
            schema = {
                'name': tool_schema['title'],
                'description': tool_schema['description'],
                'properties': tool_schema['properties'],
            }
            if 'required' in schema:
                schema['required'] = tool_schema['required']
            tools_schemas.append(schema)

        return tools_schemas

    def execute(self, invoked_tools: List[Dict[str, Union[str, Dict[str, str]]]]) -> Tuple[List[str], Any]:
        """
        Execute the tools provided in a list of tools.

        If the list only contains the default conversational one, the response is marked as final response.

        Args:
            invoked_tools: The list of tools provided by the LLM.

        Returns:
            A tuple of the following elements:
                - A list of tool messages. Each message contains the tool name and the response of the tool.
                - The final response of the tool chain, i.e the response of the last tool.

        Raises:
            TypeError: If any of the tool names (`invoked_tools`) is not a string.
        """
        # Create a map of tools with their name
        if self.tools is not None:
            tools_map = {tool.name: tool for tool in self.tools}
        else:
            tools_map = {}

        # Tool message template
        tool_msg = "Tool '{name}'. Response: {response}"
        tools_msgs = []

        assert all(isinstance(tool['tool'], str) for tool in invoked_tools), TypeError('The tool name must be a string')

        # If the only tool is the conversational response, return the response only
        if len(invoked_tools) == 1 and invoked_tools[0]['tool'].lower() == 'conversationalresponse':  # type: ignore
            return [invoked_tools[0]['tool_input']['response']], None  # type: ignore

        # Otherwise, invoke each tool and return both the responses and the messages
        for tool in invoked_tools:
            if tool['tool'].lower() != 'conversationalresponse':  # type: ignore
                response = tools_map[tool['tool'].lower()].invoke(tool['tool_input'])  # type: ignore
                tools_msgs.append(tool_msg.format(name=tool['tool'], response=str(response)))

        # All the messages, but only the last response will be returned
        return tools_msgs, response

    def jsonFinder(self, input_string: str) -> Optional[str]:
        """
        Find the json structures in a string response.

        If bad formatted, the json structure is corrected via the LLM.

        Args:
            input_string: The string containing the json structure.

        Returns:
            The optional corrected json structure.
        """
        # The regex json pattern
        json_pattern = re.compile(r'(\{.*\}|\[.*\])', re.DOTALL)

        # Find the first JSON structure in the string
        json_match = json_pattern.search(input_string)

        # If a JSON structure was found, return it as a string
        if json_match is not None:
            json_str = json_match.group(1)
            try:
                json.loads(json_str)
            except:
                json_correction_prompt = """|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a json format corrector tool<|eot_id|><|start_header_id|>user<|end_header_id|>
                fix the following json file: {json} 
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                fixed json: """  # noqa E501
                json_correction_prompt_template = PromptTemplate.from_template(json_correction_prompt)
                json_correction_chain = json_correction_prompt_template | self.llm
                json_str = json_correction_chain.invoke({'json': json_str})
        else:
            # If no JSON structure was found, return None
            json_str = None
            logging.warning('No tool json structure found for function calling.')
        return json_str

    def msgs_to_llama3_str(self, messages: List[BaseMessage]) -> str:
        """
        Convert a list of langchain messages with roles to expected LLmana 3 input.

        Args:
            messages: The list of langchain messages.

        Returns:
            The LLM input string.

        Raises:
            ValueError: If the input `messages` is not a list of langchain messages with supported roles (types).
                The only supported message types are `system`, `human`, `ai`, `tool`.
        """
        formatted_messages = []
        for message in messages:
            if message.type == 'system':
                sys_placeholder = (
                    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>system<|end_header_id|> {msg}'
                )
                formatted_messages.append(sys_placeholder.format(msg=message.content))
            elif message.type == 'human':
                human_placeholder = '<|eot_id|><|start_header_id|>user<|end_header_id|>\nUser: {msg} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\nAssistant:'  # noqa E501
                formatted_messages.append(human_placeholder.format(msg=message.content))
            elif message.type == 'ai':
                assistant_placeholder = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nAssistant: {msg}'
                formatted_messages.append(assistant_placeholder.format(msg=message.content))
            elif message.type == 'tool':
                tool_placeholder = '<|eot_id|><|start_header_id|>tools<|end_header_id|>\n{msg} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\nAssistant:'  # noqa E501
                formatted_messages.append(tool_placeholder.format(msg=message.content))
            else:
                raise ValueError(f'Invalid message type: {message.type}')
        return '\n'.join(formatted_messages)

    def function_call_llm(self, query: str, debug: bool = False) -> Tuple[List[str], Any]:
        """
        Invocation method for function calling workflow.

        Args:
            query: The query to execute.
            debug: Whether to print debug information. Defaults to False.

        Returns:
            A tuple containing the following elements:
                - The generated tool messages.
                - The final LLM response, i.e. the response of the last invocated tool.

        Raises:
            TypeError: If `query` is not of type str or `debug` is not of type bool.
        """
        # Checks the inputs
        assert isinstance(query, str), TypeError(f'Query must be a string. Got {type(query)}.')
        assert isinstance(debug, bool), TypeError(f'Debug must be a boolean. Got {type(debug)}.')

        # Prompt template for function calling
        function_calling_template = ChatPromptTemplate.from_messages([('system', self.system_prompt)])

        # Prompt for function calling
        function_calling_prompt = function_calling_template.format_prompt(tools=self.tools_schemas).to_messages()

        # Append the query to the prompt
        function_calling_prompt.append(HumanMessage(query))

        # Prompt for the relevant tools
        prompt = self.msgs_to_llama3_str(function_calling_prompt)

        # Invoke the LLM to find the relevant tools
        llm_response = self.llm.invoke(prompt)

        # Chain to parse the LLM response and find the json structure of the tools
        json_parsing_chain = RunnableLambda(self.jsonFinder) | JsonOutputParser()

        # Parse the LLM response to find the json structure of the proposed tools
        parsed_tools_llm_response = json_parsing_chain.invoke(llm_response)

        # Execute the proposed tools and extract the tool messages and the final response
        tools_messages, response = self.execute(parsed_tools_llm_response)

        # Debugging
        if debug:
            pprint(function_calling_prompt)

        return tools_messages, response
