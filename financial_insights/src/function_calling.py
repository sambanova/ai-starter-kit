import json
import os
import re
import sys
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import yaml
from dotenv import load_dotenv
from langchain_community.llms.sambanova import SambaStudio, Sambaverse
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool, Tool

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)


load_dotenv(os.path.join(repo_dir, '.env'))
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')


FUNCTION_CALLING_SYSTEM_PROMPT = """You are an helpful assistant and you have access to the following tools:

{tools}

You must always select one or more of the above tools and answer with only a list of JSON objects matching the following schema:

```json
[{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}]
```

If one tool is called after another, the subsequent tool must follow the previous tool in the list.
Please call all the relevant tool.

Your answer should be in the same language as the initial query.


"""  # noqa E501


# tool schema
class ConversationalResponse(BaseModel):
    (
        'Respond conversationally only if no other tools should be called for a given query, '
        'or if you have a final answer. response must be in the same language as the user query'
    )

    response: str = Field(
        ..., description='Conversational response to the user. Must be in the same language as the user query.'
    )


class FunctionCallingLlm:
    """
    function calling llm class
    """

    def __init__(
        self,
        tools: Optional[Union[StructuredTool, Tool, List[Union[StructuredTool, Tool]]]] = None,
        default_tool: Optional[Union[StructuredTool, Tool, Type[BaseModel]]] = ConversationalResponse,
        system_prompt: Optional[str] = None,
        config_path: str = CONFIG_PATH,
    ) -> None:
        """
        Args:
            tools (Optional[Union[StructuredTool, Tool, List[Union[StructuredTool, Tool]]]]): The tools to use.
            default_tool (Optional[Union[StructuredTool, Tool, Type[BaseModel]]]): The default tool to use.
                defaults to ConversationalResponse
            system_prompt (Optional[str]): The system prompt to use. defaults to FUNCTION_CALLING_SYSTEM_PROMPT
            config_path (str): The path to the config file. defaults to CONFIG_PATH
        """
        configs = self.get_config_info(config_path)
        self.llm_info = configs[0]
        self.llm = self.set_llm()
        if isinstance(tools, Tool) or isinstance(tools, StructuredTool):
            tools = [tools]
        self.tools = tools
        if system_prompt is None:
            self.system_prompt = FUNCTION_CALLING_SYSTEM_PROMPT
        tools_schemas = self.get_tools_schemas(tools, default=default_tool)
        self.tools_schemas = '\n'.join([json.dumps(tool, indent=2) for tool in tools_schemas])

    def get_config_info(self, config_path: str) -> tuple[Dict[str, str]]:
        """
        Loads json config file
        """
        # Read config file
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        llm_info = config['llm']

        return (llm_info,)

    def set_llm(self) -> Union[SambaStudio, Sambaverse]:
        """
        Set the LLM to use.
        sambaverse, sambastudio and  CoE endpoints implemented.
        """

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
            tools (Optional[Union[StructuredTool, Tool, list]]): The tools to use.
            default (Optional[Union[StructuredTool, Tool, Type[BaseModel]]]): The default tool to use.
        """
        if tools is None or isinstance(tools, list):
            pass
        elif isinstance(tools, Tool) or isinstance(tools, StructuredTool):
            tools = [tools]
        else:
            raise TypeError('tools must be a Tool or a list of Tools')

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

    def execute(self, invoked_tools: List[Dict[str, Union[str, Dict[str, str]]]]) -> tuple[List[str], Any]:
        """
        Given a list of tool executions the llm return as required
        execute them given the name with the mane in tools_map and the input arguments
        if there is only one tool call and it is default conversational one, the response is marked as final response

        Args:
            invoked_tools (List[dict]): The list of tool executions generated by the LLM.
        """
        if self.tools is not None:
            tools_map = {tool.name: tool for tool in self.tools}
        else:
            tools_map = {}
        tool_msg = "Tool '{name}'. Response: {response}"
        tools_msgs = []
        assert all(isinstance(tool['tool'], str) for tool in invoked_tools), 'The tool name must be a string'
        if len(invoked_tools) == 1 and invoked_tools[0]['tool'].lower() == 'conversationalresponse':  # type: ignore
            return [invoked_tools[0]['tool_input']['response']], None  # type: ignore
        for tool in invoked_tools:
            if tool['tool'].lower() != 'conversationalresponse':  # type: ignore
                response = tools_map[tool['tool'].lower()].invoke(tool['tool_input'])  # type: ignore
                tools_msgs.append(tool_msg.format(name=tool['tool'], response=str(response)))
        # The last response will be returned
        return tools_msgs, response

    def jsonFinder(self, input_string: str) -> Optional[str]:
        """
        find json structures ina  llm string response, if bad formatted using LLM to correct it

        Args:
            input_string (str): The string to find the json structure in.
        """
        json_pattern = re.compile(r'(\{.*\}|\[.*\])', re.DOTALL)
        # Find the first JSON structure in the string
        json_match = json_pattern.search(input_string)
        if json_match:
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
            # implement here not finding json format parsing to json or error rising
            json_str = None
        return json_str

    def msgs_to_llama3_str(self, msgs: List[BaseMessage]) -> str:
        """
        convert a list of langchain messages with roles to expected LLmana 3 input

        Args:
            msgs (list): The list of langchain messages.
        """
        formatted_msgs = []
        for msg in msgs:
            if msg.type == 'system':
                sys_placeholder = (
                    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>system<|end_header_id|> {msg}'
                )
                formatted_msgs.append(sys_placeholder.format(msg=msg.content))
            elif msg.type == 'human':
                human_placeholder = '<|eot_id|><|start_header_id|>user<|end_header_id|>\nUser: {msg} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\nAssistant:'  # noqa E501
                formatted_msgs.append(human_placeholder.format(msg=msg.content))
            elif msg.type == 'ai':
                assistant_placeholder = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nAssistant: {msg}'
                formatted_msgs.append(assistant_placeholder.format(msg=msg.content))
            elif msg.type == 'tool':
                tool_placeholder = '<|eot_id|><|start_header_id|>tools<|end_header_id|>\n{msg} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\nAssistant:'  # noqa E501
                formatted_msgs.append(tool_placeholder.format(msg=msg.content))
            else:
                raise ValueError(f'Invalid message type: {msg.type}')
        return '\n'.join(formatted_msgs)

    def function_call_llm(self, query: str, max_it: int = 5, debug: bool = False) -> Tuple[List[str], Any]:
        """
        invocation method for function calling workflow

        Args:
            query (str): The query to execute.
            max_it (int, optional): The maximum number of iterations. Defaults to 5.
            debug (bool, optional): Whether to print debug information. Defaults to False.
        """
        function_calling_chat_template = ChatPromptTemplate.from_messages([('system', self.system_prompt)])
        history = function_calling_chat_template.format_prompt(tools=self.tools_schemas).to_messages()
        history.append(HumanMessage(query))

        json_parsing_chain = RunnableLambda(self.jsonFinder) | JsonOutputParser()

        prompt = self.msgs_to_llama3_str(history)
        llm_response = self.llm.invoke(prompt)

        parsed_tools_llm_response = json_parsing_chain.invoke(llm_response)

        tools_messages, response = self.execute(parsed_tools_llm_response)

        if debug:
            pprint(history)
        return tools_messages, response
