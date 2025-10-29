import os
from typing import Any, List, Optional, Tuple, Union

import yaml
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool, Tool
from langchain_sambanova import ChatSambaNova
from pydantic import BaseModel, SecretStr

from financial_assistant.prompts.function_calling_prompts import FUNCTION_CALLING_PROMPT_TEMPLATE
from financial_assistant.src.utilities import get_logger, time_llm

logger = get_logger()

# Max number of retries for tool invocation
MAX_RETRIES = 3


class SambaNovaLLM:
    """A class for initializing and managing a Large Language Model (LLM) and performing function calls."""

    def __init__(
        self,
        config_path: str,
        tools: Optional[Union[BaseTool, Tool, StructuredTool, List[Union[BaseTool, Tool, StructuredTool]]]] = None,
        default_tool: Optional[BaseTool | Tool | StructuredTool | type[BaseModel]] = None,
        system_prompt: Optional[str] = FUNCTION_CALLING_PROMPT_TEMPLATE,
        sambanova_api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            config_path: The path to the config file.
            tools: The optional tools to use.
            default_tool: The optional default tool to use. Defaults to `ConversationalResponse`.
            system_prompt: The optional system prompt to use. Defaults to `FUNCTION_CALLING_SYSTEM_PROMPT`.
            sambanova_api_key: The optional sambanova api key for authentication.

        Raises:
            TypeError: If `tools` is not a list of
                `langchain_core.tools.StructuredTool` or `langchain_core.tools.Tool` objects.
            TypeError: If `default_tool` is not a `langchain_core.tools.StructuredTool` object.
            TypeError: If `system_prompt` is not a string.
        """
        # Load the configs from the config file
        self.llm_info = self.get_llm_config_info(config_path)

        # Check the LLM information
        self.check_llm_info()

        # Set the LLM
        self.llm = self.set_llm(sambanova_api_key=sambanova_api_key)

        # Set the tools
        if tools is not None and not isinstance(tools, list):
            tools = [tools]
        self._tools = tools

        # Set the system prompt
        if not isinstance(system_prompt, str):
            raise TypeError('System prompt must be a string.')
        self.system_prompt = system_prompt

    @property
    def tools(self) -> Optional[Union[BaseTool, Tool, StructuredTool, List[Union[BaseTool, Tool, StructuredTool]]]]:
        """Getter method for tools."""

        return self._tools

    @tools.setter
    def tools(
        self,
        tools: Optional[Union[BaseTool, Tool, StructuredTool, List[Union[BaseTool, Tool, StructuredTool]]]] = None,
        default_tool: Optional[BaseTool | Tool | StructuredTool | type[BaseModel]] = None,
    ) -> None:
        """Setter method for tools."""

        # Set the list of tools to use
        if isinstance(tools, Tool) or isinstance(tools, StructuredTool):
            tools = [tools]
        if tools is not None:
            if not (
                isinstance(tools, list)
                and all(isinstance(tool, StructuredTool) or isinstance(tool, Tool) for tool in tools)
            ):
                raise TypeError('tools must be a list of StructuredTool or Tool objects.')
        self._tools = tools

        # Set the tools schemas
        if default_tool is not None:
            if not isinstance(default_tool, (StructuredTool, Tool, type(BaseModel))):
                raise TypeError('Default tool must be a StructuredTool.')

    def get_llm_config_info(self, config_path: str) -> Any:
        """
        Loads the json config file.

        Args:
            config_path: Path to the config json file.

        Returns:
            A tuple of dictionaries containing the llm information as a single element.

        Raises:
            TypeError: If `config_path` not found or `config_path` is not a string.
        """
        if not isinstance(config_path, str):
            raise TypeError('Config path must be a string.')

        # Read config file
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)

        # Get the llm information
        llm_info = config['llm']

        return llm_info

    def check_llm_info(self) -> None:
        """Check the llm information."""

        if not isinstance(self.llm_info, dict):
            raise TypeError('LLM information must be a dictionary.')
        if not all(isinstance(key, str) for key in self.llm_info):
            raise TypeError('LLM information keys must be strings')

        if not isinstance(self.llm_info['model'], str):
            raise TypeError('LLM `model` must be a string.')
        if not isinstance(self.llm_info['max_tokens'], int):
            raise TypeError('LLM `max_tokens` must be an integer.')
        if not isinstance(self.llm_info['temperature'], float):
            raise TypeError('LLM `temperature` must be a float.')

    def set_llm(self, sambanova_api_key: Optional[str] = None) -> ChatSambaNova:
        """
        Set the LLM to use.

        Returns:
            The LLM to use.

        Raises:
            TypeError: If the LLM API parameters are not of the expected type.
        """
        # Get the Sambanova API key
        if sambanova_api_key is None:
            sambanova_api_key = os.getenv('SAMBANOVA_API_KEY')
        assert sambanova_api_key is not None
        # Instantiate the LLM
        llm = ChatSambaNova(
            **self.llm_info,
            api_key=SecretStr(sambanova_api_key),
        )
        return llm

    def invoke_tools(self, query: str) -> Any:
        """
        Invocation method for the function calling workflow.

        Bind the relevant tools and execute them with the given query.

        Args:
            query: The query to execute.

        Returns:
            The LLM response, resulting from the execution of the relevant tool.

        Raises:
            TypeError: If `query` is not of type `str`.
        """
        # Checks the inputs
        if not isinstance(query, str):
            raise TypeError(f'Query must be a string. Got {type(query)}.')

        # Bind the tools to the LLM
        llm_with_tools, messages, ai_message = self.bind_tools(query)

        # Invoke the tools
        if self._tools is not None:
            for tool_call in ai_message.tool_calls:
                for i in range(MAX_RETRIES):
                    try_count = 0
                    # Extract the tool to call
                    selected_tool = {tool.name: tool for tool in self._tools}[tool_call['name'].lower()]
                    try:
                        # Invoke the tool
                        answer = selected_tool.invoke(tool_call['args'])
                        if answer is not None:
                            messages.append(ToolMessage(answer, tool_call_id=tool_call['id']))
                            break
                    except:
                        continue

        # Conversational response
        if self._tools is None or 'conversational' in ai_message.tool_calls[-1]['name'].lower():
            conversational_answer = llm_with_tools.invoke(messages).content
            return conversational_answer
        else:
            try:
                return answer
            except:
                raise Exception('The tools could not be invoked successfully.')

    @time_llm
    def bind_tools(
        self, query: str
    ) -> Tuple[Runnable[LanguageModelInput, BaseMessage], List[BaseMessage], BaseMessage]:
        """
        Bind the relevant tools to be invoked based on the query.

        Args:
            query: The query to be used.

        Returns:
            A tuple with the following elements:
                - The LLM with tools if tools are available to call, otherwise the simple LLM.
                - A list of messages.
                - The AI message.
        """
        # Initialize the list of messages with the Human Message of the query
        messages: List[BaseMessage] = [HumanMessage(query)]

        if self._tools is not None:
            #  Bind the tools to the LLM and retrieve the tool call arguments
            llm_with_tools = self.llm.bind_tools(tools=self._tools)
            ai_message = llm_with_tools.invoke(messages, tool_choice='required')
        else:
            # Use the LLM without tools
            llm_with_tools = self.llm
            ai_message = self.llm.invoke(messages)

        # Append the AI Message to the list of messages
        messages.append(ai_message)

        return llm_with_tools, messages, ai_message
