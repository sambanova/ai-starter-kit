"""
Module for defining custom tools used in the educational content generation system.

This module provides base classes and implementations for custom tools that can be
used by AI agents in the content generation process.
"""

from typing import Type

from crewai_tools import BaseTool
from pydantic import BaseModel, Field


class MyCustomToolInput(BaseModel):
    """
    Input schema for MyCustomTool.

    This class defines the expected input structure for the custom tool,
    ensuring proper type validation and documentation.

    Attributes:
        argument (str): Description of the argument.
    """

    argument: str = Field(..., description='Description of the argument.')


class MyCustomTool(BaseTool):
    """
    Custom tool implementation for specific content generation tasks.

    This class provides a template for implementing custom tools that can be
    used by AI agents in the content generation workflow.

    Attributes:
        name (str): The name of the tool.
        description (str): Detailed description of the tool's purpose and usage.
        args_schema (Type[BaseModel]): The input schema class for the tool.

    Example:
        ```python
        tool = MyCustomTool()
        result = tool.run(argument="example input")
        ```
    """

    name: str = 'Name of my tool'
    description: str = (
        'Clear description for what this tool is useful for, you agent will need this information to use it.'
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        """
        Execute the tool's main functionality.

        This method implements the core functionality of the custom tool.
        Override this method to implement specific tool behavior.

        Args:
            argument (str): The input argument for the tool.

        Returns:
            str: The result of the tool's execution.

        Note:
            This is a placeholder implementation. Subclasses should override
            this method with actual functionality.
        """
        return 'this is an example of a tool output, ignore it and move along.'
