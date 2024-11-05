"""
Main exceptions for the finance assistant.
"""

from typing import Optional

LLM_EXCEPTION_MESSAGE = (
    'We are experiencing an issue with the language model. '
    'Please try again in a few minutes or change your SAMBANOVA_API_KEY.'
)


class LLMException(Exception):
    """Exception raised when the Large Language Model (LLM) is not working."""

    def __init__(self, message: str = LLM_EXCEPTION_MESSAGE, error_code: Optional[int] = None) -> None:
        """
        Initializes the LLMException.

        Args:
            message (str): The error message.
            error_code (int, optional): The error code associated with the exception. Defaults to 500.
        """
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self) -> str:
        """Returns a string representation of the exception."""

        if self.error_code is not None:
            return f'LLMException(error_code={self.error_code}, message="{self.message}")'
        else:
            return f'LLMException(message="{self.message}")'

    def __repr__(self) -> str:
        """Returns a string representation of the exception that can be used to recreate it."""

        if self.error_code is not None:
            return f'LLMException(message="{self.message}", error_code={self.error_code})'
        else:
            return f'LLMException(message="{self.message}")'


class VectorStoreException(Exception):
    """Exception raised when the vector store cannot be retrieved."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return f'VectorStoreException({self.message})'


class TableNotFoundException(Exception):
    """Exception raised when no relevant tables are found in the SQL database."""

    def __init__(self, message: str = 'No relevant SQL tables found.') -> None:
        self.message = message

    def __str__(self) -> str:
        return f'TableNotFoundException({self.message})'


class ToolNotFoundException(Exception):
    """Exception raised when no tool can be found for a given tool name."""

    def __init__(self, message: str = 'The tool does not feature in the list of available tools.') -> None:
        self.message = message

    def __str__(self) -> str:
        return f'ToolNotFoundException({self.message})'
