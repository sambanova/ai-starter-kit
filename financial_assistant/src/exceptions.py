"""
Main exceptions for the finance assistant.
"""

from typing import Optional


class LLMException(Exception):
    """
    Exception raised when the Large Language Model (LLM) is not working.

    Attributes:
        message (str): The error message.
        error_code (int): The error code associated with the exception.
    """

    def __init__(self, message: str, error_code: Optional[int] = None) -> None:
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
        """
        Returns a string representation of the exception.

        Returns:
            str: A string representation of the exception.
        """
        if self.error_code is not None:
            return f'LLMException(error_code={self.error_code}, message="{self.message}")'
        else:
            return f'LLMException(message="{self.message}")'

    def __repr__(self) -> str:
        """
        Returns a string representation of the exception that can be used to recreate it.

        Returns:
            str: A string representation of the exception that can be used to recreate it.
        """
        if self.error_code is not None:
            return f'LLMException(message="{self.message}", error_code={self.error_code})'
        else:
            return f'LLMException(message="{self.message}")'
