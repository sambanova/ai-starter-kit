import copy
import json
import logging
import os
from functools import wraps
from typing import Any, Callable

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_method(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to log method calls and save to JSON file.

    This decorator logs the method call, including the method name, class name,
    method arguments, keyword arguments, and the result of the method. The log
    information is saved to a JSON file.

    Args:
        func: The method to be decorated.

    Returns:
        The decorated method.
    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Wrapper function to log method calls and save to JSON file.

        This function logs the method call, including the method name, class name,
        method arguments, keyword arguments, and the result of the method. The log
        information is saved to a JSON file.

        Args:
            self: The instance of the class.
            *args: The method arguments.
            **kwargs: The method keyword arguments.

        Returns:
            The result of the method.
        """
        # Log method call
        logger.info(f"Method '{func.__name__}' called on {self.__class__.__name__}")

        result = func(self, *args, **kwargs)

        serializable_result = {}
        serializable_args = copy.deepcopy(args)

        # Handle JSON serialization errors from Langchain Documents object
        if isinstance(result, dict) and result.get('documents'):
            docs = result['documents']
            serializable_result['documents'] = '\n\n'.join(doc.page_content for doc in docs)
        else:
            serializable_result = result

        # Handle JSON serialization errors from Langchain Documents object
        if 'documents' in args[0] and args[0]['documents'] is not None:
            docs = args[0]['documents']
            serializable_args[0]['documents'] = '\n\n'.join(doc.page_content for doc in docs)
        else:
            serializable_args = args

        # Save log info to JSON file
        log_data = {
            'method': func.__name__,
            'class': self.__class__.__name__,
            'args': serializable_args,
            'kwargs': kwargs,
            'result': serializable_result,
        }

        log_dir = './../logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(f'{log_dir}/{func.__name__}.json', 'a') as f:
            json.dump(log_data, f)
            f.write('\n')

        return result

    return wrapper
