import os
import shutil
from pathlib import Path
from typing import Optional, Union


def clear_directory(directory: Union[str, Path]) -> Optional[str]:
    """
    Clears all contents of the specified directory, including all files
    and subdirectories.

    Args:
        directory (str): The path to the directory to clear.

    Returns:
        Optional[str]: An error message if the directory does not exist or
                       is not a directory, otherwise None if the operation
                       is successful.

    Raises:
        OSError: If an error occurs while deleting files or directories.
    """
    if not os.path.exists(directory):
        return f'Directory {directory} does not exist.'

    if not os.path.isdir(directory):
        return f'The path {directory} is not a directory.'

    try:
        # Iterate over the contents of the directory
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            # Remove directory or file
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove subdirectory
            else:
                os.remove(item_path)  # Remove file
    except OSError as e:
        raise OSError(f'Error while clearing directory: {e}')

    return None
