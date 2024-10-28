import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd
from weave import Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(utils_dir, '..'))
sys.path.append(utils_dir)
sys.path.append(repo_dir)

from abc import ABC, abstractmethod


class DatasetConverter(ABC):
    """
    Abstract base class for dataset converters.

    This class provides a common interface for converting different types of datasets
    into a standardized format.
    """

    @abstractmethod
    def convert(self) -> List[Dict[str, str]]:
        """
        Converts the dataset to a list of dictionaries.

        This method must be implemented by any concrete subclass of DatasetConverter.
        It takes no arguments and returns a list of dictionaries, where each dictionary
        represents a single data point in the dataset.
        """
        pass


class JSONDatasetConverter(DatasetConverter):
    """
    Converter for JSON data.

    This class loads JSON data from a file and converts it into a list of dictionaries.
    """

    def __init__(self, path: str) -> None:
        self.json_data = self.load_data(path)

    def load_data(self, path: str) -> Any:
        """
        Load data from a JSON file.

        Args:
            path (str): The path to the JSON file to load.

        Returns:
            Any: The loaded JSON data.

        Raises:
            FileNotFoundError: If path of the file is not correct.
        """
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f'Error loading JSON data from {path}: {str(e)}')

    def convert(self) -> List[Dict[str, str]]:
        """
        Convert JSON data to a list of dictionaries.

        Returns:
            List[Dict[str, str]]: The converted JSON data.
        """
        return self.json_data


class DataFrameDatasetConverter(DatasetConverter):
    """
    Converter for CSV data.

    This class loads CSV data from a file and converts it into a list of dictionaries.
    """

    def __init__(self, filepath: str) -> None:
        self.dataframe = self.load_data(filepath)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            filepath (str): The path to the CSV file to load.

        Returns:
            pd.DataFrame: The loaded CSV data.

        Raises:
            FileNotFoundError: If path of the file is not correct.
        """
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError as e:
            raise ValueError(f'Error loading CSV data from {filepath}: {str(e)}')

    def convert(self) -> List[Dict[str, str]]:
        """
        Convert the DataFrame to a list of dictionaries.

        Returns:
            List[Dict[str, str]]: The converted DataFrame data.
        """
        return self.dataframe.to_dict(orient='records')


class DatasetConverterFactory:
    """
    Factory to create dataset converters based on the input type.

    This class provides a way to create dataset converters based on the file extension
    of the input data.
    """

    @staticmethod
    def create_converter(filepath: str) -> DatasetConverter:
        """
        Create a dataset converter based on the file extension.

         Args:
             filepath (str): The path to the input data file.

         Returns:
             DatasetConverter: The created dataset converter.

         Raises:
             ValueError: If the file type is not supported.
        """
        _, extension = os.path.splitext(filepath)
        extension = extension.lower()

        if extension == '.json':
            return JSONDatasetConverter(filepath)
        elif extension in ['.csv', '.txt']:
            return DataFrameDatasetConverter(filepath)
        else:
            raise ValueError('Unsupported file type. Please provide a JSON or CSV file.')


class WeaveDatasetManager:
    """
    Class to manage Weave datasets.

    This class provides a way to create Weave datasets from input data files.
    """

    def __init__(self) -> None:
        self._type = 'weave'

    def create_dataset(self, name: str, filepath: str) -> Dataset:
        """
         Create a Weave dataset from the input data file.

        Args:
            name (str): The name of the dataset.
            filepath (str): The path to the input data file.

        Returns:
            Dataset: The created Weave dataset.
        """
        converter = DatasetConverterFactory.create_converter(filepath)
        converted_data = converter.convert()
        return Dataset(name=name, rows=converted_data)
