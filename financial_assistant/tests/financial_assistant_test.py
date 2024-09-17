"""
Enterprise Knowledge Retriever (EKR) Test Script

This script tests the functionality of the Dinancial Assistant using unittest.
It parses documents using the SambaParse service, creates a vector store, and tests the question-answering capabilities.

Usage:
    python tests/financial_assistant_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import logging
import unittest
from typing import Any, Dict, List, Tuple, Type, TypeVar

import pandas
from matplotlib.figure import Figure

from financial_assistant.src.llm import SambaNovaLLM
from financial_assistant.streamlit.utilities_app import create_temp_dir_with_subdirs, delete_temp_dir

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import streamlit

from financial_assistant.src.tools_database import create_stock_database, query_stock_database
from financial_assistant.src.tools_stocks import get_historical_price, get_stock_info
from financial_assistant.streamlit.app_stock_data import handle_stock_data_analysis, handle_stock_query
from financial_assistant.streamlit.app_stock_database import handle_database_creation
from financial_assistant.streamlit.constants import *
from financial_assistant.streamlit.utilities_app import initialize_session

# Create a generic variable that can be 'Parent', or any subclass.
T = TypeVar('T', bound='FinancialAssistantTest')


# Let's use this as a template for further CLI tests. setup, tests, teardown and assert at the end.
class FinancialAssistantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls: Type[T]) -> None:
        # Initialize the session
        initialize_session(session_state=streamlit.session_state, prod_mode=False, cache_dir=TEST_CACHE_DIR)

        # Initialize the LLM
        streamlit.session_state.llm = SambaNovaLLM()

        # Create the cache and its main subdirectories
        subdirectories = [
            streamlit.session_state.source_dir,
            streamlit.session_state.pdf_sources_directory,
            streamlit.session_state.pdf_generation_directory,
        ]
        create_temp_dir_with_subdirs(streamlit.session_state.cache_dir, subdirectories)

    def check_get_stock_info(self, response: Dict[str, str]) -> None:
        """Check the response of the tool `get_stock_info`."""

        # Assert that the response is a dictionary
        self.assertIsInstance(response, dict)
        # Assert that the response contains the expected keys
        self.assertIn(DEFAULT_COMPANY_NAME.upper(), response)
        # Assert that the response is a string
        self.assertIsInstance(response[DEFAULT_COMPANY_NAME.upper()], str)
        # Assert that the response is a png file
        self.assertTrue(response[DEFAULT_COMPANY_NAME.upper()].endswith('.png'))

    def check_get_historical_price(self, response: Tuple[pandas.DataFrame, Figure, List[str]]) -> None:
        """Check the response of the tool `get_historical_prices`."""

        # Assert that the response is a tuple of three elements
        self.assertIsInstance(response, tuple)
        self.assertEqual(len(response), 3)
        # Assert that the first element of the tuple is a `matplotlib.Figure`
        self.assertIsInstance(response[0], Figure)
        # Assert that the second element of the tuple is a `pandas.DataFrame
        self.assertIsInstance(response[1], pandas.DataFrame)
        # Assert that the third element of the tuple is a list of strings
        self.assertIsInstance(response[2], list)
        for item in response[2]:
            self.assertIsInstance(item, str)

    def check_create_stock_database(self, response: Dict[str, List[str]]) -> None:
        """Check the response of the tool `create_stock_database`."""

        # Assert that the response is a dictionary
        self.assertIsInstance(response, dict)

        # Assert that the response contains the expected keys
        self.assertListEqual(list(response), [DEFAULT_COMPANY_NAME.upper()])

        # Assert that the response contains the expected values
        self.assertIsInstance(response[DEFAULT_COMPANY_NAME.upper()], list)

        # Assert that each value of the list of tables is a string that starts with the expected prefix
        for table in response[DEFAULT_COMPANY_NAME.upper()]:
            self.assertIsInstance(table, str)
            self.assertTrue(table.startswith(f'{DEFAULT_COMPANY_NAME.lower()}_'))

    def check_query_stock_database(self, response: Any, method: str = 'text-to-SQL') -> None:
        """Check the response of the tool `query_stock_database`."""

        if method == 'text-to-SQL':
            self.assertIsInstance(response, str)
        elif method == 'PandasAI-SqliteConnector':
            self.assertIsInstance(response, dict)
            self.assertListEqual(list(response), [DEFAULT_COMPANY_NAME.upper()])
            self.assertIsInstance(response[DEFAULT_COMPANY_NAME.upper()], list)
            for item in response[DEFAULT_COMPANY_NAME.upper()]:
                self.assertIsInstance(item, str)
                self.assertTrue(item.endswith('.png'))
        else:
            raise ValueError(f'`method` should be either `text-to-SQL` or `PandasAI-SqliteConnector`. Got {method}.')

    def test_tools_stock_data(self) -> None:
        """Test for the tool `get_stock_info`."""

        # Invoke the tool to answer the user's query
        response = get_stock_info.invoke(
            {
                'user_query': DEFAULT_STOCK_QUERY,
                'company_list': [DEFAULT_COMPANY_NAME],
                'dataframe_name': DEFAULT_DATAFRAME_NAME,
            }
        )

        # Check the response
        self.check_get_stock_info(response)

    def test_handle_stock_query(self) -> None:
        """Test for `handle_stock_query`, i.e. function calling for the tool `get_stock_info`"""

        # The default user query
        query = DEFAULT_STOCK_QUERY

        # The dataframe name
        dataframe_name = DEFAULT_DATAFRAME_NAME

        # Invoke the LLM to answer the user's query
        response = handle_stock_query(query, dataframe_name)

        # Check the response
        self.check_get_stock_info(response)

    def test_get_historical_price(self) -> None:
        """Test for the tool `get_historical_price`."""

        # Invoke the tool to answer the user' query
        response = get_historical_price(
            {
                'company_list': [DEFAULT_COMPANY_NAME],
                'start_date': DEFAULT_START_DATE,
                'end_date': DEFAULT_END_DATE,
            }
        )

        # Check the response
        self.check_get_historical_price(response)

    def test_handle_stock_data_analysis(self) -> None:
        """Test for `handle_stock_data_analysis`, i.e. function calling for the tool `get_historical_price`."""

        # The default user query
        query = DEFAULT_HISTORICAL_STOCK_PRICE_QUERY

        # Invoke the LLM to answer the user's query
        response = handle_stock_data_analysis(
            user_question=query,
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE,
        )

        # Check the response
        self.check_get_historical_price(response)

    def test_create_stock_database(self) -> None:
        """Test for the tool `create_stock_database`."""

        # Invoke the tool to answer the user' query
        response = create_stock_database(
            {
                'company_list': [DEFAULT_COMPANY_NAME],
                'start_date': DEFAULT_START_DATE,
                'end_date': DEFAULT_END_DATE,
            }
        )

        # Check the response
        self.check_create_stock_database(response)

    def test_handle_database_creation(self) -> None:
        """Test for `handle_database_creation`, i.e. function calling for the tool `create_stock_database`."""

        response = handle_database_creation(
            requested_companies=DEFAULT_COMPANY_NAME,
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE,
        )

        # Check the response
        self.check_create_stock_database(response)

    def test_query_stock_database(self) -> None:
        """Test for the tool `query_stock_database`."""

        # Create the database
        create_stock_database(
            {
                'company_list': [DEFAULT_COMPANY_NAME],
                'start_date': DEFAULT_START_DATE,
                'end_date': DEFAULT_END_DATE,
            }
        )

        # List of available methods
        method_list = ['text-to-SQL', 'PandasAI-SqliteConnector']

        for method in method_list:
            # Invoke the tool to answer the user' query
            response = query_stock_database(
                {
                    'user_query': DEFAULT_STOCK_QUERY,
                    'company_list': [DEFAULT_COMPANY_NAME],
                    'method': method,
                }
            )

            # Check the response
            self.check_query_stock_database(response, method)


if __name__ == '__main__':
    unittest.main()

    # delete_temp_dir(temp_dir=TEST_CACHE_DIR)
