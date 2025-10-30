"""
Financial Assistant Test Script.

This script tests the functionality of the Financial Assistant starter kit using `unittest`.

Usage:
    python financial_assistant/tests/financial_assistant_test.py
    python financial_assistant/tests/financial_assistant_test.py --suite suite_name

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import argparse
import os
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

import pandas
import streamlit
from matplotlib.figure import Figure

# Main directories
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from financial_assistant.constants import *
from financial_assistant.streamlit.utilities_app import (
    create_temp_dir_with_subdirs,
    delete_temp_dir,
    initialize_session,
    save_historical_price_callback,
    save_output_callback,
)

# Initialize the session
initialize_session(session_state=streamlit.session_state, prod_mode=False, cache_dir=TEST_CACHE_DIR)

from financial_assistant.src.tools_database import create_stock_database, query_stock_database
from financial_assistant.src.tools_filings import retrieve_filings
from financial_assistant.src.tools_pdf_generation import pdf_rag
from financial_assistant.src.tools_stocks import get_historical_price, get_stock_info
from financial_assistant.src.tools_yahoo_news import scrape_yahoo_finance_news
from financial_assistant.src.utilities import get_logger
from financial_assistant.streamlit.app_financial_filings import handle_financial_filings
from financial_assistant.streamlit.app_pdf_report import handle_pdf_generation, handle_pdf_rag
from financial_assistant.streamlit.app_stock_data import handle_stock_data_analysis, handle_stock_query
from financial_assistant.streamlit.app_stock_database import handle_database_creation, handle_database_query
from financial_assistant.streamlit.app_yfinance_news import handle_yfinance_news

# Instantiate the logger
logger = get_logger()

# Wait time for test retries in seconds
RETRY_SLEEP_TIME = 30


class FinancialAssistantTest(unittest.TestCase):
    """Test class for the Financial Assistant starter kit."""

    time_start: float
    time_end: float

    @classmethod
    def setUpClass(cls) -> None:
        """Records the start time of the tests."""

        cls.time_start = time.time()

    def setUp(self) -> None:
        """Set up before every test."""

        # Initialize SEC EDGAR credentials
        if os.getenv('SEC_API_ORGANIZATION') is None:
            os.environ['SEC_API_ORGANIZATION'] = 'SambaNova'
        if os.getenv('SEC_API_EMAIL') is None:
            os.environ['SEC_API_EMAIL'] = f'user_{streamlit.session_state.session_id}@sambanova_cloud.com'

        # Create the cache and its main subdirectories
        subdirectories = [
            streamlit.session_state.sources_dir,
            streamlit.session_state.pdf_sources_dir,
            streamlit.session_state.pdf_generation_dir,
        ]
        create_temp_dir_with_subdirs(streamlit.session_state.cache_dir, subdirectories)

        # List of available methods for database query
        self.method_list = ['text-to-SQL', 'PandasAI-SqliteConnector']

        # Wait for the LLM to be ready
        time.sleep(1)

    @classmethod
    def tearDownClass(cls) -> None:
        """Calculates and logs the total time taken to run the tests."""

        cls.time_end = time.time()
        total_time = cls.time_end - cls.time_start
        logger.info(f'Total execution time: {total_time:.2f} seconds.')

    def test_get_stock_info(self) -> None:
        """Test for the tool `get_stock_info`."""

        # Invoke the tool to answer the user query
        response = get_stock_info.invoke(
            {
                'user_query': DEFAULT_STOCK_QUERY,
                'company_list': [DEFAULT_TICKER_NAME],
                'dataframe_name': DEFAULT_DATAFRAME_NAME,
            }
        )

        # Check the response
        self.check_get_stock_info(response)

    def test_handle_stock_query(self) -> None:
        """Test for `handle_stock_query`, i.e. function calling for the tool `get_stock_info`."""

        # The default user query
        query = DEFAULT_STOCK_QUERY

        # The dataframe name
        dataframe_name = DEFAULT_DATAFRAME_NAME

        # Invoke the LLM to answer the user query
        response = handle_stock_query(query, dataframe_name)

        # Check the response
        self.check_get_stock_info(response)

        # Save response to cache
        save_output_callback(response, streamlit.session_state.stock_query_path, query)

    def test_get_historical_price(self) -> None:
        """Test for the tool `get_historical_price`."""

        # Invoke the tool to answer the user query
        response = get_historical_price.invoke(
            {
                'company_list': [DEFAULT_TICKER_NAME],
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

        # Invoke the LLM to answer the user query
        response = handle_stock_data_analysis(
            user_question=query,
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE,
        )

        # Check the response
        self.check_get_historical_price(response)

        # Save response to cache
        save_historical_price_callback(
            query,
            response[2],
            response[1],
            response[0],
            DEFAULT_START_DATE,
            DEFAULT_END_DATE,
            streamlit.session_state.stock_query_path,
        )

    def test_create_stock_database(self) -> None:
        """Test for the tool `create_stock_database`."""

        # Invoke the tool to answer the user query
        response = create_stock_database.invoke(
            {
                'company_list': [DEFAULT_TICKER_NAME],
                'start_date': DEFAULT_START_DATE,
                'end_date': DEFAULT_END_DATE,
            }
        )

        # Check the response
        self.check_create_stock_database(response)

    def test_handle_database_creation(self) -> None:
        """Test for `handle_database_creation`, i.e. function calling for the tool `create_stock_database`."""

        # Invoke the LLM to create the database
        response = handle_database_creation(
            requested_companies=DEFAULT_TICKER_NAME,
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE,
        )

        # Check the response
        self.check_create_stock_database(response)

    def test_query_stock_database(self) -> None:
        """Test for the tool `query_stock_database`."""

        # Create the database
        create_stock_database.invoke(
            {
                'company_list': [DEFAULT_TICKER_NAME],
                'start_date': DEFAULT_START_DATE,
                'end_date': DEFAULT_END_DATE,
            }
        )

        for method in self.method_list:
            # Invoke the tool to answer the user query
            response = query_stock_database.invoke(
                {
                    'user_query': DEFAULT_STOCK_QUERY,
                    'company_list': [DEFAULT_TICKER_NAME],
                    'method': method,
                }
            )

            # Check the response
            self.check_query_stock_database(response, method)

    def test_handle_database_query(self) -> None:
        """Test for `handle_database_query`, i.e. function calling for the tool `query_stock_database`."""

        # Create the database
        create_stock_database.invoke(
            {
                'company_list': [DEFAULT_TICKER_NAME],
                'start_date': DEFAULT_START_DATE,
                'end_date': DEFAULT_END_DATE,
            }
        )

        # The user query
        query = DEFAULT_STOCK_QUERY

        for method in self.method_list:
            # Invoke the LLM to answer the user query
            response = handle_database_query(
                user_question=query,
                query_method=method,
            )

            # Check the response
            self.check_query_stock_database(response, method)

            # Save response to cache
            save_output_callback(response, streamlit.session_state.db_query_path, query)

    def test_scrape_yahoo_finance_news(self) -> None:
        """Test for the tool `scrape_yahoo_finance_news`."""

        # Invoke the tool to answer the user query
        response, url_list = scrape_yahoo_finance_news.invoke(
            {
                'company_list': [DEFAULT_TICKER_NAME],
                'user_query': DEFAULT_RAG_QUERY,
            }
        )

        # Check the response
        self.check_scrape_yahoo_finance_news(response, url_list)

    def test_handle_yfinance_news(self) -> None:
        """Test for `handle_yfinance_news`, i.e. function calling for the tool `scrape_yahoo_finance_news`."""

        # The default user query
        query = DEFAULT_RAG_QUERY

        # Invoke the LLM to answer the user query
        response, url_list = handle_yfinance_news(
            user_question=query,
        )

        # Check the response
        self.check_scrape_yahoo_finance_news(response, url_list)

        # Save response to cache
        content = response + '\n\n'.join(url_list)
        save_output_callback(content, streamlit.session_state.yfinance_news_path, query)

    def test_retrieve_filings(self) -> None:
        """Test for the tool `retrieve_filings`."""

        # Invoke the tool to answer the user query
        response = retrieve_filings.invoke(
            {
                'user_question': DEFAULT_RAG_QUERY,
                'company_list': [DEFAULT_TICKER_NAME],
                'filing_type': DEFAULT_FILING_TYPE,
                'filing_quarter': DEFAULT_FILING_QUARTER,
                'year': DEFAULT_FILING_YEAR,
            }
        )

        # Check the response
        self.check_retrieve_filings(response)

    def test_handle_financial_filings(self) -> None:
        """Test for `handle_financial_filings`, i.e. function calling for the tool `retrieve_filings`"""

        # The user query
        query = DEFAULT_RAG_QUERY

        # Invoke the LLM to answer the user query
        response = handle_financial_filings(
            user_question=DEFAULT_RAG_QUERY,
            company_name=DEFAULT_TICKER_NAME,
            filing_type=DEFAULT_FILING_TYPE,
            filing_quarter=DEFAULT_FILING_QUARTER,
            selected_year=DEFAULT_FILING_YEAR,
        )

        # Check the response
        self.check_retrieve_filings(response)

        # Save response to cache
        save_output_callback(response, streamlit.session_state.filings_path, query)

    def test_handle_pdf_generation(self) -> None:
        """Test the tool `handle_pdf_generation`."""

        report_title = DEFAULT_PDF_TITLE
        report_name = report_title.lower().replace(' ', '_') + '.pdf'

        data_paths = dict()
        data_paths['stock_query'] = streamlit.session_state.stock_query_path
        data_paths['stock_database'] = streamlit.session_state.db_query_path
        data_paths['yfinance_news'] = streamlit.session_state.yfinance_news_path
        data_paths['filings'] = streamlit.session_state.filings_path

        # Generate the PDF report
        pdf_handler = handle_pdf_generation(report_title, report_name, data_paths, True)

        # Check the PDF report output
        self.assertIsInstance(pdf_handler, bytes)

    def test_pdf_rag(self) -> None:
        """Test the tool `pdf_rag`."""

        report_title = DEFAULT_PDF_TITLE
        report_name = report_title.lower().replace(' ', '_') + '.pdf'

        # The user query
        query = DEFAULT_PDF_RAG_QUERY

        # The pdf files names
        pdf_files_names = [os.path.join(streamlit.session_state.pdf_generation_dir, report_name)]

        # Invoke the tool to answer the user query
        response = pdf_rag.invoke(
            {
                'user_query': query,
                'pdf_files_names': pdf_files_names,
            }
        )

        # Check the response
        self.assertIsInstance(response, str)

    def test_handle_pdf_rag(self) -> None:
        """Test `handle_pdf_rag`, i.e. function calling for the tool `pdf_rag`."""

        report_title = DEFAULT_PDF_TITLE
        report_name = report_title.lower().replace(' ', '_') + '.pdf'

        # The user query
        query = DEFAULT_PDF_RAG_QUERY

        # The pdf files names
        pdf_files_names = [os.path.join(streamlit.session_state.pdf_generation_dir, report_name)]

        # Invoke the LLM to answer the user query
        response = handle_pdf_rag(query, pdf_files_names)

        # Check the response
        self.assertIsInstance(response, str)

    def test_delete_cache(self) -> None:
        """Delete the cache directory and its subdirectories."""

        delete_temp_dir(temp_dir=TEST_CACHE_DIR, verbose=False)

    def check_get_stock_info(self, response: Dict[str, str]) -> None:
        """Check the response of the tool `get_stock_info`."""

        # Ensure that the response is a dictionary
        self.assertIsInstance(response, dict)
        # Ensure that the response contains the expected keys
        self.assertIn(DEFAULT_TICKER_NAME.upper(), response)
        # Ensure that the response is a string
        self.assertIsInstance(response[DEFAULT_TICKER_NAME.upper()], str)
        # Ensure that the response is a png file
        self.assertTrue(response[DEFAULT_TICKER_NAME.upper()].endswith('.png'))

    def check_get_historical_price(self, response: Tuple[pandas.DataFrame, Figure, List[str]]) -> None:
        """Check the response of the tool `get_historical_prices`."""

        # Ensure that the response is a tuple of three elements
        self.assertIsInstance(response, tuple)
        self.assertEqual(len(response), 3)

        # Ensure that the first element of the tuple is a `matplotlib.Figure`
        self.assertIsInstance(response[0], Figure)

        # Ensure that the second element of the tuple is a `pandas.DataFrame
        self.assertIsInstance(response[1], pandas.DataFrame)

        # Ensure that the third element of the tuple is a list of strings
        self.assertIsInstance(response[2], list)
        for item in response[2]:
            self.assertIsInstance(item, str)

    def check_create_stock_database(self, response: Dict[str, List[str]]) -> None:
        """Check the response of the tool `create_stock_database`."""

        # Ensure that the response is a dictionary
        self.assertIsInstance(response, dict)

        # Ensure that the response contains the expected keys
        self.assertListEqual(list(response), [DEFAULT_TICKER_NAME.upper()])

        # Ensure that the response contains the expected values
        self.assertIsInstance(response[DEFAULT_TICKER_NAME.upper()], list)

        # Ensure that each value of the list of tables is a string that starts with the expected prefix
        for table in response[DEFAULT_TICKER_NAME.upper()]:
            self.assertIsInstance(table, str)
            self.assertTrue(table.startswith(f'{DEFAULT_TICKER_NAME.lower()}_'))

    def check_query_stock_database(self, response: Any, method: str = 'text-to-SQL') -> None:
        """Check the response of the tool `query_stock_database`."""

        if method == 'text-to-SQL':
            # Ensure that the response is a string
            self.assertIsInstance(response, str)

        elif method == 'PandasAI-SqliteConnector':
            # Ensure that the response is a dictionary
            self.assertIsInstance(response, dict)

            # Ensure that the response contains the expected keys
            self.assertListEqual(list(response), [DEFAULT_TICKER_NAME.upper()])

            # Ensure that the response contains the expected values
            self.assertIsInstance(response[DEFAULT_TICKER_NAME.upper()], list)
            for item in response[DEFAULT_TICKER_NAME.upper()]:
                self.assertIsInstance(item, str)
                self.assertTrue(item.endswith('.png'))
        else:
            raise ValueError(f'`method` should be either `text-to-SQL` or `PandasAI-SqliteConnector`. Got {method}.')

    def check_scrape_yahoo_finance_news(self, response: str, url_list: List[str]) -> None:
        """Check the response of the tool `scrape_yahoo_finance_news`."""

        # Ensure that `response` is a string
        self.assertIsInstance(response, str)

        # Ensure that `url_list` is a list of strings that are all URLs
        self.assertIsInstance(url_list, list)
        for url in url_list:
            self.assertIsInstance(url, str)
            self.assertTrue(url.startswith('https://'))

    def check_retrieve_filings(self, response: Dict[str, str]) -> None:
        """Check the response of the tool `retrieve_filings`."""

        # Ensure that the response is a dictionary
        self.assertIsInstance(response, dict)

        # Ensure that the response contains the expected keys
        self.assertEqual(list(response), [DEFAULT_TICKER_NAME.upper()])

        # Ensure that the response contains the expected values
        self.assertIsInstance(response[DEFAULT_TICKER_NAME.upper()], str)


class CustomTextTestResult(unittest.TextTestResult):
    """Custom test result class to collect and log individual test results."""

    max_retry: int = 1
    global_retry_count: int = 0
    successes: List[unittest.TestCase] = list()

    def __init__(self, stream: Any, descriptions: bool, verbosity: int) -> None:
        """Initializes the custom test result."""

        super().__init__(stream, descriptions, verbosity)

        # Initialize the list of test results and the list of successes
        self.results_list: List[Dict[str, Any]] = list()
        self.successes: List[unittest.TestCase] = list()

        # Set the number of retries: 1 means run failed or errored tests twice
        self.max_retry: int = 1

        # Initialize the global retry count
        self.global_retry_count: int = 0

    def _initialize_retry(self, test: unittest.TestCase) -> None:
        """Initialize the retry count for the given test case."""

        if not hasattr(test, 'retry'):
            setattr(test, 'retry', self.max_retry)

    def _sleep(self) -> None:
        """Wait for the given number of seconds before continuing with the next test run."""

        time.sleep(RETRY_SLEEP_TIME)

    def startTest(self, test: unittest.TestCase) -> None:
        """Start a test case."""

        logger.info(f'Running test {test._testMethodName}.')
        super().startTest(test)

    def addSuccess(self, test: unittest.TestCase) -> None:
        """Logs a passed test."""

        # Initialize the retry count for this test case
        self._initialize_retry(test)

        # Add the test to the list of successful tests
        super().addSuccess(test)

        # Append to the list of successful tests
        self.successes.append(test)
        self.results_list.append({'name': test._testMethodName, 'status': 'PASSED'})

        # Log the retry status for this test case
        if getattr(test, 'retry', self.max_retry) < self.max_retry:
            logger.warning(f'Test {test._testMethodName} passed on retry.')
        else:
            logger.info(f'Test {test._testMethodName} passed on first attempt.')

    def addFailure(self, test: unittest.TestCase, err: Any) -> None:
        """Logs a failed test."""

        # Initialize the retry count for this test case
        self._initialize_retry(test)

        if getattr(test, 'retry', self.max_retry) > 0:
            logger.warning(f'Retrying {test._testMethodName} after failure.')

            # Wait
            self._sleep()

            # Retrieve the retry count for this test case
            retry = getattr(test, 'retry')

            # Decrease the retry count for this test case
            setattr(test, 'retry', retry - 1)

            # Increase the global retry count
            self.global_retry_count += 1

            # Run the test case again
            test.run(self)
        else:
            # Add the test to the list of failed tests
            super().addFailure(test, err)

            # Append to the list of failed tests
            self.results_list.append({'name': test._testMethodName, 'status': 'FAILED', 'message': str(err[1])})

    def addError(self, test: unittest.TestCase, err: Any) -> None:
        """Logs a test with errors."""

        # Initialize the retry count for this test case
        self._initialize_retry(test)

        if getattr(test, 'retry', self.max_retry) > 0:
            logger.warning(f'Retrying {test._testMethodName} after error.')

            # Wait
            self._sleep()

            # Retrieve the retry count for this test case
            retry = getattr(test, 'retry')

            # Decrease the retry count for this test case
            setattr(test, 'retry', retry - 1)

            # Increase the global retry count
            self.global_retry_count += 1

            # Run the test case again
            test.run(self)
        else:
            # Add the test to the list of errored tests
            super().addError(test, err)

            # Append to the list of errored tests
            self.results_list.append({'name': test._testMethodName, 'status': 'ERROR', 'message': str(err[1])})


def main_suite() -> unittest.TestSuite:
    """Test suite to define the order of the test execution."""

    # List all the test cases here in order of execution
    suite_list = [
        'test_handle_stock_query',  # 3 LLM calls
        'test_handle_stock_data_analysis',  # 2 LLM calls
        'test_handle_database_creation',  # 2 LLM calls
        'test_handle_database_query',  # 8 LLM calls (both `text-to-SQL` and `PandasAI-SqliteConnector`)
        'test_handle_yfinance_news',  # 3 LLM calls
        'test_handle_financial_filings',  # 3 LLM calls
        'test_handle_pdf_generation',  # 8 LLM calls
        'test_handle_pdf_rag',  # 2 LLM calls
        'test_delete_cache',  # 0 LLM call
    ]

    # Add all the tests to the suite
    suite = unittest.TestSuite()
    for suite_item in suite_list:
        suite.addTest(FinancialAssistantTest(suite_item))

    return suite


def suite_github_pull_request() -> unittest.TestSuite:
    """Test suite for GitHub actions on `pull_request`, i.e. to be run at every pull request commit."""

    # List all the test cases here in order of execution
    suite_list = ['test_handle_stock_query']  # 3 LLM calls

    # Add all the tests to the suite
    suite = unittest.TestSuite()
    for suite_item in suite_list:
        suite.addTest(FinancialAssistantTest(suite_item))

    return suite


# Suite registry for the tests
suite_registry = {
    'main': main_suite(),
    'github_pull_request': suite_github_pull_request(),
}


def main(suite: Optional[unittest.TestSuite] = None) -> int:
    """Main program to run a test suite."""

    if suite is not None:
        if not isinstance(suite, unittest.TestSuite):
            raise TypeError(f'`suite` must be of type `unittest.TestSuite`. Got type {type(suite)}.')
    else:
        suite = main_suite()

    # The test runner
    runner = unittest.TextTestRunner(resultclass=CustomTextTestResult)

    # Run the tests
    test_results = runner.run(suite)

    logger.info('Test Results:')

    if not hasattr(test_results, 'results_list'):
        raise Exception('The test results does not have the attribute `results_list`.')
    if not hasattr(test_results, 'global_retry_count'):
        raise Exception('The test results does not have the attribute `global_retry_count`.')
    if not hasattr(test_results, 'successes'):
        raise Exception('The test results does not have the attribute `successes`.')

    # Log results for each test
    for result in test_results.results_list:
        logger.info(f'{result["name"]}: {result["status"]}')
        if 'message' in result:
            logger.warning(f'  Message: {result["message"]}')

    # Log the number of tests that passed and failed: successes, failures, and errors
    failed_tests = len(test_results.failures) + len(test_results.errors)
    logger.info(
        'Number of tests passed: '
        f'{test_results.testsRun - test_results.global_retry_count - failed_tests}/'  # type: ignore
        f'{test_results.testsRun - test_results.global_retry_count}'  # type: ignore
    )
    logger.info(f'Number of successful tests: {len(test_results.successes)}')  # type: ignore
    logger.info(f'Number of failed tests: {len(test_results.failures)}')
    logger.info(f'Number of errored tests: {len(test_results.errors)}')

    # Return the number of failed tests
    if failed_tests > 0:
        logger.error(f'Number of failed tests: {failed_tests}')
        return failed_tests
    else:
        logger.info('All tests passed successfully!')
        return 0


if __name__ == '__main__':
    # Add the argument --suite to run a specific suite of tests
    parser = argparse.ArgumentParser(description='Add a test suite for `financial_assistant`.')
    parser.add_argument('--suite', default='main', type=str, help='Suite for the tests.')

    # Select the suite to run
    args = parser.parse_args()
    try:
        suite = suite_registry[args.suite]
    except KeyError as e:
        suite = suite_registry['main']

    # Run the tests
    exit_status = main(suite)
    exit(exit_status)
