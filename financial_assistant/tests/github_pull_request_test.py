import unittest

from financial_assistant.tests.financial_assistant_test import FinancialAssistantTest, main


def suite_github_pull_request() -> unittest.TestSuite:
    """Test suite for GitHub actions on `pull_request`."""

    # List all the test cases here in order of execution
    suite_list = [
        'test_handle_database_creation',
        'test_handle_database_query',
        'test_handle_financial_filings',
    ]

    # Add all the tests to the suite
    suite = unittest.TestSuite()
    for suite_item in suite_list:
        suite.addTest(FinancialAssistantTest(suite_item))

    return suite


if __name__ == '__main__':
    exit_status = main(suite_github_pull_request())
    exit(exit_status)
