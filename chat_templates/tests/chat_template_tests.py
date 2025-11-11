"""
ChatTemplateManager Unit Tests
Verifies tokenizer loading, chat template rendering,
completions invocation, and parsing logic.
"""

import json
import logging
import os
import sys
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from chat_templates.src.chat_template import ChatTemplateManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatTemplateManagerTestCase(unittest.TestCase):
    manager: ChatTemplateManager

    @classmethod
    def setUpClass(cls) -> None:
        cls.manager = ChatTemplateManager()

    def test_initial_state(self) -> None:
        self.assertIn('llama_json_parser', self.manager.parsers)
        self.assertIn('deepseek_xml_parser', self.manager.parsers)

    def test_jinja_validation_pass_and_fail(self) -> None:
        valid = '{% for msg in messages %}{{ msg.role }}: {{ msg.content }}{% endfor %}'
        self.assertTrue(self.manager._validate_jinja_template(valid))
        with self.assertRaises(ValueError):
            self.manager._validate_jinja_template('{% for msg in messages %}{{ msg.role }}')

    def test_set_and_get_custom_template(self) -> None:
        dummy_template = 'Hello {{ messages[0].content }}'
        model = 'dummy-model'
        self.manager.set_chat_template(model, dummy_template)
        self.assertEqual(self.manager.chat_templates[model], dummy_template)

    def test_apply_custom_jinja_template(self) -> None:
        model = 'dummy-model'
        self.manager.set_chat_template(model, 'User: {{ messages[0].content }}')
        rendered = self.manager.apply_chat_template(model, [{'role': 'user', 'content': 'Hi'}])
        self.assertEqual('User: Hi', rendered)

    @patch('chat_templates.src.chat_template.SambaNova')
    def test_completions_invoke_mocked(self, mock_client: MagicMock) -> None:
        # Mock API response
        mock_instance = MagicMock()
        mock_instance.completions.create.return_value.choices = [MagicMock(text='mocked result')]
        mock_client.return_value = mock_instance

        output = self.manager.completions_invoke('prompt text', 'model')
        self.assertEqual(output, 'mocked result')

    def test_llama_parser(self) -> None:
        text = """
        {"name": "get_population", "parameters": {"city": "Paris"}}
        {"name": "get_attractions", "parameters": {"city": "Paris", "limit": 3}}
        """
        result = self.manager.llama3_parser(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['function']['name'], 'get_population')
        self.assertEqual(json.loads(result[0]['function']['arguments']), {'city': 'Paris'})

    def test_deepseek_parser(self) -> None:
        text = """
        <｜tool▁calls▁begin｜>
        <｜tool▁call▁begin｜>get_population<｜tool▁sep｜>{"city": "Paris"}<｜tool▁call▁end｜>
        <｜tool▁call▁begin｜>get_attractions<｜tool▁sep｜>{"city": "Paris", "limit": 3}<｜tool▁call▁end｜>
        <｜tool▁calls▁end｜>
        """
        result = self.manager.deepseek_v3_parser(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['function']['name'], 'get_population')
        self.assertEqual(json.loads(result[0]['function']['arguments']), {'city': 'Paris'})

    def test_add_custom_parser_and_parse(self) -> None:
        code = """
def parse(response: str):
    return [{
        "id": "call_x",
        "type": "function",
        "function": {"name": "dummy", "arguments": '{"key": "value"}'}
    }]
"""
        self.manager.add_custom_tool_parser('custom', code)
        msg = self.manager.parse_to_message('irrelevant text', 'custom')
        self.assertIsNone(msg['content'])
        self.assertEqual(msg['tool_calls'][0]['function']['name'], 'dummy')
        self.assertEqual(msg['tool_calls'][0]['function']['arguments'], '{"key": "value"}')

    def test_parse_to_message_no_tool(self) -> None:
        text = 'plain text only'
        msg = self.manager.parse_to_message(text, 'nonexistent')
        self.assertEqual(msg['content'], 'plain text only')
        self.assertEqual(msg['tool_calls'], [])


class CustomTextTestResult(unittest.TextTestResult):
    test_results: List[Dict[str, Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.test_results: List[Dict[str, Any]] = []

    def _get_test_name(self, test: unittest.TestCase) -> str:
        """Handle both TestCase and _ErrorHolder objects."""
        return getattr(test, '_testMethodName', str(test))

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self.test_results.append({'name': self._get_test_name(test), 'status': 'PASSED'})

    def addFailure(self, test: unittest.TestCase, err: Any) -> None:
        super().addFailure(test, err)
        self.test_results.append({'name': self._get_test_name(test), 'status': 'FAILED', 'message': str(err[1])})

    def addError(self, test: unittest.TestCase, err: Any) -> None:
        super().addError(test, err)
        self.test_results.append({'name': self._get_test_name(test), 'status': 'ERROR', 'message': str(err[1])})


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(ChatTemplateManagerTestCase)
    test_result = unittest.TextTestRunner(resultclass=CustomTextTestResult).run(suite)

    logger.info('\nTest Results:')
    assert hasattr(test_result, 'test_results')
    for result in test_result.test_results:
        logger.info(f'{result["name"]}: {result["status"]}')
        if 'message' in result:
            logger.info(f'  Message: {result["message"]}')

    failed_tests = len(test_result.failures) + len(test_result.errors)
    logger.info(f'\nTests passed: {test_result.testsRun - failed_tests}/{test_result.testsRun}')

    if failed_tests:
        logger.error(f'Number of failed tests: {failed_tests}')
        return failed_tests
    else:
        logger.info('All tests passed successfully!')
        return 0


if __name__ == '__main__':
    exit_code = main()
    if exit_code == 0:
        print("All CLI tests for chat_templates passed")
    sys.exit(exit_code)
