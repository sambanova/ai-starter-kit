"""
ChatTemplateManager Unit Tests
Verifies tokenizer loading, chat template rendering,
completions invocation, and parsing logic.
"""

import os
import sys
import json
import unittest
import logging
from unittest.mock import patch, MagicMock

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from src.chat_template import ChatTemplateManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ChatTemplateManagerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = ChatTemplateManager()
        
    def test_initial_state(self):
        self.assertIn("llama_json_parser", self.manager.parsers)
        self.assertIn("deepseek_xml_parser", self.manager.parsers)

    def test_jinja_validation_pass_and_fail(self):
        valid = "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}{% endfor %}"
        self.assertTrue(self.manager._validate_jinja_template(valid))
        with self.assertRaises(ValueError):
            self.manager._validate_jinja_template("{% for msg in messages %}{{ msg.role }}")

    def test_set_and_get_custom_template(self):
        dummy_template = "Hello {{ messages[0].content }}"
        model = "dummy-model"
        self.manager.set_chat_template(model, dummy_template)
        self.assertEqual(self.manager.chat_templates[model], dummy_template)
        
    def test_apply_custom_jinja_template(self):
        model = "dummy-model"
        self.manager.set_chat_template(model, "User: {{ messages[0].content }}")
        rendered = self.manager.apply_chat_template(model, [{"role": "user", "content": "Hi"}])
        self.assertEqual("User: Hi", rendered)

    @patch("src.chat_template.SambaNova")
    def test_completions_invoke_mocked(self, mock_client):
        # Mock API response
        mock_instance = MagicMock()
        mock_instance.completions.create.return_value.choices = [MagicMock(text="mocked result")]
        mock_client.return_value = mock_instance

        output = self.manager.completions_invoke("prompt text", "model")
        self.assertEqual(output, "mocked result")
        
    
    def test_llama_parser(self):
        text = """
        {"name": "get_population", "parameters": {"city": "Paris"}}
        {"name": "get_attractions", "parameters": {"city": "Paris", "limit": 3}}
        """
        result = self.manager.llama3_parser(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["function"]["name"], "get_population")
        self.assertEqual(json.loads(result[0]["function"]["arguments"]), {"city": "Paris"})

    def test_deepseek_parser(self):
        text = """
        <｜tool▁calls▁begin｜>
        <｜tool▁call▁begin｜>get_population<｜tool▁sep｜>{"city": "Paris"}<｜tool▁call▁end｜>
        <｜tool▁call▁begin｜>get_attractions<｜tool▁sep｜>{"city": "Paris", "limit": 3}<｜tool▁call▁end｜>
        <｜tool▁calls▁end｜>
        """
        result = self.manager.deepseek_v3_parser(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["function"]["name"], "get_population")
        self.assertEqual(json.loads(result[0]["function"]["arguments"]), {"city": "Paris"})
    
    def test_add_custom_parser_and_parse(self):
        code = """
def parse(response: str):
    return [{
        "id": "call_x",
        "type": "function",
        "function": {"name": "dummy", "arguments": '{"key": "value"}'}
    }]
"""
        self.manager.add_custom_tool_parser("custom", code)
        msg = self.manager.parse_to_message("irrelevant text", "custom")
        self.assertIsNone(msg["content"])
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "dummy")
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], '{"key": "value"}')
    
    def test_parse_to_message_no_tool(self):
        text = "plain text only"
        msg = self.manager.parse_to_message(text, "nonexistent")
        self.assertEqual(msg["content"], "plain text only")
        self.assertEqual(msg["tool_calls"], [])   
    
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(ChatTemplateManagerTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)