"""
Unit tests for vLLM benchmark module.

These tests cover the parts of vllm_benchmark.py that can be tested
without requiring actual API calls or vLLM installation.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from benchmarking.benchmarking_utils import find_family_model_type, get_tokenizer_model_name


class TestModelFamilyDetection(unittest.TestCase):
    """Tests for model family detection in benchmarking_utils."""

    def test_find_family_model_type_mistral(self):
        self.assertEqual(find_family_model_type("mistral-7b-instruct"), "mistral")
        self.assertEqual(find_family_model_type("Mistral-7B"), "mistral")

    def test_find_family_model_type_llama2(self):
        self.assertEqual(find_family_model_type("llama-2-7b-chat"), "llama2")
        self.assertEqual(find_family_model_type("Llama2-70b"), "llama2")

    def test_find_family_model_type_llama3(self):
        self.assertEqual(find_family_model_type("llama-3-8b-instruct"), "llama3")
        self.assertEqual(find_family_model_type("Llama-3.1-70B"), "llama3")
        self.assertEqual(find_family_model_type("llama3.1-8b"), "llama3")
        self.assertEqual(find_family_model_type("llama3p1-8b"), "llama3")
        self.assertEqual(find_family_model_type("Llama-3.2-1B"), "llama3")
        self.assertEqual(find_family_model_type("llama3.3-70b"), "llama3")

    def test_find_family_model_type_llama4(self):
        self.assertEqual(find_family_model_type("llama-4-maverick"), "llama4")
        self.assertEqual(find_family_model_type("Llama-4-Scout"), "llama4")

    def test_find_family_model_type_deepseek(self):
        self.assertEqual(find_family_model_type("deepseek-coder-6.7b"), "deepseek")
        self.assertEqual(find_family_model_type("DeepSeek-R1"), "deepseek")

    def test_find_family_model_type_qwen(self):
        self.assertEqual(find_family_model_type("qwen-2.5-7b-instruct"), "qwen")
        self.assertEqual(find_family_model_type("QwQ-32B"), "qwen")

    def test_find_family_model_type_solar(self):
        self.assertEqual(find_family_model_type("solar-10.7b-instruct"), "solar")

    def test_find_family_model_type_allam(self):
        self.assertEqual(find_family_model_type("allam-2-7b-instruct"), "allam")

    def test_find_family_model_type_minimax(self):
        self.assertEqual(find_family_model_type("minimax-text-01"), "minimax")

    def test_find_family_model_type_default(self):
        self.assertEqual(find_family_model_type("unknown-model"), "llama2")
        self.assertEqual(find_family_model_type("gpt-4"), "llama2")


class TestTokenizerModelName(unittest.TestCase):
    """Tests for tokenizer model name resolution."""

    def test_get_tokenizer_model_name_mistral(self):
        result = get_tokenizer_model_name("mistral-7b-instruct-v0.2")
        self.assertEqual(result, "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")

    def test_get_tokenizer_model_name_llama3(self):
        result = get_tokenizer_model_name("llama-3-8b-instruct")
        self.assertEqual(result, "unsloth/llama-3-8b-Instruct")

    def test_get_tokenizer_model_name_llama31(self):
        result = get_tokenizer_model_name("llama-3.1-8b-instruct")
        self.assertEqual(result, "unsloth/Llama-3.1-8B-Instruct")

    def test_get_tokenizer_model_name_llama32(self):
        result = get_tokenizer_model_name("llama-3.2-1b-instruct")
        self.assertEqual(result, "unsloth/Llama-3.2-1B-Instruct")

    def test_get_tokenizer_model_name_llama33(self):
        result = get_tokenizer_model_name("llama-3.3-70b-instruct")
        self.assertEqual(result, "unsloth/Llama-3.3-70B-Instruct")

    def test_get_tokenizer_model_name_llama4_maverick(self):
        result = get_tokenizer_model_name("llama-4-maverick-17b-128e-instruct")
        self.assertEqual(result, "unsloth/Llama-4-Maverick-17B-128E-Instruct")

    def test_get_tokenizer_model_name_llama4_scout(self):
        result = get_tokenizer_model_name("llama-4-scout-17b-16e-instruct")
        self.assertEqual(result, "unsloth/Llama-4-Scout-17B-16E-Instruct")

    def test_get_tokenizer_model_name_deepseek_coder(self):
        result = get_tokenizer_model_name("deepseek-coder-1.3b-base")
        self.assertEqual(result, "deepseek-ai/deepseek-coder-1.3b-base")

    def test_get_tokenizer_model_name_deepseek_r1(self):
        result = get_tokenizer_model_name("deepseek-r1")
        self.assertEqual(result, "deepseek-ai/DeepSeek-R1")

    def test_get_tokenizer_model_name_qwen(self):
        result = get_tokenizer_model_name("qwen-2.5-7b-instruct")
        self.assertEqual(result, "unsloth/Qwen2.5-72B-Instruct")

    def test_get_tokenizer_model_name_qwen_coder(self):
        result = get_tokenizer_model_name("qwen2.5-coder-32b")
        self.assertEqual(result, "unsloth/Qwen2.5-Coder-32B-Instruct")

    def test_get_tokenizer_model_name_qwq(self):
        result = get_tokenizer_model_name("qwq-32b")
        self.assertEqual(result, "unsloth/QwQ-32B")

    def test_get_tokenizer_model_name_solar(self):
        result = get_tokenizer_model_name("solar-10.7b-instruct")
        self.assertEqual(result, "upstage/SOLAR-10.7B-Instruct-v1.0")

    def test_get_tokenizer_model_name_minimax(self):
        result = get_tokenizer_model_name("minimax-text-01")
        self.assertEqual(result, "MiniMaxAI/MiniMax-M2.5")


class TestVLLMBenchmarkExecutor(unittest.TestCase):
    """Tests for VLLMBenchmarkExecutor class."""

    def setUp(self):
        from benchmarking.src.vllm_benchmark import VLLMBenchmarkExecutor

        self.VLLMBenchmarkExecutor = VLLMBenchmarkExecutor
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init_creates_results_directory(self):
        executor = self.VLLMBenchmarkExecutor(
            model_name="test-model",
            results_dir=self.temp_dir,
        )
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_init_stores_parameters(self):
        executor = self.VLLMBenchmarkExecutor(
            model_name="test-model",
            results_dir=self.temp_dir,
            timeout=300,
            user_metadata={"key": "value"},
        )
        self.assertEqual(executor.model_name, "test-model")
        self.assertEqual(executor.results_dir, self.temp_dir)
        self.assertEqual(executor.timeout, 300)
        self.assertEqual(executor.user_metadata, {"key": "value"})

    def test_init_default_timeout(self):
        executor = self.VLLMBenchmarkExecutor(
            model_name="test-model",
            results_dir=self.temp_dir,
        )
        self.assertEqual(executor.timeout, 600)

    def test_stop_benchmark_sets_event(self):
        executor = self.VLLMBenchmarkExecutor(
            model_name="test-model",
            results_dir=self.temp_dir,
        )
        self.assertFalse(executor.stop_event.is_set())
        executor.stop_benchmark()
        self.assertTrue(executor.stop_event.is_set())


class TestVLLMResultParsing(unittest.TestCase):
    """Tests for vLLM result parsing and conversion."""

    def setUp(self):
        from benchmarking.src.vllm_benchmark import VLLMBenchmarkExecutor

        self.VLLMBenchmarkExecutor = VLLMBenchmarkExecutor
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_find_and_parse_results(self):
        executor = self.VLLMBenchmarkExecutor(
            model_name="test-model",
            results_dir=self.temp_dir,
        )

        result_data = {
            "completed": 10,
            "failed": 0,
            "num_prompts": 10,
            "ttfts": [0.1] * 10,
            "output_lens": [100] * 10,
            "input_lens": [50] * 10,
            "itls": [[0.01] * 100] * 10,
            "mean_ttft_ms": 100,
            "median_ttft_ms": 95,
            "std_ttft_ms": 10,
            "duration": 10.0,
            "request_throughput": 1.0,
            "output_throughput": 10.0,
        }

        result_file = os.path.join(self.temp_dir, "results.json")
        with open(result_file, "w") as f:
            json.dump(result_data, f)

        results, file_path = executor._find_and_parse_results(self.temp_dir)

        self.assertEqual(results["completed"], 10)
        self.assertEqual(file_path, result_file)

    def test_find_and_parse_results_no_files_raises(self):
        executor = self.VLLMBenchmarkExecutor(
            model_name="test-model",
            results_dir=self.temp_dir,
        )

        with self.assertRaises(Exception) as context:
            executor._find_and_parse_results(self.temp_dir)

        self.assertIn("No result JSON files found", str(context.exception))

    def test_create_compatible_output(self):
        executor = self.VLLMBenchmarkExecutor(
            model_name="test-model",
            results_dir=self.temp_dir,
        )

        vllm_results = {
            "completed": 2,
            "failed": 0,
            "num_prompts": 2,
            "ttfts": [0.1, 0.2],
            "output_lens": [100, 150],
            "input_lens": [50, 60],
            "itls": [[0.01] * 100, [0.01] * 150],
            "mean_ttft_ms": 150,
            "median_ttft_ms": 140,
            "std_ttft_ms": 50,
            "duration": 5.0,
            "request_throughput": 0.4,
            "output_throughput": 50.0,
        }

        result_file = os.path.join(self.temp_dir, "results.json")
        with open(result_file, "w") as f:
            json.dump(vllm_results, f)
        executor.result_file_path = result_file

        executor._create_compatible_output(vllm_results, 50, 100, 1)

        self.assertIsNotNone(executor.individual_responses_file_path)
        self.assertIsNotNone(executor.summary_file_path)

        with open(executor.individual_responses_file_path, "r") as f:
            individual = json.load(f)
        self.assertEqual(len(individual), 2)
        self.assertIsNone(individual[0]["error_code"])

        with open(executor.summary_file_path, "r") as f:
            summary = json.load(f)
        self.assertEqual(summary["model"], "test-model")
        self.assertEqual(summary["results_num_completed_requests"], 2)
        self.assertEqual(summary["results_num_failed_requests"], 0)

    def test_create_compatible_output_with_failures(self):
        executor = self.VLLMBenchmarkExecutor(
            model_name="test-model",
            results_dir=self.temp_dir,
        )

        vllm_results = {
            "completed": 1,
            "failed": 1,
            "num_prompts": 2,
            "ttfts": [0.1, 0.0],
            "output_lens": [100, 0],
            "input_lens": [50, 50],
            "itls": [[0.01] * 100, []],
            "mean_ttft_ms": 100,
            "median_ttft_ms": 100,
            "std_ttft_ms": 0,
            "duration": 5.0,
            "request_throughput": 0.2,
            "output_throughput": 20.0,
        }

        result_file = os.path.join(self.temp_dir, "results.json")
        with open(result_file, "w") as f:
            json.dump(vllm_results, f)
        executor.result_file_path = result_file

        executor._create_compatible_output(vllm_results, 50, 100, 1)

        with open(executor.individual_responses_file_path, "r") as f:
            individual = json.load(f)

        self.assertEqual(len(individual), 2)
        self.assertIsNone(individual[0]["error_code"])
        self.assertIsNotNone(individual[1]["error_code"])

        with open(executor.summary_file_path, "r") as f:
            summary = json.load(f)
        self.assertEqual(summary["results_num_failed_requests"], 1)
        self.assertEqual(summary["results_error_rate"], 0.5)


class TestPromptProcessing(unittest.TestCase):
    """Tests for prompt processing logic."""

    def test_prompt_yaml_loading(self):
        import yaml

        prompt_yaml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../prompts/user-prompt_template-text_instruct.yaml",
        )

        if os.path.exists(prompt_yaml_path):
            with open(prompt_yaml_path, "r") as f:
                prompt_data = yaml.safe_load(f)

            self.assertIn("default_prompt", prompt_data)
            self.assertIsInstance(prompt_data["default_prompt"], list)
            self.assertGreater(len(prompt_data["default_prompt"]), 0)
            self.assertIn("template", prompt_data["default_prompt"][0])


if __name__ == "__main__":
    unittest.main()