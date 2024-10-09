import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import time
import unittest
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Type

import requests

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Timeout variables (in seconds)
STREAMLIT_START_TIMEOUT = 25
STREAMLIT_CHECK_TIMEOUT = 5
CLI_COMMAND_TIMEOUT = 1200  # 10 minutes

# List of starter kits to test
STARTER_KITS: List[str] = [
    'benchmarking',
    'enterprise_knowledge_retriever',
    'financial_assistant',
    'function_calling',
    'search_assistant',
    'benchmarking',
    'image_search',
    'multimodal_knowledge_retriever',
    'post_call_analysis',
    'prompt_engineering',
    'web_crawled_data_retriever',
]

# Dictionary to store CLI test commands for each kit
CLI_TEST_COMMANDS: Dict[str, str] = {
    'benchmarking': './run_synthetic_dataset.sh',
    'enterprise_knowledge_retriever': 'python tests/ekr_test.py',
    'financial_assistant': 'python tests/financial_assistant_test.py',
    'function_calling': 'python tests/fc_test.py',
    'multimodal_knowledge_retriever': 'python tests/multimodal_knowledge_retriever_test.py',
    'post_call_analysis': 'python tests/pca_test.py',
    'prompt_engineering': 'python tests/prompt_engineering_test.py',
    'search_assistant': 'python tests/search_assistant_test.py',
    'image_search': 'python tests/image_search_test.py',

}


class TestEnvironment:
    """Enum-like class for test environments."""

    LOCAL = 'local'
    DOCKER = 'docker'


class TestResult:
    def __init__(
        self, kit: str, test_name: str, status: str, duration: float, message: str = '', date: str = ''
    ) -> None:
        self.kit = kit
        self.test_name = test_name
        self.status = status
        self.duration = duration
        self.message = message
        self.date = date  # Added date attribute


class StarterKitTest(unittest.TestCase):
    root_dir: ClassVar[str]
    env: ClassVar[str]
    run_streamlit: ClassVar[bool]
    run_cli: ClassVar[bool]
    is_docker: ClassVar[bool]
    csv_writer: ClassVar[Optional[csv.writer]]
    csv_file: ClassVar[Optional[Any]]

    @classmethod
    def setUpClass(cls: Type['StarterKitTest']) -> None:
        cls.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.is_docker = os.environ.get('DOCKER_ENV', 'false').lower() == 'true'
        cls.setup_csv_writer()

        if not cls.is_docker:
            cls.activate_base_venv()
            if cls.env == TestEnvironment.LOCAL:
                cls.run_make_clean()
                cls.run_make_all()

    @classmethod
    def tearDownClass(cls: Type['StarterKitTest']) -> None:
        if cls.csv_file:
            cls.csv_file.close()

    @classmethod
    def setup_csv_writer(cls: Type['StarterKitTest']) -> None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = '/app/test_results' if cls.is_docker else os.path.join(cls.root_dir, 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        csv_filename = os.path.join(results_dir, f'test_results_{timestamp}.csv')

        cls.csv_file = open(csv_filename, 'w', newline='')
        cls.csv_writer = csv.writer(cls.csv_file)
        # Updated CSV header to include 'Date'
        cls.csv_writer.writerow(['Kit', 'Test Name', 'Status', 'Duration (s)', 'Message', 'Date'])
        logging.info(f'Test results will be saved to {csv_filename}')

    @classmethod
    def write_test_result(cls, result: TestResult) -> None:
        if cls.csv_writer:
            cls.csv_writer.writerow(
                [
                    result.kit,
                    result.test_name,
                    result.status,
                    f'{result.duration:.2f}',
                    result.message,
                    result.date,  # Write the date to the CSV
                ]
            )
            cls.csv_file.flush()  # Ensure the result is written immediately

    @classmethod
    def run_make_clean(cls: Type['StarterKitTest']) -> None:
        logging.info('Running make clean...')
        subprocess.run(['make', 'clean'], cwd=cls.root_dir, check=True, capture_output=False)

    @classmethod
    def run_make_all(cls: Type['StarterKitTest']) -> None:
        logging.info('Running make all...')
        subprocess.run(['make', 'all'], cwd=cls.root_dir, check=True, capture_output=False)

    @classmethod
    def activate_base_venv(cls: Type['StarterKitTest']) -> None:
        base_venv_path = os.path.join(cls.root_dir, '.venv')
        if not os.path.exists(base_venv_path):
            logging.info('Base virtual environment not found. Creating it...')
            subprocess.run(['make', 'venv'], cwd=cls.root_dir, check=True)

        activate_this = os.path.join(base_venv_path, 'bin', 'activate_this.py')
        if os.path.exists(activate_this):
            exec(open(activate_this).read(), {'__file__': activate_this})
        else:
            logging.warning(f'activate_this.py not found at {activate_this}. Falling back to manual activation.')
            os.environ['VIRTUAL_ENV'] = base_venv_path
            os.environ['PATH'] = f"{base_venv_path}/bin:{os.environ['PATH']}"
            sys.prefix = base_venv_path

        logging.info(f'Activated base virtual environment at {base_venv_path}')

    def run_streamlit_test(self, kit: str) -> None:
        if not self.run_streamlit:
            logging.info(f'Skipping Streamlit test for {kit}')
            return

        start_time = time.time()
        logging.info(f'\nTesting {kit} Streamlit app...')
        kit_dir = os.path.join(self.root_dir, kit)
        process = subprocess.Popen(
            ['streamlit', 'run', 'streamlit/app.py', '--browser.gatherUsageStats', 'false'],
            cwd=kit_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            logging.info(f'Waiting for Streamlit to start (timeout: {STREAMLIT_START_TIMEOUT}s)...')
            time.sleep(STREAMLIT_START_TIMEOUT)
            self._check_streamlit_accessibility(kit)
            status = 'PASSED'
            message = ''
        except AssertionError as e:
            status = 'FAILED'
            message = str(e)
        finally:
            process.terminate()
            process.wait()

        duration = time.time() - start_time
        result = TestResult(kit, 'Streamlit', status, duration, message)
        self.write_test_result(result)

    def _check_streamlit_accessibility(self, kit: str) -> None:
        logging.info('Checking Streamlit accessibility...')
        for attempt in range(3):
            try:
                response = requests.get('http://localhost:8501', timeout=STREAMLIT_CHECK_TIMEOUT)
                self.assertEqual(response.status_code, 200, f'Streamlit app for {kit} is not running')
                logging.info(f'Streamlit app for {kit} is running successfully')
                return
            except (requests.ConnectionError, requests.Timeout):
                logging.info(f'Attempt {attempt + 1}: Streamlit not accessible yet. Waiting...')
                time.sleep(STREAMLIT_CHECK_TIMEOUT)

        logging.error(f'Streamlit app for {kit} failed to start after 3 attempts')
        self.fail(f'Streamlit app for {kit} failed to start after 3 attempts')

    def run_cli_test(self, kit: str) -> None:
        if not self.run_cli:
            logging.info(f'Skipping CLI test for {kit}')
            return

        start_time = time.time()
        logging.info(f'\nRunning CLI test for {kit}...')
        kit_dir = os.path.join(self.root_dir, kit)
        cli_command = CLI_TEST_COMMANDS.get(kit, 'echo "No CLI test command specified"')

        try:
            process = subprocess.Popen(
                cli_command, shell=True, cwd=kit_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            output_lines = []
            while True:
                # Read line by line from the subprocess output
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line, end='')  # Display output in real-time
                    output_lines.append(line)
                # Enforce timeout
                if time.time() - start_time > CLI_COMMAND_TIMEOUT:
                    process.kill()
                    raise subprocess.TimeoutExpired(cli_command, CLI_COMMAND_TIMEOUT)

            return_code = process.poll()
            output = ''.join(output_lines)

            if return_code == 0:
                logging.info(f'All CLI tests for {kit} passed')
                self.parse_subtest_results(kit, output, time.time() - start_time)
            else:
                logging.error(f'CLI test for {kit} failed. Return code: {return_code}')
                logging.error(f'Error output:\n{output}')
                result = TestResult(
                    kit, 'CLI', 'FAILED', time.time() - start_time, f'Return code: {return_code}\n{output}'
                )
                self.write_test_result(result)

        except subprocess.TimeoutExpired:
            process.kill()
            logging.error(f'CLI test for {kit} timed out after {CLI_COMMAND_TIMEOUT} seconds')
            result = TestResult(
                kit, 'CLI', 'TIMEOUT', CLI_COMMAND_TIMEOUT, f'Timed out after {CLI_COMMAND_TIMEOUT} seconds'
            )
            self.write_test_result(result)

    def parse_subtest_results(self, kit: str, output: str, total_duration: float) -> None:
        detailed_tests_found = False  # Flag to track if detailed tests are found
        # Split output into lines
        lines = output.strip().split('\n')
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Ignore lines like 'Tests passed'
            if 'Tests passed' in line:
                continue

            # Use regex to parse the line
            match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (\w+) - (.*)$', line)
            if match:
                date_str = match.group(1)
                log_level = match.group(2)
                message = match.group(3).strip()

                # Skip lines that are not test logs
                if not message.startswith('test_'):
                    continue

                test_name = message

                # Assuming status is 'PASSED' unless indicated otherwise
                status = 'PASSED'

                result = TestResult(kit, test_name, status, 0.0, '', date_str)
                self.write_test_result(result)
                detailed_tests_found = True  # Set flag to True since we found a detailed test
            else:
                # Line did not match the expected format; you can log or ignore it
                logging.debug(f'Line did not match pattern: {line}')
                continue

        if not detailed_tests_found:
            # No detailed tests found
            # Check for "All CLI tests passed" message
            all_passed_pattern = re.compile(r'All CLI tests for .+ passed')
            if all_passed_pattern.search(output):
                # Write general result with status "PASSED" and current timestamp
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                result = TestResult(kit, 'CLI', 'PASSED', total_duration, 'All CLI tests passed', current_time)
                self.write_test_result(result)
            else:
                # No detailed tests and no "All CLI tests passed" message
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                result = TestResult(
                    kit, 'CLI', 'UNKNOWN', total_duration, 'No detailed test results found', current_time
                )
                self.write_test_result(result)


# Move create_test_methods outside the class
def create_test_methods() -> None:
    for kit in STARTER_KITS:

        def test_kit(self: Any, kit: str = kit) -> None:
            self.run_streamlit_test(kit)
            self.run_cli_test(kit)

        test_kit.__name__ = f'test_{kit}'
        setattr(StarterKitTest, test_kit.__name__, test_kit)


create_test_methods()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tests for AI Starter Kit')
    parser.add_argument(
        '--env', choices=['local', 'docker'], default='local', help='Specify the test environment (default: local)'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Run tests in verbose mode')
    parser.add_argument('--skip-streamlit', action='store_true', help='Skip Streamlit tests')
    parser.add_argument('--skip-cli', action='store_true', help='Skip CLI tests')
    args, unknown = parser.parse_known_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    StarterKitTest.run_streamlit = not args.skip_streamlit
    StarterKitTest.run_cli = not args.skip_cli
    StarterKitTest.env = args.env

    suite = unittest.TestLoader().loadTestsFromTestCase(StarterKitTest)
    unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
