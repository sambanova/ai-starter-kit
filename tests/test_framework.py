"""
Test framework for AI Starter Kit.

This module provides a unittest-based framework for testing various starter kits
in both local and Docker environments. It includes functionality to set up the
testing environment, run Streamlit applications, and validate their operation.
It also includes placeholders for CLI/script tests.

If you are running .sh files as CLI tests, ensure you have made them executable.
"""

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
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Type

import pandas as pd
import requests

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(repo_dir)


# Timeout variables (in seconds)
STREAMLIT_START_TIMEOUT = 25
STREAMLIT_CHECK_TIMEOUT = 5
CLI_COMMAND_TIMEOUT = 1200  # 20 minutes

# List of starter kits to test
STARTER_KITS: List[str] = [
    'benchmarking',
    'enterprise_knowledge_retriever',
    # 'financial_assistant',
    'function_calling',
    'search_assistant',
    'multimodal_knowledge_retriever',
    'document_comparison',
    'utils',
]

# Dictionary to store CLI test commands for each kit
CLI_TEST_COMMANDS: Dict[str, str] = {
    'benchmarking': (
        'python src/evaluator.py '
        '--mode synthetic '
        "--model-names 'Meta-Llama-3.1-8B-Instruct Meta-Llama-3.3-70B-Instruct Meta-Llama-3.1-405B-Instruct' "
        "--results-dir './data/results/llmperf' "
        '--num-concurrent-requests 1 '
        '--timeout 600 '
        '--num-input-tokens 1000 '
        '--num-output-tokens 1000 '
        '--multimodal-image-size na '
        '--num-requests 2 '
        '--use-multiple-prompts False '
        '--save-llm-responses False '
        '--llm-api sncloud'
    ),
    'enterprise_knowledge_retriever': 'python tests/ekr_test.py',
    'financial_assistant': 'python tests/financial_assistant_test.py',
    'function_calling': 'python tests/fc_test.py',
    'multimodal_knowledge_retriever': ('python tests/multimodal_knowledge_retriever_test.py'),
    'search_assistant': 'python tests/search_assistant_test.py',
    'document_comparison': 'python tests/dc_test.py',
    'utils': 'python tests/api_testing.py',
}


class TestEnvironment:
    """Enum-like class for test environments."""

    LOCAL = 'local'
    DOCKER = 'docker'


class TestResult:
    """Class to store individual test results."""

    def __init__(
        self,
        kit: str,
        test_name: str,
        status: str,
        duration: float,
        message: str = '',
        date: str = '',
        commit_hash: str = '',
        commit_url: str = '',
    ) -> None:
        self.kit = kit
        self.test_name = test_name
        self.status = status
        self.duration = duration
        self.message = message
        self.date = date
        self.commit_hash = commit_hash
        self.commit_url = commit_url


class CsvWriter(Protocol):
    """Protocol for CSV writer to ensure type checking."""

    def writerow(self, row: List[Any]) -> None: ...

    def writerows(self, rows: List[List[Any]]) -> None: ...


class StarterKitTest(unittest.TestCase):
    """Main test class for the AI Starter Kits."""

    root_dir: ClassVar[str]
    env: ClassVar[str]
    run_streamlit: ClassVar[bool]
    run_cli: ClassVar[bool]
    is_docker: ClassVar[bool]
    recreate_venv: ClassVar[bool]
    csv_writer: ClassVar[Optional[CsvWriter]]
    csv_file: ClassVar[Optional[Any]]
    csv_filename: ClassVar[str]
    test_results: ClassVar[List[TestResult]] = []
    any_test_failed: ClassVar[bool] = False
    commit_hash: ClassVar[str] = ''
    commit_url: ClassVar[str] = ''

    @classmethod
    def setUpClass(cls: Type['StarterKitTest']) -> None:
        """Set up the testing environment."""
        cls.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.is_docker = os.environ.get('DOCKER_ENV', 'false').lower() == 'true'
        cls.setup_csv_writer()

        # Retrieve the commit hash from the environment variable
        cls.commit_hash = os.environ.get('GIT_COMMIT_HASH', 'unknown')

        # Construct the commit URL
        cls.commit_url = f'https://github.com/sambanova/ai-starter-kit/commit/{cls.commit_hash}'

        if not cls.is_docker:
            cls.activate_base_venv()
            if cls.env == TestEnvironment.LOCAL:
                if cls.recreate_venv:
                    cls.run_make_clean()
                cls.run_make_all()

    @classmethod
    def tearDownClass(cls: Type['StarterKitTest']) -> None:
        """Clean up after all tests have run."""
        if cls.csv_file:
            cls.csv_file.close()

        # Calculate durations using pandas
        cls.calculate_durations()

    @classmethod
    def setup_csv_writer(cls: Type['StarterKitTest']) -> None:
        """Set up the CSV writer for recording test results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = '/app/test_results' if cls.is_docker else os.path.join(cls.root_dir, 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        cls.csv_filename = os.path.join(results_dir, f'test_results_{timestamp}.csv')

        cls.csv_file = open(cls.csv_filename, 'w', newline='')
        cls.csv_writer = csv.writer(cls.csv_file)
        # Updated CSV header to include 'Commit URL'
        cls.csv_writer.writerow(
            ['Kit', 'Test Name', 'Status', 'Duration (s)', 'Message', 'Date', 'Commit Hash', 'Commit URL']
        )
        logging.info(f'Test results will be saved to {cls.csv_filename}')

    @classmethod
    def calculate_kit_durations(cls, df: Any) -> Any:
        """Calculate kit durations, preserving existing non-zero durations."""
        # Convert date and sort
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])

        # Calculate time differences between kit start times
        kit_times = df.groupby('Kit')['Date'].first().sort_values()

        # Calculate durations by taking difference with previous time
        kit_durations = (kit_times - kit_times.shift()).dt.total_seconds()

        # Map durations to kits, excluding first kit (it keeps original duration)
        duration_map = pd.Series(kit_durations.values[1:], index=kit_times.index[1:])

        # Create a mask for rows that need updating (duration is 0)
        zero_duration_mask = (df['Duration (s)'].astype(float) == 0.0) & (df['Kit'] != kit_times.index[0])

        # Update only the rows with zero duration, excluding first kit
        if zero_duration_mask.any():
            df.loc[zero_duration_mask, 'Duration (s)'] = df.loc[zero_duration_mask, 'Kit'].map(duration_map)

        return df.sort_values(['Kit', 'Date'])

    @classmethod
    def calculate_durations(cls) -> None:
        """Calculate kit durations, preserving existing non-zero durations."""
        try:
            import pandas as pd
        except ImportError:
            logging.error('Pandas is not installed. Please install pandas to calculate durations.')
            return

        df = pd.read_csv(cls.csv_filename)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S,%f')

        # Use the calculate_kit_durations function
        df = cls.calculate_kit_durations(df)

        # Update test_results with new durations
        for index, row in df.iterrows():
            for result in cls.test_results:
                if (
                    result.kit == row['Kit']
                    and result.test_name == row['Test Name']
                    and result.date == row['Date'].strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                ):
                    result.duration = float(row['Duration (s)'])
                    break

        # Write the updated DataFrame back to CSV
        df.to_csv(cls.csv_filename, index=False)

    @classmethod
    def write_test_result(cls, result: TestResult) -> None:
        """Write a test result to the CSV file and store it."""
        cls.test_results.append(result)
        if result.status != 'PASSED':
            cls.any_test_failed = True
        if cls.csv_writer and cls.csv_file:
            cls.csv_writer.writerow(
                [
                    result.kit,
                    result.test_name,
                    result.status,
                    f'{result.duration:.6f}',
                    result.message,
                    result.date,
                    result.commit_hash,
                    result.commit_url,
                ]
            )
            cls.csv_file.flush()  # Ensure the result is written immediately

    @classmethod
    def run_make_clean(cls: Type['StarterKitTest']) -> None:
        """Run 'make clean' to clean the environment."""
        logging.info('Running make clean...')
        subprocess.run(['make', 'clean'], cwd=cls.root_dir, check=True, capture_output=False)

    @classmethod
    def run_make_all(cls: Type['StarterKitTest']) -> None:
        """Run 'make all' to set up the environment."""
        logging.info('Running make all...')
        subprocess.run(['make', 'all'], cwd=cls.root_dir, check=True, capture_output=False)

    @classmethod
    def activate_base_venv(cls: Type['StarterKitTest']) -> None:
        """Activate the base virtual environment."""
        base_venv_path = os.path.join(cls.root_dir, '.venv')
        if not os.path.exists(base_venv_path):
            logging.info('Base virtual environment not found. Creating it...')
            subprocess.run(['make', 'venv'], cwd=cls.root_dir, check=True)

        activate_this = os.path.join(base_venv_path, 'bin', 'activate_this.py')
        if os.path.exists(activate_this):
            with open(activate_this) as f:
                exec(f.read(), {'__file__': activate_this})
        else:
            logging.warning(f'activate_this.py not found at {activate_this}. Falling back to manual activation.')
            os.environ['VIRTUAL_ENV'] = base_venv_path
            os.environ['PATH'] = f"{base_venv_path}/bin:{os.environ['PATH']}"
            sys.prefix = base_venv_path

        logging.info(f'Activated base virtual environment at {base_venv_path}')

    def run_streamlit_test(self, kit: str) -> None:
        """Run the Streamlit test for a given starter kit."""
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
        result = TestResult(
            kit,
            'Streamlit',
            status,
            duration,
            message,
            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
            commit_hash=self.commit_hash,
            commit_url=self.commit_url,
        )
        self.write_test_result(result)

    def _check_streamlit_accessibility(self, kit: str) -> None:
        """Check if the Streamlit app is accessible."""
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
        """Run the CLI test for a given starter kit."""
        if not self.run_cli:
            logging.info(f'Skipping CLI test for {kit}')
            return

        start_time = time.time()
        logging.info(f'\nRunning CLI test for {kit}...')
        kit_dir = os.path.join(self.root_dir, kit)
        cli_command = CLI_TEST_COMMANDS.get(kit, 'echo "No CLI test command specified"')

        try:
            process = subprocess.Popen(
                cli_command,
                shell=True,
                cwd=kit_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            if process.stdout is None:
                logging.error('Process stdout is None')
                return

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
                self.parse_subtest_results(kit, output, time.time() - start_time, return_code)
            else:
                logging.error(f'CLI test for {kit} failed. Return code: {return_code}')
                logging.error(f'Error output:\n{output}')
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                result = TestResult(
                    kit,
                    'CLI',
                    'FAILED',
                    time.time() - start_time,
                    f'Return code: {return_code}\n{output}',
                    date=current_time,
                    commit_hash=self.commit_hash,
                    commit_url=self.commit_url,
                )
                self.write_test_result(result)

        except subprocess.TimeoutExpired:
            process.kill()
            logging.error(f'CLI test for {kit} timed out after {CLI_COMMAND_TIMEOUT} seconds')
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            result = TestResult(
                kit,
                'CLI',
                'TIMEOUT',
                CLI_COMMAND_TIMEOUT,
                f'Timed out after {CLI_COMMAND_TIMEOUT} seconds',
                date=current_time,
                commit_hash=self.commit_hash,
                commit_url=self.commit_url,
            )
            self.write_test_result(result)

    def parse_subtest_results(self, kit: str, output: str, total_duration: float, return_code: int) -> None:
        """Parse subtest results from CLI output."""
        detailed_tests_found = False  # Flag to track if detailed tests are found
        # Split output into lines
        lines = output.strip().split('\n')
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Use regex to parse the line
            match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (\w+) - (.*)$', line)
            if match:
                date_str = match.group(1)
                log_level = match.group(2)
                message = match.group(3).strip()

                if message.startswith('test_'):
                    # Remove ': PASSED' or ': FAILED' suffix from the test name
                    test_name = re.sub(r':\s*(PASSED|FAILED)$', '', message)

                    # Determine status based on log level
                    status = 'PASSED' if log_level == 'INFO' else 'FAILED'

                    result = TestResult(
                        kit,
                        test_name,
                        status,
                        0.0,
                        '',
                        date_str,
                        commit_hash=self.commit_hash,
                        commit_url=self.commit_url,
                    )
                    self.write_test_result(result)
                    detailed_tests_found = True  # Set flag to True since we found a detailed test
                elif message == f'All CLI tests for {kit} passed':
                    # We have the success message for the kit
                    result = TestResult(
                        kit,
                        'CLI',
                        'PASSED',
                        total_duration,
                        'All CLI tests passed',
                        date_str,
                        commit_hash=self.commit_hash,
                        commit_url=self.commit_url,
                    )
                    self.write_test_result(result)
                    detailed_tests_found = True
            elif f'All CLI tests for {kit} passed' in line:
                # Line did not match the expected format; check for success message
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                result = TestResult(
                    kit,
                    'CLI',
                    'PASSED',
                    total_duration,
                    'All CLI tests passed',
                    current_time,
                    commit_hash=self.commit_hash,
                    commit_url=self.commit_url,
                )
                self.write_test_result(result)
                detailed_tests_found = True
            else:
                # Line did not match the expected format; check for benchmarking success
                if kit == 'benchmarking' and 'Tasks Executed!' in line:
                    # Mark as PASSED
                    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                    result = TestResult(
                        kit,
                        'CLI',
                        'PASSED',
                        total_duration,
                        'Benchmarking CLI test passed.',
                        date_str,
                        commit_hash=self.commit_hash,
                        commit_url=self.commit_url,
                    )
                    self.write_test_result(result)
                    detailed_tests_found = True
                    break  # No need to process further lines
                logging.debug(f'Line did not match pattern: {line}')
                continue

        if not detailed_tests_found:
            # No detailed tests found and no 'All CLI tests for {kit} passed' message
            # For 'benchmarking', if return code is 0, and output indicates success, mark as PASSED
            if kit == 'benchmarking' and return_code == 0:
                # Check for success indicators
                if 'Tasks Executed!' in output or 'Results for token benchmark' in output:
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                    result = TestResult(
                        kit,
                        'CLI',
                        'PASSED',
                        total_duration,
                        'Benchmarking CLI test passed.',
                        current_time,
                        commit_hash=self.commit_hash,
                        commit_url=self.commit_url,
                    )
                    self.write_test_result(result)
                else:
                    # Mark as FAILED
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                    result = TestResult(
                        kit,
                        'CLI',
                        'FAILED',
                        total_duration,
                        'No detailed test results found',
                        current_time,
                        commit_hash=self.commit_hash,
                        commit_url=self.commit_url,
                    )
                    self.write_test_result(result)
            else:
                # Mark as FAILED
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                result = TestResult(
                    kit,
                    'CLI',
                    'FAILED',
                    total_duration,
                    'No detailed test results found',
                    current_time,
                    commit_hash=self.commit_hash,
                    commit_url=self.commit_url,
                )
                self.write_test_result(result)


# Move create_test_methods outside the class
def create_test_methods() -> None:
    """Dynamically create test methods for each starter kit."""
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
        '--env',
        choices=['local', 'docker'],
        default='local',
        help='Specify the test environment (default: local)',
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Run tests in verbose mode')
    parser.add_argument('--skip-streamlit', action='store_true', help='Skip Streamlit tests')
    parser.add_argument('--skip-cli', action='store_true', help='Skip CLI tests')
    parser.add_argument(
        '--recreate-venv',
        action='store_true',
        help='Recreate the virtual environment by running make clean',
    )
    args, unknown = parser.parse_known_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    StarterKitTest.run_streamlit = not args.skip_streamlit
    StarterKitTest.run_cli = not args.skip_cli
    StarterKitTest.env = args.env
    StarterKitTest.recreate_venv = args.recreate_venv

    suite = unittest.TestLoader().loadTestsFromTestCase(StarterKitTest)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)

    # Exit with a non-zero code if any tests failed
    if StarterKitTest.any_test_failed or not result.wasSuccessful():
        sys.exit(1)
