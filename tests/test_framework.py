"""
Test framework for AI Starter Kit.

This module provides a unittest-based framework for testing various starter kits
in both local and Docker environments. It includes functionality to set up the
testing environment, run Streamlit applications, and validate their operation.
It also includes placeholders for CLI/script tests.
"""

import os
import sys
import unittest
import subprocess
import time
from typing import List, Optional, Dict
import requests
import argparse
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# List of starter kits to test
STARTER_KITS: List[str] = [
    'enterprise_knowledge_retriever',
    'function_calling',
    'search_assistant',
    'benchmarking'
]

# Dictionary to store CLI test commands for each kit
CLI_TEST_COMMANDS: Dict[str, str] = {
    # 'enterprise_knowledge_retriever': 'python cli_test.py --mode retrieve',
    # 'function_calling': 'python cli_test.py --function example_function',
    # 'search_assistant': 'python cli_test.py --query "test query"',
    'benchmarking': './run.sh' #This runs the benchmarking suite. 
}

class TestEnvironment:
    """Enum-like class for test environments."""
    LOCAL = 'local'
    DOCKER = 'docker'

class StarterKitTest(unittest.TestCase):
    root_dir: str
    env: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if cls.env == TestEnvironment.LOCAL:
            cls.run_make_clean()
            cls.run_make_all()
            cls.setup_local()
        elif cls.env == TestEnvironment.DOCKER:
            cls.run_make_clean()
            cls.check_docker_daemon()
            cls.setup_docker()

    @classmethod
    def check_docker_daemon(cls) -> None:
        try:
            subprocess.run(['docker', 'info'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            raise unittest.SkipTest("Docker daemon is not running. Skipping Docker tests.")

    @classmethod
    def run_make_clean(cls) -> None:
        logging.info("Running make clean...")
        subprocess.run(['make', 'clean'], cwd=cls.root_dir, check=True, capture_output=False)

    @classmethod
    def run_make_all(cls) -> None:
        logging.info("Running make all...")
        subprocess.run(['make', 'all'], cwd=cls.root_dir, check=True, capture_output=False)

    @classmethod
    def setup_local(cls) -> None:
        cls.activate_venv()

    @classmethod
    def setup_docker(cls) -> None:
        cls.run_docker_build()

    @classmethod
    def run_docker_build(cls) -> None:
        logging.info("Building Docker image...")
        subprocess.run(['make', 'docker-build'], cwd=cls.root_dir, check=True, capture_output=False)

    @classmethod
    def activate_venv(cls) -> None:
        venv_path = os.path.join(cls.root_dir, '.venv')
        activate_this = os.path.join(venv_path, 'bin', 'activate_this.py')
        exec(open(activate_this).read(), {'__file__': activate_this})
        os.environ['PATH'] = f"{venv_path}/bin:{os.environ['PATH']}"

    def run_streamlit_test(self, kit: str) -> None:
        if self.env == TestEnvironment.LOCAL:
            self.run_local_streamlit_test(kit)
        elif self.env == TestEnvironment.DOCKER:
            self.run_docker_streamlit_test(kit)

    def run_local_streamlit_test(self, kit: str) -> None:
        logging.info(f"\nTesting {kit} locally...")
        kit_dir = os.path.join(self.root_dir, kit)
        process = subprocess.Popen(['streamlit', 'run', 'streamlit/app.py', '--browser.gatherUsageStats', 'false'],
                                   cwd=kit_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            logging.info("Waiting for Streamlit to start...")
            time.sleep(25)  # Wait longer for Streamlit to start
            self._check_streamlit_accessibility(kit)
        finally:
            process.terminate()
            process.wait()

    def run_docker_streamlit_test(self, kit: str) -> None:
        logging.info(f"\nTesting {kit} in Docker...")
        container_id = subprocess.check_output([
            'docker', 'run', '-d', 
            '-p', '8501:8501', 
            'ai-starter-kit', 
            'tail', '-f', '/dev/null'
        ]).decode().strip()
        logging.info(f"Started container: {container_id}")
        
        try:
            logging.info("Waiting for container to start...")
            time.sleep(15)
            
            logging.info("Checking container status...")
            container_status = subprocess.check_output(['docker', 'inspect', '-f', '{{.State.Status}}', container_id]).decode().strip()
            logging.info(f"Container status: {container_status}")
            self.assertEqual(container_status, 'running', f"Container for {kit} is not running")
            
            logging.info(f"Running Streamlit for {kit}...")
            streamlit_cmd = f"cd /app/{kit} && streamlit run streamlit/app.py --browser.gatherUsageStats false"
            subprocess.run(['docker', 'exec', '-d', container_id, 'sh', '-c', streamlit_cmd], check=True)
            
            logging.info("Waiting for Streamlit to start...")
            time.sleep(25)
            
            self._check_streamlit_accessibility(kit)
            
        finally:
            logging.info(f"Stopping and removing container {container_id}")
            subprocess.run(['docker', 'stop', container_id], check=True)
            subprocess.run(['docker', 'rm', container_id], check=True)

    def _check_streamlit_accessibility(self, kit: str) -> None:
        logging.info("Checking Streamlit accessibility...")
        for attempt in range(3):
            try:
                response = requests.get('http://localhost:8501')
                self.assertEqual(response.status_code, 200, f"Streamlit app for {kit} is not running")
                logging.info(f"Streamlit app for {kit} is running successfully")
                return
            except requests.ConnectionError:
                logging.info(f"Attempt {attempt + 1}: Streamlit not accessible yet. Waiting...")
                time.sleep(5)
        
        logging.error(f"Streamlit app for {kit} failed to start after 3 attempts")
        self.fail(f"Streamlit app for {kit} failed to start after 3 attempts")

    def run_cli_test(self, kit: str) -> None:
        if self.env == TestEnvironment.LOCAL:
            self.run_local_cli_test(kit)
        elif self.env == TestEnvironment.DOCKER:
            self.run_docker_cli_test(kit)

    def run_local_cli_test(self, kit: str) -> None:
        logging.info(f"\nRunning CLI test for {kit} locally...")
        kit_dir = os.path.join(self.root_dir, kit)
        
        # Get the CLI test command for this kit
        cli_command = CLI_TEST_COMMANDS.get(kit, 'echo "No CLI test command specified"')
        
        # Placeholder: Run the CLI test command 
        result = subprocess.run(cli_command, shell=True, cwd=kit_dir, capture_output=False)
        self.assertEqual(result.returncode, 0, f"CLI test for {kit} failed")
        logging.info(f"CLI test output for {kit}:\n{result.stdout}")

        # Placeholder: Always pass
        logging.info(f"CLI test for {kit} passed (placeholder)")
        pass

    def run_docker_cli_test(self, kit: str) -> None:
        logging.info(f"\nRunning CLI test for {kit} in Docker...")
        container_id = subprocess.check_output([
            'docker', 'run', '-d', 
            'ai-starter-kit', 
            'tail', '-f', '/dev/null'
        ]).decode().strip()
        
        try:
            # Get the CLI test command for this kit
            cli_command = CLI_TEST_COMMANDS.get(kit, 'echo "No CLI test command specified"')
            docker_cli_command = f"cd /app/{kit} && {cli_command}"
            
            # Placeholder: Run the CLI test command in Docker 
            result = subprocess.run(['docker', 'exec', container_id, 'sh', '-c', docker_cli_command], 
                                    capture_output=False)
            self.assertEqual(result.returncode, 0, f"Docker CLI test for {kit} failed")
            logging.info(f"Docker CLI test output for {kit}:\n{result.stdout}")

            # Placeholder: Always pass
            logging.info(f"Docker CLI test for {kit} passed (placeholder)")
            pass
            
        finally:
            subprocess.run(['docker', 'stop', container_id], check=True)
            subprocess.run(['docker', 'rm', container_id], check=True)

def create_test_methods() -> None:
    for kit in STARTER_KITS:
        def test_kit(self, kit=kit):
            self.run_streamlit_test(kit)
            self.run_cli_test(kit)
        test_kit.__name__ = f'test_{kit}'
        setattr(StarterKitTest, test_kit.__name__, test_kit)

create_test_methods()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tests for AI Starter Kit')
    parser.add_argument('--env', choices=['local', 'docker', 'all'], default='all',
                        help='Specify the test environment (default: all)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Run tests in verbose mode')
    args, unknown = parser.parse_known_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.env in ['local', 'all']:
        logging.info("Running local tests...")
        StarterKitTest.env = TestEnvironment.LOCAL
        local_suite = unittest.TestLoader().loadTestsFromTestCase(StarterKitTest)
        unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(local_suite)

    if args.env in ['docker', 'all']:
        logging.info("\nRunning Docker tests...")
        StarterKitTest.env = TestEnvironment.DOCKER
        docker_suite = unittest.TestLoader().loadTestsFromTestCase(StarterKitTest)
        unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(docker_suite)