#!/usr/bin/env bash
# Ensure this script runs with bash

# Usage:
#   ./run_tests.sh [environment] [options]
#
# Environments:
#   local   - Run tests in local environment
#   docker  - Run tests in Docker environment
#   all     - Run tests in both local and Docker environments (default)
#
# Options:
#   -v, --verbose       Enable verbose output
#   --skip-streamlit    Skip Streamlit tests
#   --skip-cli          Skip CLI tests
#   -h, --help          Display this help message
#
# Examples:
#   ./run_tests.sh local --verbose
#   ./run_tests.sh docker --skip-streamlit
#   ./run_tests.sh all --skip-cli
#   ./run_tests.sh local --verbose --skip-streamlit --skip-cli

# Change to the directory containing this script
cd "$(dirname "$0")"

# Run Make-Clean
make clean-test-suite

# Deactivate any active virtual environment or conda environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating virtual environment: $VIRTUAL_ENV"
    deactivate
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Deactivating conda environment: $CONDA_DEFAULT_ENV"
    conda deactivate
fi

# Unset environment variables
unset VIRTUAL_ENV
unset CONDA_DEFAULT_ENV

# Ensure pyenv is installed and set up
if ! make ensure-pyenv; then
    echo "Error: Failed to set up pyenv. Exiting."
    exit 1
fi

# Set up and activate the test suite environment
if ! make setup-test-suite; then
    echo "Error: Failed to set up test suite environment. Exiting."
    exit 1
fi

if [ -f .test_suite_venv/bin/activate ]; then
    . .test_suite_venv/bin/activate
else
    echo "Error: Virtual environment not created. Check the setup-test-suite make target."
    exit 1
fi

# Add the current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Parse command line arguments
env="all"
verbose="false"
skip_streamlit="false"
skip_cli="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        local|docker|all)
            env="$1"
            shift
            ;;
        -v|--verbose)
            verbose="true"
            shift
            ;;
        --skip-streamlit)
            skip_streamlit="true"
            shift
            ;;
        --skip-cli)
            skip_cli="true"
            shift
            ;;
        -h|--help)
            # Display usage information
            sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Construct the test command
test_command="python tests/test_framework.py --env $env"

if [ "$verbose" == "true" ]; then
    test_command+=" --verbose"
fi

if [ "$skip_streamlit" == "true" ]; then
    test_command+=" --skip-streamlit"
fi

if [ "$skip_cli" == "true" ]; then
    test_command+=" --skip-cli"
fi

# Run tests
echo "Running tests with command: $test_command"
eval $test_command

# Deactivate the test suite environment
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

# Clean up the test suite environment
make clean-test-suite