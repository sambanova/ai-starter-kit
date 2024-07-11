#!/usr/bin/env bash
# Ensure this script runs with bash

# Change to the directory containing this script
cd "$(dirname "$0")"

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

# Check if pyenv is installed and set it up if it exists
if command -v pyenv &> /dev/null; then
    echo "pyenv found. Setting up environment..."
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
else
    echo "pyenv not found. Please install pyenv and run this script again."
    echo "You can install pyenv by following the instructions at: https://github.com/pyenv/pyenv#installation"
    exit 1
fi

# Set up and activate the test suite environment
make setup-test-suite
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

# Run tests based on the environment
echo "Running $env tests..."
if [ "$verbose" == "true" ]; then
    python tests/test_framework.py --env "$env" --verbose
else
    python tests/test_framework.py --env "$env"
fi

# Deactivate the test suite environment
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

# Clean up the test suite environment
make clean-test-suite