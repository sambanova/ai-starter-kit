#!/usr/bin/env bash
# Ensure this script runs with bash

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