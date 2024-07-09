#!/bin/bash

# Change to the directory containing this script
cd "$(dirname "$0")"

# Deactivate any active virtual environment in a subshell
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating virtual environment: $VIRTUAL_ENV"
    (
        deactivate
    )
fi

# Unset VIRTUAL_ENV to ensure it's not carried over
unset VIRTUAL_ENV

# Add the current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Function to run tests
run_tests() {
    local env=$1
    local verbose=$2
    if [ "$verbose" == "true" ]; then
        python -m tests.test_framework --env "$env" --verbose
    else
        python -m tests.test_framework --env "$env"
    fi
}

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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run tests based on the environment
echo "Running $env tests..."
run_tests "$env" "$verbose"