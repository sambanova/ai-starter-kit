#!/bin/bash

# AI Starter Kit Test Runner
# --------------------------
#
# This script runs tests for the AI Starter Kit project. It can run tests in local
# environment, Docker environment, or both. It also supports verbose mode for
# more detailed output.
#
# Usage:
#   ./run_tests.sh [environment] [options]
#
# Environment:
#   local   - Run tests in local environment only
#   docker  - Run tests in Docker environment only
#   all     - Run tests in both local and Docker environments (default)
#
# Options:
#   -v, --verbose  Run tests in verbose mode
#
# Examples:
#   ./run_tests.sh                  # Run all tests
#   ./run_tests.sh local            # Run local tests with make only
#   ./run_tests.sh docker -v        # Run Docker tests in verbose mode
#   ./run_tests.sh all --verbose    # Run all tests in verbose mode
#
# Note: Make sure you have the necessary dependencies installed and
# Docker running (if testing Docker environment) before running this script.

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
run_tests "$env" "$verbose"