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

set -e

# Change to the directory containing this script
cd "$(dirname "$0")"

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

# Load and clean environment variables from .env file
if [ -f ".env" ]; then
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Trim leading and trailing whitespace from key and value
        key=$(echo "$key" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        value=$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        
        # Skip comments and empty lines
        if [[ ! $key =~ ^# && -n $key ]]; then
            # Export the cleaned variable
            export "$key=$value"
        fi
    done < .env
fi

# Construct the test command
test_options=""
[ "$verbose" == "true" ] && test_options+=" --verbose"
[ "$skip_streamlit" == "true" ] && test_options+=" --skip-streamlit"
[ "$skip_cli" == "true" ] && test_options+=" --skip-cli"

run_local_tests() {
    echo "Running local tests..."
    make clean-test-suite
    make setup-test-suite
    #. .venv/bin/activate
    python tests/test_framework.py --env local $test_options
    deactivate
}

run_docker_tests() {
    echo "Building Docker image for testing..."
    make docker-build

    echo "Running Docker tests..."
    # Create a temporary file with cleaned environment variables
    temp_env_file=$(mktemp)
    env | grep -v '^_' > "$temp_env_file"

    # Pass cleaned environment variables to Docker
    docker run --rm \
        --env-file "$temp_env_file" \
        -e DOCKER_ENV=true \
        -v $(pwd)/test_results:/app/test_results \
        -w /app \
        "${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${DOCKER_TAG}" \
        python tests/test_framework.py --env docker $test_options

    # Remove the temporary file
    rm "$temp_env_file"
}

case $env in
    local)
        run_local_tests
        ;;
    docker)
        run_docker_tests
        ;;
    all)
        run_local_tests
        run_docker_tests
        ;;
esac