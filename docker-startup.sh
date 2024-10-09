#!/bin/bash
set -e

# Source the environment variables if .env file is provided
if [ -f "/.env" ]; then
    set -a
    source /.env
    set +a
fi

# Check if we're in PROD_MODE or TEST_MODE
if [ "${PROD_MODE,,}" = "true" ]; then
    echo "Running in PROD_MODE, updating configs..."
    python3 /app/utils/prod/update_config.py prod
elif [ "${TEST_MODE,,}" = "true" ]; then
    echo "Running in TEST_MODE, updating configs..."
    python3 /app/utils/prod/update_config.py test
else
    echo "Neither PROD_MODE nor TEST_MODE is set to true. Skipping config updates."
    echo "PROD_MODE: $PROD_MODE"
    echo "TEST_MODE: $TEST_MODE"
fi

# Start the parsing service if needed
if [ "${START_PARSING_SERVICE,,}" = "true" ]; then
    cd /app/utils/parsing/unstructured-api
    make run-web-app &
    cd /app
fi

# Execute the command passed to docker run
exec "$@"